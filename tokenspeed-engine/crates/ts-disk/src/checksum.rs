// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Hardware-accelerated CRC32C (Castagnoli) checksums and an in-memory table
//! keyed by `(file_offset, len)` for end-to-end chunk integrity verification.
//!
//! Storing checksums in host memory avoids changing the on-disk block layout
//! or adding space overhead to `O_DIRECT` aligned slots while catching silent
//! bit-flips, torn writes, and misdirected NVMe reads before corrupted KV
//! blocks can poison model attention.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
};

use parking_lot::Mutex;

/// Reflected Castagnoli polynomial (RFC 3720 / iSCSI / NVMe).
const CRC32C_POLY: u32 = 0x82F6_3B78;

const fn make_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32C_POLY;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const CRC32C_TABLE: [u32; 256] = make_crc32c_table();

/// Portable table-driven CRC32C fallback.
fn crc32c_sw(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[idx];
    }
    !crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(bytes: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};

    let mut crc = u64::from(!0u32);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        crc = _mm_crc32_u64(crc, word);
    }
    let mut crc32 = crc as u32;
    for &byte in chunks.remainder() {
        crc32 = _mm_crc32_u8(crc32, byte);
    }
    !crc32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_arm(bytes: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};

    let mut crc = !0u32;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        crc = __crc32cd(crc, word);
    }
    for &byte in chunks.remainder() {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

/// Compute the Castagnoli CRC32C of `bytes`, using hardware instructions
/// (x86_64 SSE4.2 or aarch64 CRC) when available.
pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            // SAFETY: SSE4.2 support was verified at runtime above.
            return unsafe { crc32c_sse42(bytes) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            // SAFETY: ARMv8 CRC extension support was verified at runtime above.
            return unsafe { crc32c_arm(bytes) };
        }
    }
    crc32c_sw(bytes)
}

/// In-memory map of `(file_offset, len) -> crc32c` for chunks written through
/// this executor.
pub(crate) struct ChecksumTable {
    enabled: AtomicBool,
    entries: Mutex<HashMap<(u64, usize), u32>>,
}

impl Default for ChecksumTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ChecksumTable {
    pub(crate) fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Record the checksum of a chunk about to be written.
    pub(crate) fn record(&self, file_offset: u64, bytes: &[u8]) {
        if !self.enabled() {
            return;
        }
        let crc = crc32c(bytes);
        self.entries.lock().insert((file_offset, bytes.len()), crc);
    }

    /// Drop any recorded checksum for a chunk whose write failed.
    pub(crate) fn invalidate(&self, file_offset: u64, len: usize) {
        if !self.enabled() {
            return;
        }
        self.entries.lock().remove(&(file_offset, len));
    }

    /// Verify a completed read against the checksum recorded on write.
    ///
    /// Unrecorded ranges (e.g. direct reads of pre-populated test files) pass
    /// through without error.
    pub(crate) fn verify(&self, file_offset: u64, bytes: &[u8]) -> Result<(), (u32, u32)> {
        if !self.enabled() {
            return Ok(());
        }
        let expected = {
            let entries = self.entries.lock();
            entries.get(&(file_offset, bytes.len())).copied()
        };
        if let Some(expected) = expected {
            let actual = crc32c(bytes);
            if actual != expected {
                return Err((expected, actual));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3720_check_value() {
        // RFC 3720 Section B.4 standard CRC32C check value for "123456789".
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c_sw(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c_sw(b""), 0);
    }

    #[test]
    fn hardware_matches_software_across_lengths() {
        let mut buf = vec![0u8; 4099];
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = ((i * 31 + 17) & 0xFF) as u8;
        }
        for len in [0, 1, 3, 7, 8, 9, 15, 16, 63, 64, 511, 512, 4096, 4099] {
            assert_eq!(
                crc32c(&buf[..len]),
                crc32c_sw(&buf[..len]),
                "mismatch at len={len}"
            );
        }
    }

    #[test]
    fn table_detects_single_bit_flip_and_respects_toggle() {
        let table = ChecksumTable::new();
        let mut data = vec![0xA5u8; 4096];
        table.record(8192, &data);
        assert_eq!(table.verify(8192, &data), Ok(()));
        // Unrecorded offset passes through.
        assert_eq!(table.verify(16384, &data), Ok(()));

        // Flip a single bit.
        data[1337] ^= 0x01;
        let err = table.verify(8192, &data).unwrap_err();
        assert_ne!(err.0, err.1);

        // Disabling verification bypasses the mismatch.
        table.set_enabled(false);
        assert_eq!(table.verify(8192, &data), Ok(()));

        // Invalidate removes the entry once re-enabled.
        table.set_enabled(true);
        table.invalidate(8192, data.len());
        assert_eq!(table.verify(8192, &data), Ok(()));
    }
}
