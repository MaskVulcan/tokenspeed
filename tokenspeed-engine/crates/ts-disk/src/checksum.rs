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

//! Hardware-accelerated CRC32C (Castagnoli) checksums and an in-memory range
//! table keyed by logical `Io` `(file_offset, len)` for end-to-end integrity
//! verification independent of `Pieces::split` fixed-buffer boundaries.
//!
//! Storing checksums in host memory avoids changing the on-disk block layout
//! or adding space overhead to `O_DIRECT` aligned slots while catching silent
//! bit-flips, torn writes, and misdirected NVMe reads before corrupted KV
//! blocks can poison model attention.

use std::{
    collections::BTreeMap,
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

fn gf2_matrix_times(mat: &[u32; 32], mut vec: u32) -> u32 {
    let mut sum = 0u32;
    let mut idx = 0;
    while vec != 0 {
        if vec & 1 != 0 {
            sum ^= mat[idx];
        }
        vec >>= 1;
        idx += 1;
    }
    sum
}

fn gf2_matrix_square(square: &mut [u32; 32], mat: &[u32; 32]) {
    for n in 0..32 {
        square[n] = gf2_matrix_times(mat, mat[n]);
    }
}

/// Combine CRC32C of two contiguous byte slices:
/// `crc32c(a || b) == crc32c_combine(crc32c(a), crc32c(b), b.len())`.
pub(crate) fn crc32c_combine(mut crc1: u32, crc2: u32, mut len2: usize) -> u32 {
    if len2 == 0 {
        return crc1;
    }
    let mut odd = [0u32; 32];
    let mut even = [0u32; 32];
    // Operator for one zero bit in the reflected Castagnoli polynomial.
    odd[0] = CRC32C_POLY;
    let mut row = 1u32;
    for item in &mut odd[1..] {
        *item = row;
        row <<= 1;
    }
    // Operator for two zero bits, then four zero bits.
    gf2_matrix_square(&mut even, &odd);
    gf2_matrix_square(&mut odd, &even);
    // Apply `len2` zero bytes to `crc1`.
    loop {
        gf2_matrix_square(&mut even, &odd);
        if len2 & 1 != 0 {
            crc1 = gf2_matrix_times(&even, crc1);
        }
        len2 >>= 1;
        if len2 == 0 {
            break;
        }
        gf2_matrix_square(&mut odd, &even);
        if len2 & 1 != 0 {
            crc1 = gf2_matrix_times(&odd, crc1);
        }
        len2 >>= 1;
        if len2 == 0 {
            break;
        }
    }
    crc1 ^ crc2
}

/// In-memory map of non-overlapping file ranges `file_offset -> (len, crc32c)`
/// for logical `Io` ranges written through this executor.
pub(crate) struct ChecksumTable {
    enabled: AtomicBool,
    entries: Mutex<BTreeMap<u64, (usize, u32)>>,
}

impl Default for ChecksumTable {
    fn default() -> Self {
        Self::new()
    }
}

fn invalidate_range_locked(entries: &mut BTreeMap<u64, (usize, u32)>, file_offset: u64, len: usize) {
    if len == 0 {
        return;
    }
    let end = file_offset.saturating_add(len as u64);
    if let Some((&prev_start, &(prev_len, _))) = entries.range(..file_offset).next_back() {
        if prev_start.saturating_add(prev_len as u64) > file_offset {
            entries.remove(&prev_start);
        }
    }
    let overlapping: Vec<u64> = entries.range(file_offset..end).map(|(&k, _)| k).collect();
    for key in overlapping {
        entries.remove(&key);
    }
}

impl ChecksumTable {
    pub(crate) fn new() -> Self {
        Self {
            enabled: AtomicBool::new(true),
            entries: Mutex::new(BTreeMap::new()),
        }
    }

    pub(crate) fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Record a precomputed CRC32C for `[file_offset, file_offset + len)`,
    /// evicting any previously overlapping entries so slot reuse across block
    /// sizes never leaves stale checksums behind.
    pub(crate) fn record_crc(&self, file_offset: u64, len: usize, crc: u32) {
        if !self.enabled() || len == 0 {
            return;
        }
        let mut entries = self.entries.lock();
        invalidate_range_locked(&mut entries, file_offset, len);
        entries.insert(file_offset, (len, crc));
    }

    /// Record the checksum of a range about to be or just written.
    #[cfg(test)]
    pub(crate) fn record(&self, file_offset: u64, bytes: &[u8]) {
        if !self.enabled() || bytes.is_empty() {
            return;
        }
        let crc = crc32c(bytes);
        self.record_crc(file_offset, bytes.len(), crc);
    }

    /// Drop any recorded checksums overlapping `[file_offset, file_offset + len)`.
    pub(crate) fn invalidate(&self, file_offset: u64, len: usize) {
        if len == 0 {
            return;
        }
        let mut entries = self.entries.lock();
        invalidate_range_locked(&mut entries, file_offset, len);
    }

    /// Verify a precomputed read CRC32C against the checksum recorded on write.
    ///
    /// Unrecorded ranges (e.g. direct reads of pre-populated test files) pass
    /// through without error.
    pub(crate) fn verify_crc(
        &self,
        file_offset: u64,
        len: usize,
        actual: u32,
    ) -> Result<(), (u32, u32)> {
        if !self.enabled() {
            return Ok(());
        }
        let expected = {
            let entries = self.entries.lock();
            entries
                .get(&file_offset)
                .filter(|&&(recorded_len, _)| recorded_len == len)
                .map(|&(_, crc)| crc)
        };
        if let Some(expected) = expected {
            if actual != expected {
                return Err((expected, actual));
            }
        }
        Ok(())
    }

    /// Verify a completed read slice against the checksum recorded on write.
    #[cfg(test)]
    pub(crate) fn verify(&self, file_offset: u64, bytes: &[u8]) -> Result<(), (u32, u32)> {
        if !self.enabled() {
            return Ok(());
        }
        let expected = {
            let entries = self.entries.lock();
            entries
                .get(&file_offset)
                .filter(|&&(recorded_len, _)| recorded_len == bytes.len())
                .map(|&(_, crc)| crc)
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

    #[test]
    fn crc32c_combine_matches_whole_buffer_across_splits() {
        let buf: Vec<u8> = (0..16_384)
            .map(|i| ((i * 73 + 19) & 0xFF) as u8)
            .collect();
        let full = crc32c(&buf);
        for split in [0, 1, 7, 13, 64, 4095, 4096, 4097, 8192, 12288, 16_383, 16_384] {
            let combined = crc32c_combine(crc32c(&buf[..split]), crc32c(&buf[split..]), buf.len() - split);
            assert_eq!(combined, full, "combine mismatch at split={split}");
        }
        // Three-way split.
        let c0 = crc32c(&buf[..4096]);
        let c1 = crc32c(&buf[4096..12288]);
        let c2 = crc32c(&buf[12288..]);
        let c01 = crc32c_combine(c0, c1, 8192);
        assert_eq!(crc32c_combine(c01, c2, 4096), full);
    }

    #[test]
    fn table_invalidates_overlapping_ranges_on_different_size_overwrite() {
        let table = ChecksumTable::new();
        let a = vec![0x11u8; 4096];
        let b = vec![0x22u8; 4096];
        let ab_new = vec![0x33u8; 8192];
        table.record(0, &a);
        table.record(4096, &b);
        assert_eq!(table.verify(0, &a), Ok(()));
        assert_eq!(table.verify(4096, &b), Ok(()));

        // Overwriting [0, 8192) removes both [0, 4096) and [4096, 8192).
        table.record(0, &ab_new);
        assert_eq!(table.verify(0, &ab_new), Ok(()));
        // Old sub-range checksums are gone (no false mismatch against stale CRC).
        let stale_probe = vec![0x99u8; 4096];
        assert_eq!(table.verify(0, &stale_probe), Ok(()));
        assert_eq!(table.verify(4096, &stale_probe), Ok(()));

        // Overwriting the second half [4096, 8192) evicts the enclosing [0, 8192).
        table.record(4096, &b);
        assert_eq!(table.verify(4096, &b), Ok(()));
        let whole_probe = vec![0x77u8; 8192];
        assert_eq!(table.verify(0, &whole_probe), Ok(()));
    }
}
