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

//! Level-A 0-Bit Bitwise, Chaos Fault-Injection, AIMD, and Throughput Stress
//! Suite for `ts-disk` (LongCat Lite 17B exact 4 KiB-aligned geometry):
//! - **TS-A1 (`D1-01A`, `D1-05`)**: 2,100+ clean -> poison(`0xDEADBEEF`) ->
//!   read -> overwrite -> reverse-split read cycles across 1 GiB `PIECE_BYTES`
//!   boundaries under direct `io_uring`, `EFAULT`-triggered bounce-pool
//!   `io_uring`, and bounced `psync`.
//! - **TS-A2 (`D4-03`)**: 100% single-bit silent on-disk corruption catch rate
//!   across single-chunk MLA, multi-chunk GF(2) `crc32c_combine` KDA, and
//!   straddling 1 GiB boundary offsets for all three modes.
//! - **TS-A4 (`D4-01`)**: `ReadLatencyAimd` step-down (`8->4->2->1->0`, `1`
//!   when idle) + additive recovery (`+1/100ms -> 8`) + live 16-rank NVMe
//!   contention stress.
//! - **TS-A5 (`D5-01`)**: `O_DIRECT` fixed-buffer direct vs fixed-buffer bounce
//!   pool vs `psync` throughput & CPU overhead micro-benchmark (`848 KiB`,
//!   `4 MiB`, `24.84 MiB`).

#![allow(unsafe_code)]

use std::{
    fs::OpenOptions,
    os::unix::{fs::OpenOptionsExt, io::AsRawFd},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use ts_disk::{
    Backend, Direction, DiskFile, Executor, ExecutorConfig, Io, IoError, Job, JobReport, Memory,
    ReadLatencyAimd, AIMD_MIN_SAMPLES, ALIGNMENT,
};

/// Exact LongCat Lite 17B 4 KiB-aligned LCM parent size (`24.84 MiB`).
const PARENT_BYTES: usize = 26_050_560; // 6,360 * 4,096
/// Exact LongCat Lite 17B `full_attention` MLA block (`848 KiB`, 30 per parent).
const MLA_BLOCK_BYTES: usize = 868_352; // 212 * 4,096
/// Exact LongCat Lite 17B `linear_attention` KDA state block (`24.84 MiB`, 1 per parent).
const KDA_BLOCK_BYTES: usize = PARENT_BYTES;
/// `ts-disk` fixed-buffer registration piece size (`1 GiB`).
const PIECE_BYTES: usize = 1 << 30;
/// Poison sentinel written into destination buffers before every Disk read.
const POISON_SENTINEL: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
/// Maximum I/O chunk size matching production `DiskTier` (`4 MiB`).
const MAX_IO_BYTES: usize = 4 << 20;

const _: () = {
    assert!(PARENT_BYTES % ALIGNMENT == 0);
    assert!(MLA_BLOCK_BYTES % ALIGNMENT == 0);
    assert!(MLA_BLOCK_BYTES * 30 == PARENT_BYTES);
    // Parent 41 straddles the 1 GiB PIECE_BYTES boundary:
    //   parent 41 start = 1,068,072,960 (< 1 GiB)
    //   parent 41 end   = 1,094,123,520 (> 1 GiB)
    assert!(41 * PARENT_BYTES < PIECE_BYTES);
    assert!(42 * PARENT_BYTES > PIECE_BYTES);
    // MLA sub-block 6 inside parent 41 straddles the 1 GiB PIECE_BYTES boundary:
    //   1,068,072,960 + 6 * 868,352 = 1,073,283,072 (< 1 GiB)
    //   1,073,283,072 + 868,352     = 1,074,151,424 (> 1 GiB)
    assert!(41 * PARENT_BYTES + 6 * MLA_BLOCK_BYTES < PIECE_BYTES);
    assert!(41 * PARENT_BYTES + 7 * MLA_BLOCK_BYTES > PIECE_BYTES);
};

// ---------------------------------------------------------------------------
// Exact RFC 6234 SHA-256 (self-contained, verified against NIST vectors)
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline(always)]
fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let chunks = data.chunks_exact(64);
    let rem = chunks.remainder();
    for chunk in chunks {
        let block: &[u8; 64] = chunk.try_into().unwrap();
        sha256_compress(&mut state, block);
    }
    let mut pad = [0u8; 128];
    pad[..rem.len()].copy_from_slice(rem);
    pad[rem.len()] = 0x80;
    let total_bits = (data.len() as u64) * 8;
    let pad_len = if rem.len() < 56 { 64 } else { 128 };
    pad[pad_len - 8..pad_len].copy_from_slice(&total_bits.to_be_bytes());
    for chunk in pad[..pad_len].chunks_exact(64) {
        let block: &[u8; 64] = chunk.try_into().unwrap();
        sha256_compress(&mut state, block);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn verify_sha256_vectors() {
    let abc = sha256(b"abc");
    let expected: [u8; 32] = [
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad,
    ];
    assert_eq!(abc, expected, "SHA-256 self-check failed");
}

// ---------------------------------------------------------------------------
// Scratch files, guard mappings, deterministic patterns, and CPU accounting
// ---------------------------------------------------------------------------

struct ScratchFile(PathBuf);

impl ScratchFile {
    fn new(name: &str) -> Self {
        let dir = std::env::var_os("TS_DISK_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        Self(dir.join(format!(
            "ts-disk-stress-{}-{seq}-{name}.bin",
            std::process::id()
        )))
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// 1-page `PROT_READ` anonymous mapping passed alongside read/write regions to
/// `Executor::start` to trigger a REAL kernel `EFAULT` (os error 14) on
/// `io_uring_register_buffers`, forcing `BufferMode::Bounce` exactly as on
/// Ascend 910B2C `torch_npu` `VM_PFNMAP|VM_IO` memory.
struct ReadOnlyEfaultGuard {
    ptr: *mut libc::c_void,
    len: usize,
}

impl ReadOnlyEfaultGuard {
    fn new() -> Self {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ALIGNMENT,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "PROT_READ guard mmap failed");
        Self {
            ptr,
            len: ALIGNMENT,
        }
    }

    fn region(&self) -> (usize, usize) {
        (self.ptr as usize, self.len)
    }
}

impl Drop for ReadOnlyEfaultGuard {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

/// Fast deterministic non-repeating 64-bit SplitMix64 stream (avoids accidental
/// alignment-period symmetries that could mask offset/aliasing bugs).
fn fill_seeded(buf: &mut [u8], seed: u64) {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xd1b5_4a32_d192_ed03;
    let mut chunks = buf.chunks_exact_mut(8);
    for chunk in &mut chunks {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        rem.copy_from_slice(&x.to_le_bytes()[..rem.len()]);
    }
}

fn fill_poison(mem: &mut Memory, offset: usize, len: usize) {
    let mut buf = vec![0u8; len];
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&POISON_SENTINEL);
    }
    mem.write_at(offset, &buf);
}

fn write_region_bytes(mem: &mut Memory, offset: usize, data: &[u8]) {
    mem.write_at(offset, data);
}

fn read_region_bytes(mem: &Memory, offset: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    mem.read_at(offset, &mut out);
    out
}

fn run_job(executor: &Executor, job: Job) -> Result<JobReport, IoError> {
    let (tx, rx) = mpsc::channel();
    executor
        .submit(job, move |report: JobReport| {
            let _ = tx.send(report);
        })
        .expect("submit valid job");
    let report = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("disk job timed out");
    match report.outcome {
        Ok(()) => Ok(report),
        Err(err) => Err(err),
    }
}

/// Directly flip 1 bit at `byte_offset` inside `path` via an independent
/// `O_DIRECT` file descriptor (simulating silent NVMe bit-rot bypassing the
/// executor's checksum table).
fn flip_single_bit_on_disk(path: &Path, byte_offset: u64, bit: u8) {
    assert!(bit < 8);
    let sector_offset = (byte_offset / ALIGNMENT as u64) * ALIGNMENT as u64;
    let within_sector = (byte_offset - sector_offset) as usize;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .expect("open disk file for direct bit-flip");
    let mut sector = Memory::map(ALIGNMENT).expect("aligned sector buffer");
    let nread = unsafe {
        libc::pread(
            file.as_raw_fd(),
            sector.address() as *mut libc::c_void,
            ALIGNMENT,
            sector_offset as libc::off_t,
        )
    };
    assert_eq!(nread as usize, ALIGNMENT, "pread sector failed");
    let mut bytes = vec![0u8; ALIGNMENT];
    sector.read_at(0, &mut bytes);
    bytes[within_sector] ^= 1u8 << bit;
    sector.write_at(0, &bytes);
    let nwrite = unsafe {
        libc::pwrite(
            file.as_raw_fd(),
            sector.address() as *const libc::c_void,
            ALIGNMENT,
            sector_offset as libc::off_t,
        )
    };
    assert_eq!(nwrite as usize, ALIGNMENT, "pwrite sector failed");
}

fn process_cpu_seconds() -> (f64, f64) {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(rc, 0, "getrusage failed");
    let user = usage.ru_utime.tv_sec as f64 + usage.ru_utime.tv_usec as f64 / 1e6;
    let sys = usage.ru_stime.tv_sec as f64 + usage.ru_stime.tv_usec as f64 / 1e6;
    (user, sys)
}

#[derive(Clone, Copy, Debug)]
enum ExecMode {
    IoUringDirect,
    IoUringEfaultBounce,
    PsyncBounce,
}

impl ExecMode {
    fn label(self) -> &'static str {
        match self {
            Self::IoUringDirect => "io_uring_direct_fixed",
            Self::IoUringEfaultBounce => "io_uring_efault_bounce_pool",
            Self::PsyncBounce => "psync_bounce_pool",
        }
    }
}

fn start_mode_executor(
    mode: ExecMode,
    scratch: &ScratchFile,
    file_bytes: u64,
    host: &Memory,
    push_ring: &Memory,
    ro_guard: &ReadOnlyEfaultGuard,
) -> Executor {
    let file = DiskFile::create(&scratch.0, file_bytes).expect("create O_DIRECT disk file");
    let cfg = ExecutorConfig {
        backend: match mode {
            ExecMode::IoUringDirect | ExecMode::IoUringEfaultBounce => Backend::IoUring,
            ExecMode::PsyncBounce => Backend::Psync,
        },
        queue_depth: 16,
        write_queue_depth: 8,
        max_io_bytes: MAX_IO_BYTES,
        write_bytes_per_second: u64::MAX / 4,
    };
    let executor = match mode {
        ExecMode::IoUringDirect => {
            let ex = Executor::start(
                file,
                &[
                    (host.address(), host.size()),
                    (push_ring.address(), push_ring.size()),
                ],
                cfg,
            )
            .expect("start io_uring direct executor");
            assert!(
                !ex.uses_bounce_pool(),
                "expected fixed-buffer direct io_uring without bounce pool"
            );
            ex
        }
        ExecMode::IoUringEfaultBounce => {
            // Passing the 1-page PROT_READ guard triggers real kernel EFAULT on
            // io_uring_register_buffers -> production BufferMode::Bounce.
            let ex = Executor::start(
                file,
                &[
                    (host.address(), host.size()),
                    (push_ring.address(), push_ring.size()),
                    ro_guard.region(),
                ],
                cfg,
            )
            .expect("start io_uring EFAULT bounce-pool executor");
            assert!(
                ex.uses_bounce_pool(),
                "expected EFAULT fallback to activate production io_uring bounce pool"
            );
            ex
        }
        ExecMode::PsyncBounce => {
            let ex = Executor::start_bounced(
                file,
                &[
                    (host.address(), host.size()),
                    (push_ring.address(), push_ring.size()),
                ],
                cfg,
            )
            .expect("start bounced psync executor");
            assert!(
                ex.uses_bounce_pool(),
                "expected bounced psync executor to report uses_bounce_pool() == true"
            );
            ex
        }
    };
    executor
}

// ---------------------------------------------------------------------------
// TS-A1: Level-A 0-Bit Bitwise Round-Trip Across 1 GiB PIECE_BYTES & Bounce Pool
// ---------------------------------------------------------------------------

struct ModeBitwiseSummary {
    mode: &'static str,
    bounce_active: bool,
    mla_cycles: usize,
    kda_cycles: usize,
    total_writes: usize,
    total_reads: usize,
    total_bytes_verified: u64,
    straddling_w_to_non_straddling_r: usize,
    non_straddling_w_to_straddling_r: usize,
    sha256_mismatches: usize,
    memcmp_mismatches: usize,
    sentinel_bleeds: usize,
    neighbor_corruptions: usize,
    spurious_checksum_mismatches: usize,
    elapsed: Duration,
}

fn run_ts_a1_for_mode(
    mode: ExecMode,
    mla_cycles: usize,
    kda_cycles: usize,
    host: &mut Memory,
    push_ring: &mut Memory,
    ro_guard: &ReadOnlyEfaultGuard,
) -> ModeBitwiseSummary {
    // 16 Disk parents (416.8 MiB) so disk slots turn over many times.
    let disk_parents = 16usize;
    let file_bytes = (disk_parents * PARENT_BYTES) as u64;
    let scratch = ScratchFile::new(&format!("ts-a1-{}", mode.label()));
    let executor = start_mode_executor(mode, &scratch, file_bytes, host, push_ring, ro_guard);
    let bounce_active = executor.uses_bounce_pool();

    // Key Host offsets around the exact 1 GiB PIECE_BYTES boundary (parent 41):
    let straddle_parent_off = 41 * PARENT_BYTES;
    let straddle_mla_off = straddle_parent_off + 6 * MLA_BLOCK_BYTES;
    let mla_left_guard_off = straddle_parent_off + 5 * MLA_BLOCK_BYTES;
    let mla_right_guard_off = straddle_parent_off + 7 * MLA_BLOCK_BYTES;
    let kda_left_guard_off = 40 * PARENT_BYTES;
    let kda_right_guard_off = 42 * PARENT_BYTES;
    let aligned_mla_off = 2 * PARENT_BYTES + 3 * MLA_BLOCK_BYTES;
    let aligned_kda_off = 3 * PARENT_BYTES;
    let push_slot0_off = 0usize;
    let push_slot1_off = PARENT_BYTES;

    // Seed neighbor guards and capture their golden SHA-256 digests.
    let mut left_mla_guard = vec![0u8; MLA_BLOCK_BYTES];
    let mut right_mla_guard = vec![0u8; MLA_BLOCK_BYTES];
    fill_seeded(&mut left_mla_guard, 0x1111_0001);
    fill_seeded(&mut right_mla_guard, 0x1111_0002);
    let left_mla_guard_sha = sha256(&left_mla_guard);
    let right_mla_guard_sha = sha256(&right_mla_guard);

    let mut left_kda_guard = vec![0u8; PARENT_BYTES];
    let mut right_kda_guard = vec![0u8; PARENT_BYTES];
    fill_seeded(&mut left_kda_guard, 0x2222_0001);
    fill_seeded(&mut right_kda_guard, 0x2222_0002);
    let left_kda_guard_sha = sha256(&left_kda_guard);
    let right_kda_guard_sha = sha256(&right_kda_guard);

    let mut push_guard = vec![0u8; PARENT_BYTES];
    fill_seeded(&mut push_guard, 0x3333_0001);
    let push_guard_sha = sha256(&push_guard);
    write_region_bytes(push_ring, push_slot1_off, &push_guard);

    write_region_bytes(host, mla_left_guard_off, &left_mla_guard);
    write_region_bytes(host, mla_right_guard_off, &right_mla_guard);

    let started = Instant::now();
    let mut total_writes = 0usize;
    let mut total_reads = 0usize;
    let mut total_bytes_verified = 0u64;
    let mut straddling_w_to_non_straddling_r = 0usize;
    let mut non_straddling_w_to_straddling_r = 0usize;
    let mut sha256_mismatches = 0usize;
    let mut memcmp_mismatches = 0usize;
    let mut sentinel_bleeds = 0usize;
    let mut neighbor_corruptions = 0usize;
    let mut spurious_checksum_mismatches = 0usize;

    // Reusable buffers to keep allocation overhead minimal.
    let mut mla_a = vec![0u8; MLA_BLOCK_BYTES];
    let mut mla_b = vec![0u8; MLA_BLOCK_BYTES];
    let mut verify_mla = vec![0u8; MLA_BLOCK_BYTES];

    // Phase 1: MLA cycles (868,352 B = 848 KiB) across 1 GiB PIECE_BYTES boundary.
    // Every cycle runs TWO full sub-transfers on the SAME disk slot:
    //   (a) Write from straddling Host (parent 41, sub 6) -> poison push_ring (0xDEADBEEF)
    //       -> read into non-straddling push_ring slot 0 -> verify 0-bit SHA-256 + memcmp.
    //   (b) Overwrite SAME disk slot from non-straddling Host (parent 2, sub 3)
    //       -> poison straddling Host (parent 41, sub 6) with 0xDEADBEEF
    //       -> read into straddling Host -> verify 0-bit SHA-256 + memcmp + zero spurious CRC mismatch.
    for cycle in 0..mla_cycles {
        let disk_slot = cycle % (disk_parents * 30);
        let disk_offset = (disk_slot * MLA_BLOCK_BYTES) as u64;

        // (a) Straddling Write -> Non-straddling Read (push_ring)
        fill_seeded(&mut mla_a, (cycle as u64) * 2 + 100);
        let expected_sha_a = sha256(&mla_a);
        write_region_bytes(host, straddle_mla_off, &mla_a);
        run_job(
            &executor,
            Job {
                direction: Direction::Write,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: host.address() + straddle_mla_off,
                    len: MLA_BLOCK_BYTES,
                }],
            },
        )
        .expect("MLA straddling write failed");
        total_writes += 1;

        // Poison destination (push_ring slot 0) with 0xDEADBEEF before read.
        fill_poison(push_ring, push_slot0_off, MLA_BLOCK_BYTES);
        match run_job(
            &executor,
            Job {
                direction: Direction::Read,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: push_ring.address() + push_slot0_off,
                    len: MLA_BLOCK_BYTES,
                }],
            },
        ) {
            Ok(_) => {}
            Err(IoError::ChecksumMismatch { .. }) => {
                spurious_checksum_mismatches += 1;
            }
            Err(e) => panic!("MLA non-straddling read failed: {e}"),
        }
        total_reads += 1;
        straddling_w_to_non_straddling_r += 1;

        push_ring.read_at(push_slot0_off, &mut verify_mla);
        if verify_mla != mla_a {
            memcmp_mismatches += 1;
        }
        if sha256(&verify_mla) != expected_sha_a {
            sha256_mismatches += 1;
        }
        if verify_mla.starts_with(&POISON_SENTINEL) || verify_mla.ends_with(&POISON_SENTINEL) {
            if !mla_a.starts_with(&POISON_SENTINEL) && !mla_a.ends_with(&POISON_SENTINEL) {
                sentinel_bleeds += 1;
            }
        }
        total_bytes_verified += MLA_BLOCK_BYTES as u64;

        // (b) Non-straddling Overwrite of SAME Disk slot -> Straddling Read (D1-05 false-positive test!)
        fill_seeded(&mut mla_b, (cycle as u64) * 2 + 101);
        let expected_sha_b = sha256(&mla_b);
        write_region_bytes(host, aligned_mla_off, &mla_b);
        run_job(
            &executor,
            Job {
                direction: Direction::Write,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: host.address() + aligned_mla_off,
                    len: MLA_BLOCK_BYTES,
                }],
            },
        )
        .expect("MLA non-straddling overwrite failed");
        total_writes += 1;

        fill_poison(host, straddle_mla_off, MLA_BLOCK_BYTES);
        match run_job(
            &executor,
            Job {
                direction: Direction::Read,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: host.address() + straddle_mla_off,
                    len: MLA_BLOCK_BYTES,
                }],
            },
        ) {
            Ok(_) => {}
            Err(IoError::ChecksumMismatch { .. }) => {
                spurious_checksum_mismatches += 1;
            }
            Err(e) => panic!("MLA straddling read failed: {e}"),
        }
        total_reads += 1;
        non_straddling_w_to_straddling_r += 1;

        host.read_at(straddle_mla_off, &mut verify_mla);
        if verify_mla != mla_b {
            memcmp_mismatches += 1;
        }
        if sha256(&verify_mla) != expected_sha_b {
            sha256_mismatches += 1;
        }
        if verify_mla.starts_with(&POISON_SENTINEL) || verify_mla.ends_with(&POISON_SENTINEL) {
            if !mla_b.starts_with(&POISON_SENTINEL) && !mla_b.ends_with(&POISON_SENTINEL) {
                sentinel_bleeds += 1;
            }
        }
        total_bytes_verified += MLA_BLOCK_BYTES as u64;

        // Periodically verify adjacent MLA neighbor blocks & push_ring guard slot untouched.
        if cycle % 64 == 63 || cycle + 1 == mla_cycles {
            if sha256(&read_region_bytes(host, mla_left_guard_off, MLA_BLOCK_BYTES))
                != left_mla_guard_sha
            {
                neighbor_corruptions += 1;
            }
            if sha256(&read_region_bytes(host, mla_right_guard_off, MLA_BLOCK_BYTES))
                != right_mla_guard_sha
            {
                neighbor_corruptions += 1;
            }
            if sha256(&read_region_bytes(push_ring, push_slot1_off, PARENT_BYTES))
                != push_guard_sha
            {
                neighbor_corruptions += 1;
            }
        }
    }

    // Phase 2: Full 26,050,560 B (24.84 MiB) KDA State block cycles across 1 GiB PIECE_BYTES boundary.
    write_region_bytes(host, kda_left_guard_off, &left_kda_guard);
    write_region_bytes(host, kda_right_guard_off, &right_kda_guard);

    let mut kda_a = vec![0u8; KDA_BLOCK_BYTES];
    let mut kda_b = vec![0u8; KDA_BLOCK_BYTES];
    let mut verify_kda = vec![0u8; KDA_BLOCK_BYTES];

    for cycle in 0..kda_cycles {
        let disk_parent = cycle % disk_parents;
        let disk_offset = (disk_parent * PARENT_BYTES) as u64;

        // (a) Straddling KDA Write (parent 41: 5.4 MiB piece 0 + 19.4 MiB piece 1, 7+ chunks)
        //     -> Non-straddling KDA Read (push_ring slot 0: 7 chunks in 1 piece)
        fill_seeded(&mut kda_a, 1_000_000 + (cycle as u64) * 2);
        let expected_sha_a = sha256(&kda_a);
        write_region_bytes(host, straddle_parent_off, &kda_a);
        run_job(
            &executor,
            Job {
                direction: Direction::Write,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: host.address() + straddle_parent_off,
                    len: KDA_BLOCK_BYTES,
                }],
            },
        )
        .expect("KDA straddling write failed");
        total_writes += 1;

        fill_poison(push_ring, push_slot0_off, KDA_BLOCK_BYTES);
        match run_job(
            &executor,
            Job {
                direction: Direction::Read,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: push_ring.address() + push_slot0_off,
                    len: KDA_BLOCK_BYTES,
                }],
            },
        ) {
            Ok(_) => {}
            Err(IoError::ChecksumMismatch { .. }) => {
                spurious_checksum_mismatches += 1;
            }
            Err(e) => panic!("KDA non-straddling read failed: {e}"),
        }
        total_reads += 1;
        straddling_w_to_non_straddling_r += 1;

        push_ring.read_at(push_slot0_off, &mut verify_kda);
        if verify_kda != kda_a {
            memcmp_mismatches += 1;
        }
        if sha256(&verify_kda) != expected_sha_a {
            sha256_mismatches += 1;
        }
        total_bytes_verified += KDA_BLOCK_BYTES as u64;

        // (b) Non-straddling KDA Overwrite of SAME Disk slot -> Straddling KDA Read (parent 41)
        fill_seeded(&mut kda_b, 1_000_000 + (cycle as u64) * 2 + 1);
        let expected_sha_b = sha256(&kda_b);
        write_region_bytes(host, aligned_kda_off, &kda_b);
        run_job(
            &executor,
            Job {
                direction: Direction::Write,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: host.address() + aligned_kda_off,
                    len: KDA_BLOCK_BYTES,
                }],
            },
        )
        .expect("KDA non-straddling overwrite failed");
        total_writes += 1;

        fill_poison(host, straddle_parent_off, KDA_BLOCK_BYTES);
        match run_job(
            &executor,
            Job {
                direction: Direction::Read,
                ios: vec![Io {
                    file_offset: disk_offset,
                    address: host.address() + straddle_parent_off,
                    len: KDA_BLOCK_BYTES,
                }],
            },
        ) {
            Ok(_) => {}
            Err(IoError::ChecksumMismatch { .. }) => {
                spurious_checksum_mismatches += 1;
            }
            Err(e) => panic!("KDA straddling read failed: {e}"),
        }
        total_reads += 1;
        non_straddling_w_to_straddling_r += 1;

        host.read_at(straddle_parent_off, &mut verify_kda);
        if verify_kda != kda_b {
            memcmp_mismatches += 1;
        }
        if sha256(&verify_kda) != expected_sha_b {
            sha256_mismatches += 1;
        }
        total_bytes_verified += KDA_BLOCK_BYTES as u64;

        if cycle % 10 == 9 || cycle + 1 == kda_cycles {
            if sha256(&read_region_bytes(host, kda_left_guard_off, PARENT_BYTES))
                != left_kda_guard_sha
            {
                neighbor_corruptions += 1;
            }
            if sha256(&read_region_bytes(host, kda_right_guard_off, PARENT_BYTES))
                != right_kda_guard_sha
            {
                neighbor_corruptions += 1;
            }
            if sha256(&read_region_bytes(push_ring, push_slot1_off, PARENT_BYTES))
                != push_guard_sha
            {
                neighbor_corruptions += 1;
            }
        }
    }

    assert!(!executor.degraded());
    assert!(!executor.broken());
    executor
        .close(Duration::from_secs(10))
        .expect("executor clean close");

    ModeBitwiseSummary {
        mode: mode.label(),
        bounce_active,
        mla_cycles,
        kda_cycles,
        total_writes,
        total_reads,
        total_bytes_verified,
        straddling_w_to_non_straddling_r,
        non_straddling_w_to_straddling_r,
        sha256_mismatches,
        memcmp_mismatches,
        sentinel_bleeds,
        neighbor_corruptions,
        spurious_checksum_mismatches,
        elapsed: started.elapsed(),
    }
}

#[test]
fn ts_a1_bitwise_round_trip_straddling_and_bounce_pool() {
    verify_sha256_vectors();
    // 44 parents = 1,146,224,640 B (~1.07 GiB), so parent 41 straddles 1 GiB PIECE_BYTES.
    let mut host = Memory::map(44 * PARENT_BYTES).expect("map 44-parent Host L2 pool");
    // 2 parents (~49.7 MiB) non-straddling push_ring (slot 0 active, slot 1 guard).
    let mut push_ring = Memory::map(2 * PARENT_BYTES).expect("map push_ring");
    let ro_guard = ReadOnlyEfaultGuard::new();

    // 700 full clean->evict->overwrite->cross-split cycles per mode = 2,100 cycles total
    // (4,200 Disk writes + 4,200 Disk reads verified with 0-bit SHA-256 + memcmp).
    let mla_cycles_per_mode = 660usize;
    let kda_cycles_per_mode = 40usize;
    let modes = [
        ExecMode::IoUringDirect,
        ExecMode::IoUringEfaultBounce,
        ExecMode::PsyncBounce,
    ];

    println!("=== TS-A1: Level-A 0-Bit Bitwise Round-Trip (1 GiB PIECE_BYTES Straddling + Bounce Pool) ===");
    println!(
        "Geometry: parent_bytes={PARENT_BYTES} (24.84 MiB), mla_block_bytes={MLA_BLOCK_BYTES} (848 KiB), kda_block_bytes={KDA_BLOCK_BYTES}, piece_bytes={PIECE_BYTES}"
    );
    let mut total_cycles = 0usize;
    let mut total_verified_bytes = 0u64;
    for mode in modes {
        let s = run_ts_a1_for_mode(
            mode,
            mla_cycles_per_mode,
            kda_cycles_per_mode,
            &mut host,
            &mut push_ring,
            &ro_guard,
        );
        let cycles = s.mla_cycles + s.kda_cycles;
        total_cycles += cycles;
        total_verified_bytes += s.total_bytes_verified;
        println!(
            "TS-A1 MODE={:<28} bounce={} cycles={} (mla={}, kda={}) writes={} reads={} straddle_w->non_r={} non_w->straddle_r={} verified_gib={:.2} sha256_err={} memcmp_err={} sentinel_bleed={} neighbor_err={} spurious_crc={} elapsed={:.2}s",
            s.mode,
            s.bounce_active,
            cycles,
            s.mla_cycles,
            s.kda_cycles,
            s.total_writes,
            s.total_reads,
            s.straddling_w_to_non_straddling_r,
            s.non_straddling_w_to_straddling_r,
            s.total_bytes_verified as f64 / (1u64 << 30) as f64,
            s.sha256_mismatches,
            s.memcmp_mismatches,
            s.sentinel_bleeds,
            s.neighbor_corruptions,
            s.spurious_checksum_mismatches,
            s.elapsed.as_secs_f64(),
        );
        assert_eq!(s.sha256_mismatches, 0, "SHA-256 mismatch in {}", s.mode);
        assert_eq!(s.memcmp_mismatches, 0, "memcmp mismatch in {}", s.mode);
        assert_eq!(s.sentinel_bleeds, 0, "sentinel bleed in {}", s.mode);
        assert_eq!(
            s.neighbor_corruptions, 0,
            "neighbor corruption in {}",
            s.mode
        );
        assert_eq!(
            s.spurious_checksum_mismatches, 0,
            "spurious ChecksumMismatch in {}",
            s.mode
        );
    }
    assert!(total_cycles >= 2_000);
    println!(
        "TS-A1 PASS: total_cycles={total_cycles} (2x cross-split I/Os each => {} verified reads), total_verified_gib={:.2} GiB, 0-bit diff across all 3 modes",
        total_cycles * 2,
        total_verified_bytes as f64 / (1u64 << 30) as f64
    );
}

// ---------------------------------------------------------------------------
// TS-A2: Silent On-Disk Single-Bit-Flip Chaos (D4-03)
// ---------------------------------------------------------------------------

#[test]
fn ts_a2_silent_on_disk_bit_flip_chaos() {
    verify_sha256_vectors();
    let mut host = Memory::map(44 * PARENT_BYTES).expect("map 44-parent Host L2 pool");
    let mut push_ring = Memory::map(2 * PARENT_BYTES).expect("map push_ring");
    let ro_guard = ReadOnlyEfaultGuard::new();

    let straddle_parent_off = 41 * PARENT_BYTES;
    let straddle_mla_off = straddle_parent_off + 6 * MLA_BLOCK_BYTES;
    let aligned_mla_off = 1 * PARENT_BYTES;
    let aligned_kda_off = 2 * PARENT_BYTES;
    let push_slot0_off = 0usize;

    let modes = [
        ExecMode::IoUringDirect,
        ExecMode::IoUringEfaultBounce,
        ExecMode::PsyncBounce,
    ];

    println!("=== TS-A2: Silent On-Disk Bit-Flip Chaos (D4-03) ===");
    let mut total_flips = 0usize;
    let mut total_caught = 0usize;

    for mode in modes {
        let file_bytes = (4 * PARENT_BYTES) as u64;
        let scratch = ScratchFile::new(&format!("ts-a2-{}", mode.label()));
        let executor = start_mode_executor(mode, &scratch, file_bytes, &host, &push_ring, &ro_guard);

        // 8 distinct single-bit corruption cases per mode:
        // (label, write_in_host, write_off, read_in_host, read_off, len, disk_off, flip_rel_offset, flip_bit)
        let piece_split_mla = PIECE_BYTES - straddle_mla_off; // 458,752 B
        let piece_split_kda = PIECE_BYTES - straddle_parent_off; // 5,668,864 B
        let cases: Vec<(&str, bool, usize, bool, usize, usize, u64, u64, u8)> = vec![
            (
                "mla_single_chunk_byte0",
                true,
                aligned_mla_off,
                true,
                aligned_mla_off,
                MLA_BLOCK_BYTES,
                0,
                0,
                0,
            ),
            (
                "mla_single_chunk_mid",
                true,
                aligned_mla_off,
                true,
                aligned_mla_off,
                MLA_BLOCK_BYTES,
                MLA_BLOCK_BYTES as u64,
                (MLA_BLOCK_BYTES / 2) as u64,
                3,
            ),
            (
                "mla_single_chunk_last",
                true,
                aligned_mla_off,
                true,
                aligned_mla_off,
                MLA_BLOCK_BYTES,
                (2 * MLA_BLOCK_BYTES) as u64,
                (MLA_BLOCK_BYTES - 1) as u64,
                7,
            ),
            (
                "kda_multi_chunk_chunk0",
                true,
                aligned_kda_off,
                true,
                aligned_kda_off,
                KDA_BLOCK_BYTES,
                PARENT_BYTES as u64,
                13,
                2,
            ),
            (
                "kda_multi_chunk_chunk3",
                true,
                aligned_kda_off,
                true,
                aligned_kda_off,
                KDA_BLOCK_BYTES,
                PARENT_BYTES as u64,
                (3 * MAX_IO_BYTES + 4099) as u64,
                5,
            ),
            (
                "kda_multi_chunk_tail_chunk6",
                true,
                aligned_kda_off,
                true,
                aligned_kda_off,
                KDA_BLOCK_BYTES,
                PARENT_BYTES as u64,
                (KDA_BLOCK_BYTES - 17) as u64,
                6,
            ),
            (
                "straddle_mla_w_to_push_r_piece1",
                true,
                straddle_mla_off,
                false,
                push_slot0_off,
                MLA_BLOCK_BYTES,
                (2 * PARENT_BYTES) as u64,
                (piece_split_mla + 123) as u64,
                4,
            ),
            (
                "non_straddle_kda_w_to_straddle_r_boundary",
                false,
                push_slot0_off,
                true,
                straddle_parent_off,
                KDA_BLOCK_BYTES,
                (3 * PARENT_BYTES) as u64,
                (piece_split_kda + 7) as u64,
                1,
            ),
        ];

        let mut mode_caught = 0usize;
        for (
            idx,
            &(label, w_in_host, w_off, r_in_host, r_off, len, disk_off, flip_rel, flip_bit),
        ) in cases.iter().enumerate()
        {
            total_flips += 1;
            let mut payload = vec![0u8; len];
            fill_seeded(&mut payload, 777_000 + idx as u64);
            let golden_sha = sha256(&payload);

            let w_addr = if w_in_host {
                write_region_bytes(&mut host, w_off, &payload);
                host.address() + w_off
            } else {
                write_region_bytes(&mut push_ring, w_off, &payload);
                push_ring.address() + w_off
            };
            run_job(
                &executor,
                Job {
                    direction: Direction::Write,
                    ios: vec![Io {
                        file_offset: disk_off,
                        address: w_addr,
                        len,
                    }],
                },
            )
            .expect("clean write before bit-flip");

            // Inject 1 single bit flip directly on NVMe file bypassing executor.
            flip_single_bit_on_disk(&scratch.0, disk_off + flip_rel, flip_bit);

            let r_addr = if r_in_host {
                fill_poison(&mut host, r_off, len);
                host.address() + r_off
            } else {
                fill_poison(&mut push_ring, r_off, len);
                push_ring.address() + r_off
            };
            let outcome = run_job(
                &executor,
                Job {
                    direction: Direction::Read,
                    ios: vec![Io {
                        file_offset: disk_off,
                        address: r_addr,
                        len,
                    }],
                },
            );
            match outcome {
                Err(IoError::ChecksumMismatch { expected, actual }) => {
                    assert_ne!(expected, actual, "expected != actual on bit-flip");
                    mode_caught += 1;
                    total_caught += 1;
                }
                other => panic!(
                    "TS-A2 FAILURE mode={} case={label}: expected ChecksumMismatch, got {other:?}",
                    mode.label()
                ),
            }
            // Single CRC mismatch MUST NOT trip the OS-error circuit breaker!
            assert!(
                !executor.degraded(),
                "ChecksumMismatch must not trip OS-error circuit breaker ({}, {label})",
                mode.label()
            );

            // Clean recovery: rewrite healthy payload to the same slot and verify 0-bit read.
            if w_in_host {
                write_region_bytes(&mut host, w_off, &payload);
            } else {
                write_region_bytes(&mut push_ring, w_off, &payload);
            }
            run_job(
                &executor,
                Job {
                    direction: Direction::Write,
                    ios: vec![Io {
                        file_offset: disk_off,
                        address: w_addr,
                        len,
                    }],
                },
            )
            .expect("recovery rewrite failed");
            if r_in_host {
                fill_poison(&mut host, r_off, len);
            } else {
                fill_poison(&mut push_ring, r_off, len);
            }
            run_job(
                &executor,
                Job {
                    direction: Direction::Read,
                    ios: vec![Io {
                        file_offset: disk_off,
                        address: r_addr,
                        len,
                    }],
                },
            )
            .expect("recovery read failed");
            let recovered = if r_in_host {
                read_region_bytes(&host, r_off, len)
            } else {
                read_region_bytes(&push_ring, r_off, len)
            };
            assert_eq!(sha256(&recovered), golden_sha, "recovery SHA-256 mismatch");
            assert_eq!(recovered, payload, "recovery memcmp mismatch");
        }
        println!(
            "TS-A2 MODE={:<28} bounce={} flips={} caught={}/{} (100.0%) degraded_false_trips=0 recovery=PASS",
            mode.label(),
            executor.uses_bounce_pool(),
            cases.len(),
            mode_caught,
            cases.len()
        );
        executor.close(Duration::from_secs(10)).unwrap();
    }
    assert_eq!(total_caught, total_flips);
    println!(
        "TS-A2 PASS: {total_caught}/{total_flips} (100%) single-bit on-disk corruptions caught, 0 corrupt bytes accepted, 0 false circuit-breaker trips"
    );
}

// ---------------------------------------------------------------------------
// TS-A4: ReadLatencyAimd 16-Rank Contention & Starvation-Free Recovery (D4-01)
// ---------------------------------------------------------------------------

#[test]
fn ts_a4_aimd_16_rank_contention_and_recovery() {
    println!("=== TS-A4: ReadLatencyAimd 16-Rank Contention & Starvation-Free Recovery (D4-01) ===");

    // Part A: Deterministic synthetic 3 ms .. 20 ms NVMe contention trace + additive recovery.
    let max_wd = 8usize;
    let mut aimd = ReadLatencyAimd::new();
    let base = Instant::now();
    assert_eq!(aimd.effective_write_depth_for(max_wd, true), 8);
    assert_eq!(aimd.effective_write_depth_for(max_wd, false), 8);

    // (window_label, sample_latencies_us, expected_wd_pending, expected_wd_idle)
    let schedule: &[(&str, u64, usize, usize)] = &[
        ("W00_baseline_fast", 350, 8, 8),
        ("W01_spike_3.5ms", 3_500, 4, 4),
        ("W02_spike_8.0ms", 8_000, 2, 2),
        ("W03_spike_14.0ms", 14_000, 1, 1),
        ("W04_spike_20.0ms", 20_000, 0, 1), // starvation-free: 0 pending, 1 when idle!
        ("W05_hysteresis_1.5ms", 1_500, 0, 1), // neutral band [1000, 2000] us holds steady
        ("W06_recover_400us", 400, 1, 1),
        ("W07_recover_400us", 400, 2, 2),
        ("W08_recover_400us", 400, 3, 3),
        ("W09_recover_400us", 400, 4, 4),
        ("W10_recover_400us", 400, 5, 5),
        ("W11_recover_400us", 400, 6, 6),
        ("W12_recover_400us", 400, 7, 7),
        ("W13_recover_400us", 400, 8, 8),
        ("W14_recover_cap_400us", 400, 8, 8),
    ];

    let mut cursor_ms = 0u64;
    for &(label, lat_us, exp_pending, exp_idle) in schedule {
        // Feed exactly AIMD_MIN_SAMPLES (8) samples spaced 15 ms apart
        // ((8 - 1) * 15 = 105 ms > AIMD_WINDOW = 100 ms) so each window
        // triggers once cleanly with zero leftover carryover samples.
        for s in 0..AIMD_MIN_SAMPLES {
            cursor_ms += 15;
            let jitter = (s as u64 % 3) * 20;
            aimd.record_read(
                Duration::from_micros(lat_us + jitter),
                base + Duration::from_millis(cursor_ms),
                max_wd,
            );
        }
        let got_pending = aimd.effective_write_depth_for(max_wd, true);
        let got_idle = aimd.effective_write_depth_for(max_wd, false);
        println!(
            "TS-A4-TRACE window={label:<22} p99_us~={lat_us:<6} eff_wd_reads_pending={got_pending} eff_wd_idle={got_idle}"
        );
        assert_eq!(got_pending, exp_pending, "mismatch on {label} (pending)");
        assert_eq!(got_idle, exp_idle, "mismatch on {label} (idle)");
    }

    // Part B: Live 16-rank concurrent NVMe contention on /tmp (16 Executors sharing 1 NVMe).
    const RANKS: usize = 16;
    const RANK_SLOTS: usize = 16;
    const CHUNK_LEN: usize = MLA_BLOCK_BYTES; // 848 KiB
    let file_bytes = (RANK_SLOTS * CHUNK_LEN) as u64;
    let total_reads = Arc::new(AtomicUsize::new(0));
    let total_writes = Arc::new(AtomicUsize::new(0));
    let total_errors = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..RANKS)
        .map(|rank| {
            let total_reads = total_reads.clone();
            let total_writes = total_writes.clone();
            let total_errors = total_errors.clone();
            thread::spawn(move || {
                let scratch = ScratchFile::new(&format!("ts-a4-rank{rank}"));
                let mut mem = Memory::map(RANK_SLOTS * CHUNK_LEN).expect("rank mem");
                let mut seed_buf = vec![0u8; CHUNK_LEN];
                fill_seeded(&mut seed_buf, 9000 + rank as u64);
                for s in 0..RANK_SLOTS {
                    mem.write_at(s * CHUNK_LEN, &seed_buf);
                }
                let file = DiskFile::create(&scratch.0, file_bytes).expect("rank disk file");
                let executor = Arc::new(
                    Executor::start(
                        file,
                        &[(mem.address(), mem.size())],
                        ExecutorConfig {
                            backend: Backend::IoUring,
                            queue_depth: 16,
                            write_queue_depth: 8,
                            max_io_bytes: MAX_IO_BYTES,
                            write_bytes_per_second: 200 * (1 << 20), // 200 MiB/s per rank
                        },
                    )
                    .expect("start rank executor"),
                );

                // Pre-fill all slots so reads have recorded checksums.
                for s in 0..RANK_SLOTS {
                    run_job(
                        &executor,
                        Job {
                            direction: Direction::Write,
                            ios: vec![Io {
                                file_offset: (s * CHUNK_LEN) as u64,
                                address: mem.address() + s * CHUNK_LEN,
                                len: CHUNK_LEN,
                            }],
                        },
                    )
                    .unwrap();
                }

                // Mixed foreground reads (slots 0..8) + background writes (slots 8..16) for 2.0s.
                let deadline = Instant::now() + Duration::from_secs(2);
                let mut read_lats_us = Vec::with_capacity(512);
                let mut min_eff_wd_pending = 8usize;
                let mut min_eff_wd_idle = 8usize;
                let mut rank_writes = 0usize;
                let mut iter = 0usize;

                while Instant::now() < deadline {
                    let (r_tx, r_rx) = mpsc::channel();
                    let (w_tx, w_rx) = mpsc::channel();
                    let t0 = Instant::now();
                    for k in 0..4 {
                        let slot = (iter + k) % 8;
                        let tx = r_tx.clone();
                        executor
                            .submit(
                                Job {
                                    direction: Direction::Read,
                                    ios: vec![Io {
                                        file_offset: (slot * CHUNK_LEN) as u64,
                                        address: mem.address() + slot * CHUNK_LEN,
                                        len: CHUNK_LEN,
                                    }],
                                },
                                move |rep| {
                                    let _ = tx.send(rep);
                                },
                            )
                            .unwrap();
                    }
                    let wslot = 8 + (iter % 8);
                    executor
                        .submit(
                            Job {
                                direction: Direction::Write,
                                ios: vec![Io {
                                    file_offset: (wslot * CHUNK_LEN) as u64,
                                    address: mem.address() + wslot * CHUNK_LEN,
                                    len: CHUNK_LEN,
                                }],
                            },
                            move |rep| {
                                let _ = w_tx.send(rep);
                            },
                        )
                        .unwrap();

                    for _ in 0..4 {
                        let rep = r_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                        if rep.outcome.is_ok() {
                            read_lats_us.push(rep.elapsed.as_micros() as u64);
                            total_reads.fetch_add(1, Ordering::Relaxed);
                        } else {
                            total_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let wrep = w_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    if wrep.outcome.is_ok() {
                        rank_writes += 1;
                        total_writes.fetch_add(1, Ordering::Relaxed);
                    } else {
                        total_errors.fetch_add(1, Ordering::Relaxed);
                    }
                    let _ = t0;
                    min_eff_wd_pending = min_eff_wd_pending.min(executor.effective_write_depth(true));
                    min_eff_wd_idle = min_eff_wd_idle.min(executor.effective_write_depth(false));
                    iter += 1;
                }

                // Verify starvation-free idle floor >= 1.
                assert!(
                    min_eff_wd_idle >= 1,
                    "rank {rank} idle effective_write_depth dropped to 0!"
                );
                assert!(rank_writes > 0, "rank {rank} suffered write starvation!");
                executor.close(Duration::from_secs(10)).unwrap();
                (read_lats_us, min_eff_wd_pending, min_eff_wd_idle, rank_writes)
            })
        })
        .collect();

    let mut all_lats = Vec::new();
    let mut global_min_wd_pending = 8usize;
    let mut global_min_wd_idle = 8usize;
    let mut min_rank_writes = usize::MAX;
    for h in handles {
        let (lats, mwp, mwi, rw) = h.join().expect("rank thread panicked");
        all_lats.extend(lats);
        global_min_wd_pending = global_min_wd_pending.min(mwp);
        global_min_wd_idle = global_min_wd_idle.min(mwi);
        min_rank_writes = min_rank_writes.min(rw);
    }
    all_lats.sort_unstable();
    let p50 = all_lats[all_lats.len() / 2];
    let p99 = all_lats[(all_lats.len() * 99) / 100];
    let max = *all_lats.last().unwrap();
    let errs = total_errors.load(Ordering::Relaxed);
    assert_eq!(errs, 0);
    assert!(global_min_wd_idle >= 1);
    println!(
        "TS-A4-LIVE-16RANK: ranks={RANKS} reads={} writes={} min_writes_per_rank={min_rank_writes} errors={errs} read_p50_us={p50} read_p99_us={p99} read_max_us={max} min_eff_wd_pending={global_min_wd_pending} min_eff_wd_idle={global_min_wd_idle}",
        total_reads.load(Ordering::Relaxed),
        total_writes.load(Ordering::Relaxed),
    );
    println!("TS-A4 PASS: deterministic AIMD 8->4->2->1->0 (idle=1) -> +1/100ms -> 8 verified; 16-rank live NVMe contention 0 errors, 0 cleaning starvation");
}

// ---------------------------------------------------------------------------
// TS-A5: O_DIRECT Bounce-Pool vs Non-Bounce vs psync Throughput & CPU Bench
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum BenchMode {
    IoUringFixedDirect,
    IoUringFixedBounce,
    PsyncDirect,
    PsyncBounce,
}

impl BenchMode {
    fn label(self) -> &'static str {
        match self {
            Self::IoUringFixedDirect => "io_uring_fixed_direct",
            Self::IoUringFixedBounce => "io_uring_fixed_bounce",
            Self::PsyncDirect => "psync_direct",
            Self::PsyncBounce => "psync_bounce",
        }
    }
}

struct BenchCell {
    mode: &'static str,
    bounce: bool,
    write_mb_s: f64,
    read_mb_s: f64,
    read_p50_us: u64,
    read_p99_us: u64,
    cpu_ms_per_gib: f64,
    cpu_user_s: f64,
    cpu_sys_s: f64,
}

fn bench_cell(mode: BenchMode, payload_bytes: usize, jobs_in_flight: usize) -> BenchCell {
    // Keep file ~512 MiB (exact multiple of payload_bytes) so every bench cell
    // measures real NVMe O_DIRECT throughput quickly and cleanly.
    let min_blocks = (512 * (1 << 20) / payload_bytes).max(jobs_in_flight * 2);
    let file_bytes = (min_blocks * payload_bytes) as u64;
    let scratch = ScratchFile::new(&format!("ts-a5-{}-{payload_bytes}", mode.label()));
    let mut mem = Memory::map(jobs_in_flight * payload_bytes).expect("bench staging mem");
    let mut seed = vec![0u8; payload_bytes];
    fill_seeded(&mut seed, payload_bytes as u64);
    for slot in 0..jobs_in_flight {
        mem.write_at(slot * payload_bytes, &seed);
    }

    let file = DiskFile::create(&scratch.0, file_bytes).expect("bench disk file");
    let cfg = ExecutorConfig {
        backend: match mode {
            BenchMode::IoUringFixedDirect | BenchMode::IoUringFixedBounce => Backend::IoUring,
            BenchMode::PsyncDirect | BenchMode::PsyncBounce => Backend::Psync,
        },
        queue_depth: jobs_in_flight.max(16),
        write_queue_depth: jobs_in_flight.max(16),
        max_io_bytes: MAX_IO_BYTES,
        write_bytes_per_second: u64::MAX / 4,
    };
    let executor = match mode {
        BenchMode::IoUringFixedDirect | BenchMode::PsyncDirect => {
            Executor::start(file, &[(mem.address(), mem.size())], cfg).unwrap()
        }
        BenchMode::IoUringFixedBounce | BenchMode::PsyncBounce => {
            Executor::start_bounced(file, &[(mem.address(), mem.size())], cfg).unwrap()
        }
    };
    let bounce = executor.uses_bounce_pool();
    match mode {
        BenchMode::IoUringFixedDirect | BenchMode::PsyncDirect => assert!(!bounce),
        BenchMode::IoUringFixedBounce | BenchMode::PsyncBounce => assert!(bounce),
    }

    let (cpu_u0, cpu_s0) = process_cpu_seconds();

    // Sequential write pass filling the entire file (records CRC32C for all blocks).
    let (tx, rx) = mpsc::channel::<(usize, Instant, bool)>();
    let submit = |slot: usize, dir: Direction, block_idx: usize| {
        let tx = tx.clone();
        let t0 = Instant::now();
        executor
            .submit(
                Job {
                    direction: dir,
                    ios: vec![Io {
                        file_offset: (block_idx * payload_bytes) as u64,
                        address: mem.address() + slot * payload_bytes,
                        len: payload_bytes,
                    }],
                },
                move |rep: JobReport| {
                    let _ = tx.send((slot, t0, rep.outcome.is_ok()));
                },
            )
            .unwrap();
    };

    let w_start = Instant::now();
    let mut next = 0usize;
    for slot in 0..jobs_in_flight.min(min_blocks) {
        submit(slot, Direction::Write, next);
        next += 1;
    }
    let mut done = 0usize;
    while done < min_blocks {
        let (slot, _, ok) = rx.recv_timeout(Duration::from_secs(30)).unwrap();
        assert!(ok, "bench write failed");
        done += 1;
        if next < min_blocks {
            submit(slot, Direction::Write, next);
            next += 1;
        }
    }
    let w_secs = w_start.elapsed().as_secs_f64();
    let write_mb_s = (file_bytes as f64 / (1024.0 * 1024.0)) / w_secs;

    // Random read pass for ~1.0 s (at least 1 full file pass worth of blocks or 1.0 s).
    let mut prng = 0x9e37_79b9_7f4a_7c15u64 ^ (payload_bytes as u64);
    let mut next_block = || {
        prng ^= prng << 13;
        prng ^= prng >> 7;
        prng ^= prng << 17;
        (prng as usize) % min_blocks
    };
    let r_start = Instant::now();
    let r_deadline = r_start + Duration::from_millis(1200);
    let mut inflight = 0usize;
    for slot in 0..jobs_in_flight {
        submit(slot, Direction::Read, next_block());
        inflight += 1;
    }
    let mut lats_us = Vec::with_capacity(4096);
    while inflight > 0 {
        let (slot, t0, ok) = rx.recv_timeout(Duration::from_secs(30)).unwrap();
        assert!(ok, "bench read failed");
        lats_us.push(t0.elapsed().as_micros() as u64);
        if Instant::now() < r_deadline || lats_us.len() < min_blocks {
            submit(slot, Direction::Read, next_block());
        } else {
            inflight -= 1;
        }
    }
    let r_secs = r_start.elapsed().as_secs_f64();
    let read_bytes = lats_us.len() as u64 * payload_bytes as u64;
    let read_mb_s = (read_bytes as f64 / (1024.0 * 1024.0)) / r_secs;

    let (cpu_u1, cpu_s1) = process_cpu_seconds();
    let cpu_user_s = (cpu_u1 - cpu_u0).max(0.0);
    let cpu_sys_s = (cpu_s1 - cpu_s0).max(0.0);
    let total_gib = (file_bytes + read_bytes) as f64 / (1u64 << 30) as f64;
    let cpu_ms_per_gib = ((cpu_user_s + cpu_sys_s) * 1000.0) / total_gib;

    lats_us.sort_unstable();
    let read_p50_us = lats_us[lats_us.len() / 2];
    let read_p99_us = lats_us[(lats_us.len() * 99) / 100];

    executor.close(Duration::from_secs(10)).unwrap();
    BenchCell {
        mode: mode.label(),
        bounce,
        write_mb_s,
        read_mb_s,
        read_p50_us,
        read_p99_us,
        cpu_ms_per_gib,
        cpu_user_s,
        cpu_sys_s,
    }
}

#[test]
fn ts_a5_odirect_bounce_pool_vs_direct_vs_psync_bench() {
    println!("=== TS-A5: O_DIRECT Bounce-Pool vs Direct vs psync Throughput & CPU Bench (D5-01) ===");
    let payloads = [
        ("848_KiB_MLA", MLA_BLOCK_BYTES, 16usize),
        ("4_MiB_max_io", MAX_IO_BYTES, 16usize),
        ("24.84_MiB_KDA", KDA_BLOCK_BYTES, 8usize),
    ];
    let modes = [
        BenchMode::IoUringFixedDirect,
        BenchMode::IoUringFixedBounce,
        BenchMode::PsyncDirect,
        BenchMode::PsyncBounce,
    ];

    for (name, bytes, qd) in payloads {
        let mut results = Vec::new();
        for mode in modes {
            let cell = bench_cell(mode, bytes, qd);
            println!(
                "TS-A5 payload={name:<14} ({bytes:>8} B, qd={qd:>2}) mode={:<22} bounce={:<5} write={:>8.1} MiB/s read={:>8.1} MiB/s read_p50={:>6} us read_p99={:>6} us cpu={:>6.1} ms/GiB (user={:.3}s sys={:.3}s)",
                cell.mode,
                cell.bounce,
                cell.write_mb_s,
                cell.read_mb_s,
                cell.read_p50_us,
                cell.read_p99_us,
                cell.cpu_ms_per_gib,
                cell.cpu_user_s,
                cell.cpu_sys_s,
            );
            results.push(cell);
        }
        let direct = &results[0];
        let bounce = &results[1];
        let read_ratio = (bounce.read_mb_s / direct.read_mb_s) * 100.0;
        let write_ratio = (bounce.write_mb_s / direct.write_mb_s) * 100.0;
        println!(
            "TS-A5-DELTA payload={name:<14} bounce_vs_direct: read={read_ratio:.1}% write={write_ratio:.1}% cpu_delta={:+.1} ms/GiB",
            bounce.cpu_ms_per_gib - direct.cpu_ms_per_gib,
        );
    }
    println!("TS-A5 PASS: completed all 12 benchmark cells (4 modes x 3 payload sizes) on local NVMe");
}
