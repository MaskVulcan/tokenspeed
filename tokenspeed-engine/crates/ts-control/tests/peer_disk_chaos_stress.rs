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

//! **TS-A3 (`D2-05`, `D4-02`, `D4-04`)**:
//! `push_ring` 20-slot saturation + 128 concurrent mixed MLA/KDA Disk push jobs
//! with 30% mid-flight cancellations & delayed slow-writer quarantine fencing,
//! followed by 8 consecutive real OS I/O errors tripping `ts-disk` into
//! `degraded() == true` -> mirrored-rank `Agreement.degraded` ->
//! `CacheCoordinator::InvalidateDiskCache()` -> ZMQ `ClearMedium("DISK")` wire
//! emission + zero-leak verification.

#![allow(unsafe_code)]

use std::{
    collections::HashMap,
    ffi::CString,
    os::fd::RawFd,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Barrier, Once,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_control::{
    disk::{DiskSettings, DiskTier},
    kv_events::{KvEventPublisher, KvEventsConfig},
    peer::{
        geometry::Spans,
        push::{PagePusher, PageWriter, PushEndpoint, PushItem, PushSource, PushSpan},
    },
};
use ts_disk::{
    Backend, Direction, DiskFile, Executor, ExecutorConfig, Io, IoError, Job, JobReport, Memory,
    ALIGNMENT,
};
use ts_l3::{GroupLayout, HostBuffer, HostLayout};
use ts_scheduler::{
    CacheEvent, CacheEventKind, CacheOpKind, ForwardEvent, ForwardEventKind, KvEventBatch,
    RequestSpec, Scheduler,
};
use zeromq::{Socket, SocketRecv, SubSocket};

/// Exact LongCat Lite 17B 4 KiB-aligned geometry.
const PARENT_BYTES: usize = 26_050_560; // 24.84 MiB (6,360 * 4 KiB)
const MLA_BLOCK_BYTES: usize = 868_352; // 848 KiB (212 * 4 KiB, 30 per parent)
const KDA_BLOCK_BYTES: usize = PARENT_BYTES;
const PUSH_RING_SLOTS: usize = 20;
const PUSH_RING_BYTES: usize = PUSH_RING_SLOTS * PARENT_BYTES; // 521,011,200 B
const SENTINEL: [u8; 4] = [0xCA, 0xFE, 0xBA, 0xBE];

// ---------------------------------------------------------------------------
// Self-contained RFC 6234 SHA-256
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
        sha256_compress(&mut state, chunk.try_into().unwrap());
    }
    let mut pad = [0u8; 128];
    pad[..rem.len()].copy_from_slice(rem);
    pad[rem.len()] = 0x80;
    let total_bits = (data.len() as u64) * 8;
    let pad_len = if rem.len() < 56 { 64 } else { 128 };
    pad[pad_len - 8..pad_len].copy_from_slice(&total_bits.to_be_bytes());
    for chunk in pad[..pad_len].chunks_exact(64) {
        sha256_compress(&mut state, chunk.try_into().unwrap());
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn fill_seeded(buf: &mut [u8], seed: u64) {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xd1b5_4a32_d192_ed03;
    for chunk in buf.chunks_exact_mut(8) {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes());
    }
}

struct ScratchFile(PathBuf);

impl ScratchFile {
    fn new(name: &str) -> Self {
        let dir = std::env::var_os("TS_DISK_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        Self(dir.join(format!(
            "ts-control-chaos-{}-{seq}-{name}.bin",
            std::process::id()
        )))
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Atomically redirects `target_fd` to `/dev/full` via `dup2`, causing every
/// subsequent `io_uring`/`pwrite` write on that fd to fail immediately with
/// real kernel `ENOSPC` (`IoError::Os(28)`) without closing or releasing the
/// fd number (mirroring D4-02 disk-full / hard fault).
fn inject_kernel_enospc_on_fd(target_fd: RawFd) {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn dup2(oldfd: i32, newfd: i32) -> i32;
    }
    let dev_full = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/full")
        .expect("open /dev/full for ENOSPC fault injection");
    assert_eq!(
        unsafe { dup2(dev_full.as_raw_fd(), target_fd) },
        target_fd,
        "dup2(/dev/full, disk_fd) failed"
    );
}

// ---------------------------------------------------------------------------
// Chaos PageWriter for Phase 1: tracks per-job landing memory, sentinel guards,
// controlled RDMA slow-writer delays, and strict post-cancel write fencing.
// ---------------------------------------------------------------------------

struct JobLanding {
    buffer: Mutex<Vec<u8>>,
    fenced: AtomicBool,
    delay: Duration,
    writes_seen: AtomicUsize,
}

struct ChaosPageWriter {
    ring: Arc<Memory>,
    landings: Mutex<HashMap<String, Arc<JobLanding>>>,
    late_fenced_writes: AtomicUsize,
    total_writes: AtomicUsize,
    total_bytes_written: AtomicU64,
    max_concurrent_writers: AtomicUsize,
    active_writers: AtomicUsize,
}

impl ChaosPageWriter {
    fn new(ring: Arc<Memory>) -> Self {
        Self {
            ring,
            landings: Mutex::new(HashMap::new()),
            late_fenced_writes: AtomicUsize::new(0),
            total_writes: AtomicUsize::new(0),
            total_bytes_written: AtomicU64::new(0),
            max_concurrent_writers: AtomicUsize::new(0),
            active_writers: AtomicUsize::new(0),
        }
    }

    fn register(&self, session: String, total_bytes: usize, delay: Duration) -> Arc<JobLanding> {
        let mut buf = vec![0u8; total_bytes];
        for chunk in buf.chunks_exact_mut(4) {
            chunk.copy_from_slice(&SENTINEL);
        }
        let landing = Arc::new(JobLanding {
            buffer: Mutex::new(buf),
            fenced: AtomicBool::new(false),
            delay,
            writes_seen: AtomicUsize::new(0),
        });
        self.landings.lock().insert(session, landing.clone());
        landing
    }

    fn unregister(&self, session: &str) {
        self.landings.lock().remove(session);
    }
}

impl PageWriter for ChaosPageWriter {
    fn write(&self, session: &str, spans: &Spans) -> Result<(), String> {
        let cur = self.active_writers.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_concurrent_writers
            .fetch_max(cur, Ordering::Relaxed);
        let landing = self
            .landings
            .lock()
            .get(session)
            .cloned()
            .ok_or_else(|| format!("unknown session {session}"))?;

        if landing.fenced.load(Ordering::Acquire) {
            self.late_fenced_writes.fetch_add(1, Ordering::Relaxed);
            self.active_writers.fetch_sub(1, Ordering::AcqRel);
            return Err("write hit fenced/quarantined job".to_owned());
        }

        if !landing.delay.is_zero() {
            thread::sleep(landing.delay);
        }

        // Re-check fence after simulated RDMA transfer delay!
        if landing.fenced.load(Ordering::Acquire) {
            self.late_fenced_writes.fetch_add(1, Ordering::Relaxed);
            self.active_writers.fetch_sub(1, Ordering::AcqRel);
            return Err("write completed after job fenced".to_owned());
        }

        let mut dst = landing.buffer.lock();
        for ((&src, &dest_off), &len) in spans
            .sources
            .iter()
            .zip(&spans.destinations)
            .zip(&spans.lengths)
        {
            let ring_off = src as usize - self.ring.address();
            let d0 = dest_off as usize;
            let d1 = d0 + len as usize;
            self.ring.read_at(ring_off, &mut dst[d0..d1]);
            self.total_bytes_written.fetch_add(len, Ordering::Relaxed);
        }
        landing.writes_seen.fetch_add(1, Ordering::Relaxed);
        self.total_writes.fetch_add(1, Ordering::Relaxed);
        self.active_writers.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Python / Scheduler helpers for Phase 2 (2-rank mirrored replica + ZMQ)
// ---------------------------------------------------------------------------

const PAGE: i32 = 4;
const SMALL_BLOCK_BYTES: usize = 4096;
const SMALL_HOST_BLOCKS: usize = 8;
const SMALL_DISK_BLOCKS: usize = 32;

fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
        Python::initialize();
        Python::attach(|py| {
            let module = ts_scheduler::python_module(py).unwrap();
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("tokenspeed_scheduler.tokenspeed_scheduler_ext", module)
                .unwrap();
            let package = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../tokenspeed-scheduler/python"
            );
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, package))
                .unwrap();
        });
    });
}

fn mirrored_scheduler(name: &str) -> Scheduler {
    interpreter();
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 64
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = 9
cfg.num_snapshot_pages = 9
cfg.max_retracted_requests = 8
cfg.disable_l2_cache = False
cfg.enable_kv_cache_events = True
cfg.num_host_pages = {host_pages}
cfg.host_group_packing = [1]
cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack
cfg.num_disk_pages = {disk_pages}
cfg.disk_config = ts.DiskConfig(max_operations=4, max_blocks_per_operation=64,
                                max_host_blocks=4, max_write_host_blocks=2,
                                max_queued_cleanings=64, min_prefetch_tokens=1,
                                wait_timeout_us=10_000_000, ready_timeout_us=10_000_000)
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full", block_granularity={PAGE},
                                        total_pages=9, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation=1, max_leases=8,
                                soft_pin_us=1_000_000, hard_pin_us=1_000_000,
                                lookup_timeout_us=1_000_000, min_fetch_tokens={PAGE},
                                landing_tier=ts.PeerLandingTier.Host, max_host_landing_blocks=4,
                                host_ready_timeout_us=1_000_000)
result = ts.Scheduler(cfg)
"#,
            host_pages = SMALL_HOST_BLOCKS + 1,
            disk_pages = SMALL_DISK_BLOCKS + 1,
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn small_layout(num_lcm_blocks: usize) -> HostLayout {
    HostLayout {
        groups: vec![GroupLayout {
            packing: 1,
            block_bytes: SMALL_BLOCK_BYTES,
            shard_count: 1,
        }],
        lcm_block_bytes: SMALL_BLOCK_BYTES,
        num_lcm_blocks,
        owner_rank: 0,
        tp_rank: 0,
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

// ---------------------------------------------------------------------------
// TS-A3 Test Entry Point
// ---------------------------------------------------------------------------

#[test]
fn ts_a3_push_ring_saturation_cancel_quarantine_and_degraded_clear_medium() {
    println!("=== TS-A3: 20-Slot push_ring Saturation + 30% Cancel/Quarantine + Degraded ClearMedium(\"DISK\") ===");

    // =========================================================================
    // PHASE 1: Realistic 20-slot push_ring (521 MB) + 128 concurrent Disk push
    //          jobs (mixed 848 KiB MLA + 24.84 MiB KDA) + 30% mid-flight cancel
    //          & slow writer + strict write fencing + 0-bit SHA-256 verification.
    // =========================================================================
    let disk_parents = 8usize;
    let disk_layout = HostLayout {
        groups: vec![
            GroupLayout {
                packing: 30,
                block_bytes: MLA_BLOCK_BYTES,
                shard_count: 1,
            },
            GroupLayout {
                packing: 1,
                block_bytes: KDA_BLOCK_BYTES,
                shard_count: 1,
            },
        ],
        lcm_block_bytes: PARENT_BYTES,
        num_lcm_blocks: disk_parents,
        owner_rank: 0,
        tp_rank: 0,
    };
    let scratch_p1 = ScratchFile::new("ts-a3-push-ring");
    let file_bytes = disk_layout.bytes() as u64;
    let file = DiskFile::create(&scratch_p1.0, file_bytes).expect("create Phase 1 NVMe file");
    let mut seed_mem = Memory::map(PARENT_BYTES).expect("seed buffer");
    let ring = Arc::new(Memory::map(PUSH_RING_BYTES).expect("map 20-slot 521 MiB push_ring"));

    let executor = Arc::new(
        Executor::start(
            file,
            &[
                (seed_mem.address(), seed_mem.size()),
                (ring.address(), ring.size()),
            ],
            ExecutorConfig {
                backend: Backend::IoUring,
                queue_depth: 16,
                write_queue_depth: 8,
                max_io_bytes: 4 << 20,
                write_bytes_per_second: u64::MAX / 4,
            },
        )
        .expect("start Phase 1 executor"),
    );

    // Populate Disk parents 1..=4 with 120 MLA blocks (group 0, block ids 1..=120)
    // and Disk parents 5..=8 with 4 KDA blocks (group 1, block ids 5..=8).
    let mut golden_mla: HashMap<i32, (Vec<u8>, [u8; 32])> = HashMap::new();
    let mut golden_kda: HashMap<i32, (Vec<u8>, [u8; 32])> = HashMap::new();

    for parent_idx in 0..4usize {
        let mut parent_buf = vec![0u8; PARENT_BYTES];
        for sub in 0..30usize {
            let block_id = (parent_idx * 30 + sub + 1) as i32;
            let range = disk_layout
                .owned_range(0, block_id)
                .unwrap()
                .expect("MLA range");
            assert_eq!(range.size, MLA_BLOCK_BYTES);
            let mut blk = vec![0u8; MLA_BLOCK_BYTES];
            fill_seeded(&mut blk, 10_000 + block_id as u64);
            let digest = sha256(&blk);
            parent_buf[sub * MLA_BLOCK_BYTES..(sub + 1) * MLA_BLOCK_BYTES].copy_from_slice(&blk);
            golden_mla.insert(block_id, (blk, digest));
        }
        seed_mem.write_at(0, &parent_buf);
        let (tx, rx) = mpsc::channel();
        let ios: Vec<Io> = (0..30usize)
            .map(|sub| {
                let block_id = (parent_idx * 30 + sub + 1) as i32;
                let range = disk_layout.owned_range(0, block_id).unwrap().unwrap();
                Io {
                    file_offset: range.offset as u64,
                    address: seed_mem.address() + sub * MLA_BLOCK_BYTES,
                    len: range.size,
                }
            })
            .collect();
        executor
            .submit(
                Job {
                    direction: Direction::Write,
                    ios,
                },
                move |rep: JobReport| tx.send(rep.outcome).unwrap(),
            )
            .unwrap();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
    }

    for kda_block in 5..=8i32 {
        let range = disk_layout
            .owned_range(1, kda_block)
            .unwrap()
            .expect("KDA range");
        assert_eq!(range.size, KDA_BLOCK_BYTES);
        let mut blk = vec![0u8; KDA_BLOCK_BYTES];
        fill_seeded(&mut blk, 50_000 + kda_block as u64);
        let digest = sha256(&blk);
        seed_mem.write_at(0, &blk);
        let (tx, rx) = mpsc::channel();
        executor
            .submit(
                Job {
                    direction: Direction::Write,
                    ios: vec![Io {
                        file_offset: range.offset as u64,
                        address: seed_mem.address(),
                        len: range.size,
                    }],
                },
                move |rep: JobReport| tx.send(rep.outcome).unwrap(),
            )
            .unwrap();
        rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        golden_kda.insert(kda_block, (blk, digest));
    }

    let writer = Arc::new(ChaosPageWriter::new(ring.clone()));
    let holder_push = Arc::new(PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 1).unwrap());
    holder_push
        .serve(PushSource {
            executor: executor.clone(),
            disk_layout: disk_layout.clone(),
            ring: ring.clone(),
            slot_bytes: PARENT_BYTES,
            writer: writer.clone(),
        })
        .unwrap();
    assert_eq!(holder_push.push_ring_slot_counts(), Some((20, 20)));

    let reader_push = Arc::new(PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 2).unwrap());
    let golden_mla = Arc::new(golden_mla);
    let golden_kda = Arc::new(golden_kda);

    // Fire 128 concurrent push jobs in 2 waves of 64 (saturating all 20 ring slots + 16 workers
    // + 64-entry queue while exercising ~30% mid-flight cancels & slow-writer stalls).
    const TOTAL_JOBS: usize = 128;
    const WAVE_SIZE: usize = 64;

    let succeeded_jobs = Arc::new(AtomicUsize::new(0));
    let cancelled_or_timed_out_jobs = Arc::new(AtomicUsize::new(0));
    let wire_cancel_jobs = Arc::new(AtomicUsize::new(0));
    let sha256_verified_blocks = Arc::new(AtomicUsize::new(0));
    let sha256_mismatches = Arc::new(AtomicUsize::new(0));
    let sentinel_bleeds = Arc::new(AtomicUsize::new(0));

    let phase1_start = Instant::now();
    for wave in 0..(TOTAL_JOBS / WAVE_SIZE) {
        let barrier = Arc::new(Barrier::new(WAVE_SIZE));
        let mut handles = Vec::with_capacity(WAVE_SIZE);
        for local_idx in 0..WAVE_SIZE {
            let job_idx = wave * WAVE_SIZE + local_idx;
            let holder_name = holder_push.name().to_owned();
            let reader_push = reader_push.clone();
            let writer = writer.clone();
            let golden_mla = golden_mla.clone();
            let golden_kda = golden_kda.clone();
            let barrier = barrier.clone();
            let succeeded_jobs = succeeded_jobs.clone();
            let cancelled_or_timed_out_jobs = cancelled_or_timed_out_jobs.clone();
            let wire_cancel_jobs = wire_cancel_jobs.clone();
            let sha256_verified_blocks = sha256_verified_blocks.clone();
            let sha256_mismatches = sha256_mismatches.clone();
            let sentinel_bleeds = sentinel_bleeds.clone();

            handles.push(thread::spawn(move || {
                // 30% chaos jobs (~38/128):
                // - ~24% tight budget (10..25ms) + moderate slow writer (20ms/block) -> holder deadline cancel
                // - ~6% tight budget (20ms) + 1.15s stalled RDMA write -> reader sends wire Message::Cancel!
                let is_wire_cancel = job_idx % 16 == 0; // 8 jobs (6.25%)
                let is_budget_cancel = !is_wire_cancel && (job_idx % 10 < 3); // ~30 jobs (~23.4%)
                let is_kda_mixed = job_idx % 4 == 1; // 25% push 24.84 MiB KDA + 2 MLA blocks!

                let mut items = Vec::new();
                let mut expected_blocks: Vec<(u32, i32, usize, usize)> = Vec::new();
                let mut dst_cursor = 0usize;

                if is_kda_mixed {
                    let kda_id = 5 + (job_idx % 4) as i32;
                    items.push(PushItem {
                        group: 1,
                        block: kda_id,
                        spans: vec![PushSpan {
                            offset: 0,
                            address: dst_cursor as u64,
                            length: KDA_BLOCK_BYTES as u64,
                        }],
                    });
                    expected_blocks.push((1, kda_id, dst_cursor, KDA_BLOCK_BYTES));
                    dst_cursor += KDA_BLOCK_BYTES;
                }
                let num_mla = if is_kda_mixed { 2 } else { 3 };
                for m in 0..num_mla {
                    let mla_id = ((job_idx * 3 + m) % 120 + 1) as i32;
                    items.push(PushItem {
                        group: 0,
                        block: mla_id,
                        spans: vec![PushSpan {
                            offset: 0,
                            address: dst_cursor as u64,
                            length: MLA_BLOCK_BYTES as u64,
                        }],
                    });
                    expected_blocks.push((0, mla_id, dst_cursor, MLA_BLOCK_BYTES));
                    dst_cursor += MLA_BLOCK_BYTES;
                }

                // Append a 4 KiB trailing sentinel guard to catch any destination overrun.
                let guard_offset = dst_cursor;
                let total_buf_bytes = dst_cursor + ALIGNMENT;
                let delay = if is_wire_cancel {
                    Duration::from_millis(1_150)
                } else if is_budget_cancel {
                    Duration::from_millis(20)
                } else if job_idx % 7 == 0 {
                    Duration::from_millis(4) // gentle slow writer on some healthy jobs
                } else {
                    Duration::ZERO
                };
                let budget = if is_wire_cancel {
                    Duration::from_millis(20)
                } else if is_budget_cancel {
                    Duration::from_millis(12)
                } else {
                    Duration::from_secs(30)
                };

                let session = format!("reader-job-{job_idx}");
                let landing = writer.register(session.clone(), total_buf_bytes, delay);
                barrier.wait();

                let result = reader_push.push(&holder_name, &session, budget, items);
                // Fence immediately when `push()` returns — simulating reader quarantine / page reuse!
                landing.fenced.store(true, Ordering::Release);

                match result {
                    Ok(()) => {
                        succeeded_jobs.fetch_add(1, Ordering::Relaxed);
                        let buf = landing.buffer.lock();
                        for &(group, blk_id, off, len) in &expected_blocks {
                            let actual = &buf[off..off + len];
                            let (expected_bytes, expected_sha) = if group == 0 {
                                &golden_mla[&blk_id]
                            } else {
                                &golden_kda[&blk_id]
                            };
                            if actual != expected_bytes.as_slice()
                                || sha256(actual) != *expected_sha
                            {
                                sha256_mismatches.fetch_add(1, Ordering::Relaxed);
                            } else {
                                sha256_verified_blocks.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        // Guard tail must still hold exact 0xCAFEBABE sentinel.
                        let guard = &buf[guard_offset..guard_offset + ALIGNMENT];
                        if !guard.chunks_exact(4).all(|c| c == SENTINEL) {
                            sentinel_bleeds.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(_err) => {
                        cancelled_or_timed_out_jobs.fetch_add(1, Ordering::Relaxed);
                        if is_wire_cancel {
                            wire_cancel_jobs.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                // Wait briefly before unregistering so any buggy post-return late write would trip `late_fenced_writes`!
                thread::sleep(Duration::from_millis(20));
                writer.unregister(&session);
            }));
        }
        for h in handles {
            h.join().expect("push chaos thread panicked");
        }
    }
    let phase1_elapsed = phase1_start.elapsed();

    // Wait up to 2s for holder worker teardown of any trailing cancelled jobs (must already be 0!).
    let wait_slots = Instant::now();
    while holder_push.push_ring_slot_counts() != Some((PUSH_RING_SLOTS, PUSH_RING_SLOTS))
        || holder_push.active_push_jobs() != 0
    {
        assert!(
            wait_slots.elapsed() < Duration::from_secs(5),
            "push_ring slot or job leak: slots={:?}, active_jobs={}",
            holder_push.push_ring_slot_counts(),
            holder_push.active_push_jobs()
        );
        thread::sleep(Duration::from_millis(5));
    }

    let p1_ok = succeeded_jobs.load(Ordering::Relaxed);
    let p1_cancel = cancelled_or_timed_out_jobs.load(Ordering::Relaxed);
    let p1_wire_cancel = wire_cancel_jobs.load(Ordering::Relaxed);
    let p1_late_fenced = writer.late_fenced_writes.load(Ordering::Relaxed);
    let p1_sha_ok = sha256_verified_blocks.load(Ordering::Relaxed);
    let p1_sha_err = sha256_mismatches.load(Ordering::Relaxed);
    let p1_bleeds = sentinel_bleeds.load(Ordering::Relaxed);
    let p1_bytes_gib = writer.total_bytes_written.load(Ordering::Relaxed) as f64 / (1u64 << 30) as f64;
    let p1_slots = holder_push.push_ring_slot_counts().unwrap();

    println!(
        "TS-A3-PHASE1: jobs={TOTAL_JOBS} ok={p1_ok} cancelled_or_rejected={p1_cancel} (wire_cancel={p1_wire_cancel}) sha256_blocks_ok={p1_sha_ok} sha256_err={p1_sha_err} sentinel_bleed={p1_bleeds} late_fenced_writes={p1_late_fenced} push_ring_slots={}/{} active_jobs={} max_concurrent_writers={} transferred_gib={:.2} elapsed={:.2}s",
        p1_slots.0,
        p1_slots.1,
        holder_push.active_push_jobs(),
        writer.max_concurrent_writers.load(Ordering::Relaxed),
        p1_bytes_gib,
        phase1_elapsed.as_secs_f64(),
    );

    assert_eq!(p1_ok + p1_cancel, TOTAL_JOBS);
    assert!(p1_ok >= 70, "expected >= 70 successful pushes, got {p1_ok}");
    assert!(
        p1_cancel >= 30,
        "expected >= 30 cancelled/timed-out pushes, got {p1_cancel}"
    );
    assert!(p1_wire_cancel >= 4, "expected wire Message::Cancel paths exercised");
    assert_eq!(p1_late_fenced, 0, "FENCED LATE WRITE DETECTED!");
    assert_eq!(p1_sha_err, 0, "BITWISE SHA-256 MISMATCH IN PUSH!");
    assert_eq!(p1_bleeds, 0, "SENTINEL BLEED IN PUSH LANDING!");
    assert_eq!(p1_slots, (PUSH_RING_SLOTS, PUSH_RING_SLOTS));
    assert_eq!(holder_push.active_push_jobs(), 0);

    // Tear down Phase 1 endpoints cleanly before Phase 2.
    drop(reader_push);
    drop(holder_push);
    executor.close(Duration::from_secs(10)).unwrap();

    // =========================================================================
    // PHASE 2: 8 Consecutive Real OS I/O Errors -> ts-disk degraded() == true
    //          -> Mirrored-Rank Agreement.degraded -> Scheduler InvalidateDiskCache
    //          -> ZMQ ClearMedium("DISK") Emission + Zero-Leak Audit.
    // =========================================================================
    let scratch_p2 = ScratchFile::new("ts-a3-degraded-loop");
    let host_p2 = Memory::map(SMALL_HOST_BLOCKS * SMALL_BLOCK_BYTES).unwrap();
    let mut rank0_disk = DiskTier::start(
        &DiskSettings {
            file: scratch_p2.0.clone(),
            file_bytes: (SMALL_DISK_BLOCKS * SMALL_BLOCK_BYTES) as u64,
            executor: ExecutorConfig {
                backend: Backend::IoUring,
                queue_depth: 4,
                write_queue_depth: 2,
                max_io_bytes: 1 << 20,
                write_bytes_per_second: 1 << 30,
            },
            host: HostBuffer {
                base: host_p2.address(),
                len: host_p2.size(),
            },
            host_layout: small_layout(SMALL_HOST_BLOCKS),
            disk_layout: small_layout(SMALL_DISK_BLOCKS),
            max_operations: 4,
            push_ring_bytes: PUSH_RING_SLOTS * SMALL_BLOCK_BYTES,
        },
        None,
    )
    .unwrap();

    let (p2_exec, p2_disk_layout, p2_ring, p2_slot_bytes) = rank0_disk.push_source().unwrap();
    let p2_holder_push = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 3).unwrap();
    let p2_reader_push = PushEndpoint::bind("127.0.0.1", 0, "127.0.0.1", 4).unwrap();
    let p2_writer = Arc::new(ChaosPageWriter::new(p2_ring.clone()));
    p2_holder_push
        .serve(PushSource {
            executor: p2_exec.clone(),
            disk_layout: p2_disk_layout,
            ring: p2_ring,
            slot_bytes: p2_slot_bytes,
            writer: p2_writer,
        })
        .unwrap();
    assert_eq!(
        p2_holder_push.push_ring_slot_counts(),
        Some((PUSH_RING_SLOTS, PUSH_RING_SLOTS))
    );

    // Mirrored replica: Rank 0 (faulting rank) + Rank 1 (healthy sibling).
    let mut rank0_sched = mirrored_scheduler("rank0");
    let mut rank1_sched = mirrored_scheduler("rank1");
    let scratch_r1 = ScratchFile::new("ts-a3-rank1-healthy");
    let host_r1 = Memory::map(SMALL_HOST_BLOCKS * SMALL_BLOCK_BYTES).unwrap();
    let mut rank1_disk = DiskTier::start(
        &DiskSettings {
            file: scratch_r1.0.clone(),
            file_bytes: (SMALL_DISK_BLOCKS * SMALL_BLOCK_BYTES) as u64,
            executor: ExecutorConfig {
                backend: Backend::Psync,
                queue_depth: 4,
                write_queue_depth: 2,
                max_io_bytes: 1 << 20,
                write_bytes_per_second: 1 << 30,
            },
            host: HostBuffer {
                base: host_r1.address(),
                len: host_r1.size(),
            },
            host_layout: small_layout(SMALL_HOST_BLOCKS),
            disk_layout: small_layout(SMALL_DISK_BLOCKS),
            max_operations: 4,
            push_ring_bytes: 0,
        },
        None,
    )
    .unwrap();

    // Start real ZMQ KV-event publisher + subscriber on dp_rank=0.
    let (zmq_port, replay_port) = (free_port(), free_port());
    let publisher = KvEventPublisher::start(
        &KvEventsConfig {
            endpoint: format!("tcp://*:{zmq_port}"),
            replay_endpoint: Some(format!("tcp://*:{replay_port}")),
            buffer_steps: 16,
            max_queue_size: 64,
            topic: "kv".to_owned(),
        },
        0,
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut subscriber = SubSocket::new();
    rt.block_on(async {
        subscriber
            .connect(&format!("tcp://127.0.0.1:{zmq_port}"))
            .await
            .unwrap();
        subscriber.subscribe("").await.unwrap();
    });
    // Wait until PUB/SUB handshake settles by publishing an empty keepalive/dummy batch or short sleep.
    thread::sleep(Duration::from_millis(50));

    // Helper: run one synchronized mirrored round across rank 0 and rank 1.
    let mut r0_acks: Vec<CacheEvent> = Vec::new();
    let mut r1_acks: Vec<CacheEvent> = Vec::new();
    let sync_round = |r0_sched: &mut Scheduler,
                          r1_sched: &mut Scheduler,
                          r0_disk: &mut DiskTier,
                          r1_disk: &mut DiskTier,
                          r0_acks: &mut Vec<CacheEvent>,
                          r1_acks: &mut Vec<CacheEvent>|
     -> (bool, Vec<(String, i32)>) {
        let rep0 = r0_disk.report();
        let rep1 = r1_disk.report();
        let st0 = r0_disk.agree(&[rep0.clone(), rep1.clone()], true, true).unwrap();
        let st1 = r1_disk.agree(&[rep0, rep1], true, true).unwrap();
        assert_eq!(st0.disk_degraded, st1.disk_degraded, "mirrored ranks diverged on disk_degraded!");
        assert_eq!(st0.disk_done.len(), st1.disk_done.len());

        r0_sched.advance_head(&st0, &std::mem::take(r0_acks)).unwrap();
        r1_sched.advance_head(&st1, &std::mem::take(r1_acks)).unwrap();

        let plan0 = r0_sched.next_execution_plan(true, false);
        let plan1 = r1_sched.next_execution_plan(true, false);
        r0_disk.submit(&plan0).unwrap();
        r1_disk.submit(&plan1).unwrap();

        let to_acks = |ops: Vec<ts_scheduler::CacheOpView>| -> Vec<CacheEvent> {
            ops.into_iter()
                .map(|op| CacheEvent {
                    kind: match op.kind {
                        CacheOpKind::WriteBack => CacheEventKind::WriteBackDone,
                        CacheOpKind::LoadBack => CacheEventKind::LoadBackDone,
                        CacheOpKind::SnapshotStore => CacheEventKind::SnapshotDone,
                        CacheOpKind::SnapshotRestore => CacheEventKind::RestoreDone,
                        _ => unreachable!(),
                    },
                    op_id: op.op_id,
                })
                .collect()
        };
        *r0_acks = to_acks(plan0.cache_ops());
        *r1_acks = to_acks(plan1.cache_ops());

        let rows = plan0
            .forward_batches()
            .into_iter()
            .flat_map(|b| {
                b.request_ids
                    .into_iter()
                    .zip(b.extend_prefix_lens)
                    .take(b.num_extends)
                    .collect::<Vec<_>>()
            })
            .collect();
        (st0.disk_degraded, rows)
    };

    // Seed + evict to Disk on both mirrored ranks so Disk holds live blocks while Device/Host hold others.
    let run_req = |rid: &str,
                   tokens: Vec<i32>,
                   r0_sched: &mut Scheduler,
                   r1_sched: &mut Scheduler,
                   r0_disk: &mut DiskTier,
                   r1_disk: &mut DiskTier,
                   r0_acks: &mut Vec<CacheEvent>,
                   r1_acks: &mut Vec<CacheEvent>| {
        let spec = RequestSpec {
            admission_id: 0,
            request_id: rid.to_owned(),
            session_id: String::new(),
            cache_group_id: String::new(),
            shared_prefix_tokens: Vec::new(),
            peer_holder: String::new(),
            tokens,
            max_new_tokens: 1,
            max_cached_prefix_tokens: i32::MAX,
        };
        r0_sched.submit_requests(&[spec.clone()]).unwrap();
        r1_sched.submit_requests(&[spec]).unwrap();
        let t0 = Instant::now();
        loop {
            assert!(t0.elapsed() < Duration::from_secs(10), "{rid} stalled");
            let (_, rows) = sync_round(r0_sched, r1_sched, r0_disk, r1_disk, r0_acks, r1_acks);
            if rows.iter().any(|(r, _)| r == rid) {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let finish_ev = [
            ForwardEvent {
                kind: ForwardEventKind::ExtendResult,
                request_id: rid.to_owned(),
                tokens: vec![999],
                spec_candidate_ids: Vec::new(),
                reserve_num_tokens: 0,
                bootstrap_token: -1,
            },
            ForwardEvent {
                kind: ForwardEventKind::Finish,
                request_id: rid.to_owned(),
                tokens: Vec::new(),
                spec_candidate_ids: Vec::new(),
                reserve_num_tokens: 0,
                bootstrap_token: -1,
            },
        ];
        r0_sched.advance(&finish_ev).unwrap();
        r1_sched.advance(&finish_ev).unwrap();
    };

    run_req(
        "seed",
        (1..=8).collect(),
        &mut rank0_sched,
        &mut rank1_sched,
        &mut rank0_disk,
        &mut rank1_disk,
        &mut r0_acks,
        &mut r1_acks,
    );
    for (idx, base) in [100, 200, 300].into_iter().enumerate() {
        run_req(
            &format!("fill-{idx}"),
            (base..base + 23).collect(),
            &mut rank0_sched,
            &mut rank1_sched,
            &mut rank0_disk,
            &mut rank1_disk,
            &mut r0_acks,
            &mut r1_acks,
        );
        let t0 = Instant::now();
        while rank0_sched.disk_stats().operations > 0
            || rank0_sched.disk_stats().queued_cleanings > 0
            || !r0_acks.is_empty()
        {
            assert!(t0.elapsed() < Duration::from_secs(10));
            sync_round(
                &mut rank0_sched,
                &mut rank1_sched,
                &mut rank0_disk,
                &mut rank1_disk,
                &mut r0_acks,
                &mut r1_acks,
            );
            thread::sleep(Duration::from_millis(1));
        }
    }
    let pre_r0 = rank0_sched.disk_stats();
    let pre_r1 = rank1_sched.disk_stats();
    assert!(pre_r0.disk_blocks > 0, "expected Disk blocks before fault: {pre_r0:?}");
    assert_eq!(pre_r0.disk_blocks, pre_r1.disk_blocks, "mirrored ranks diverged before fault");
    // Drain any pre-fault KV events so only post-degraded events are observed next.
    let _ = rank0_sched.drain_kv_events();

    // Inject 8 REAL consecutive OS I/O errors (ENOSPC = errno 28) on Rank 0's Executor via dup2(/dev/full, fd).
    inject_kernel_enospc_on_fd(p2_exec.raw_fd());
    for err_i in 1..=8usize {
        let (tx, rx) = mpsc::channel();
        p2_exec
            .submit(
                Job {
                    direction: Direction::Write,
                    ios: vec![Io {
                        file_offset: 0,
                        address: host_p2.address(),
                        len: SMALL_BLOCK_BYTES,
                    }],
                },
                move |rep: JobReport| tx.send(rep.outcome).unwrap(),
            )
            .expect("submit fault probe before 8th error");
        let outcome = rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(
            outcome,
            Err(IoError::Os(28)), // ENOSPC from /dev/full
            "expected real kernel ENOSPC (28) on fault probe #{err_i}, got {outcome:?}"
        );
        if err_i < 8 {
            assert!(
                !p2_exec.degraded(),
                "breaker tripped prematurely at error #{err_i}"
            );
        } else {
            assert!(p2_exec.degraded(), "breaker failed to trip on 8th OS error!");
        }
    }
    assert!(rank0_disk.degraded());
    assert!(!rank1_disk.degraded(), "rank 1 local executor is healthy");

    // Verify PushEndpoint fast-fails on degraded holder and returns 20/20 slots!
    let degraded_push_res = p2_reader_push.push(
        p2_holder_push.name(),
        "degraded-probe",
        Duration::from_secs(2),
        vec![PushItem {
            group: 0,
            block: 1,
            spans: vec![PushSpan {
                offset: 0,
                address: 0,
                length: SMALL_BLOCK_BYTES as u64,
            }],
        }],
    );
    assert!(
        degraded_push_res.is_err(),
        "push against degraded holder must fast-fail"
    );
    assert_eq!(
        p2_holder_push.push_ring_slot_counts(),
        Some((PUSH_RING_SLOTS, PUSH_RING_SLOTS))
    );

    // Next mirrored agreement round MUST propagate degraded across BOTH ranks once!
    let (agreed_degraded_r1, _) = sync_round(
        &mut rank0_sched,
        &mut rank1_sched,
        &mut rank0_disk,
        &mut rank1_disk,
        &mut r0_acks,
        &mut r1_acks,
    );
    assert!(agreed_degraded_r1, "mirrored ranks must agree disk_degraded == true");
    assert!(rank0_disk.agreed_degraded());
    assert!(rank1_disk.agreed_degraded());

    // Subsequent round MUST NOT re-fire disk_degraded (latched once).
    let (agreed_degraded_r2, _) = sync_round(
        &mut rank0_sched,
        &mut rank1_sched,
        &mut rank0_disk,
        &mut rank1_disk,
        &mut r0_acks,
        &mut r1_acks,
    );
    assert!(!agreed_degraded_r2, "disk_degraded must latch and fire once");

    // Verify BOTH schedulers dropped Disk index cleanly with zero leaked resources.
    let post_r0 = rank0_sched.disk_stats();
    let post_r1 = rank1_sched.disk_stats();
    let peer_r0 = rank0_sched.peer_stats();
    let peer_r1 = rank1_sched.peer_stats();
    assert_eq!(post_r0.disk_blocks, 0);
    assert_eq!(post_r1.disk_blocks, 0);
    assert_eq!(
        (
            post_r0.operations,
            post_r0.held_host_blocks,
            post_r0.waiting_requests,
            post_r0.ready_requests
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(
        (
            post_r1.operations,
            post_r1.held_host_blocks,
            post_r1.waiting_requests,
            post_r1.ready_requests
        ),
        (0, 0, 0, 0)
    );
    assert_eq!((peer_r0.leases, peer_r0.pinned_blocks), (0, 0));
    assert_eq!((peer_r1.leases, peer_r1.pinned_blocks), (0, 0));

    // Drain KV event batch from Rank 0, publish over REAL ZMQ PUB/SUB, and verify ClearMedium("DISK") wire bytes!
    let batch: KvEventBatch = rank0_sched.drain_kv_events();
    let cleared: Vec<_> = batch.cleared_media().collect();
    assert_eq!(cleared, vec![ts_scheduler::KvEventMedium::Disk]);

    // Publish until SUB receives (PUB drops frames before SUB connects; republish same ClearMedium batch cleanly if needed).
    let received_frames = loop {
        let publish_batch = KvEventBatch::with_cleared_media(&cleared, &[]);
        publisher.publish(publish_batch);
        let got = rt.block_on(async {
            tokio::time::timeout(Duration::from_millis(200), subscriber.recv()).await
        });
        if let Ok(Ok(msg)) = got {
            break msg.into_vec();
        }
    };
    assert_eq!(received_frames.len(), 3);
    assert_eq!(received_frames[0], Bytes::from_static(b"kv"));
    let payload = received_frames[2].as_ref();
    let contains = |needle: &[u8]| payload.windows(needle.len()).any(|w| w == needle);
    assert!(contains(b"ClearMedium"), "missing ClearMedium tag on ZMQ wire");
    assert!(contains(b"DISK"), "missing DISK medium on ZMQ wire");
    // Exact Dynamo zmq_wire tag sequence: fixarray(2) [ fixstr(11) "ClearMedium", fixstr(4) "DISK" ]
    let mut exact_clear_event = vec![0x92, 0xab];
    exact_clear_event.extend_from_slice(b"ClearMedium");
    exact_clear_event.push(0xa4);
    exact_clear_event.extend_from_slice(b"DISK");
    assert!(
        contains(&exact_clear_event),
        "ClearMedium(\"DISK\") msgpack encoding did not match Dynamo zmq_wire exact bytes!"
    );

    println!(
        "TS-A3-PHASE2: injected_os_errors=8 rank0_degraded={} rank1_local_degraded={} mirrored_agreed_once=true pre_disk_blocks={} post_disk_blocks=(r0={},r1={}) leaked_ops=0 leaked_host_blocks=0 leaked_leases=0 push_ring_slots={}/{} zmq_clear_medium_disk_wire_match=EXACT ({} bytes)",
        rank0_disk.degraded(),
        rank1_disk.degraded(),
        pre_r0.disk_blocks,
        post_r0.disk_blocks,
        post_r1.disk_blocks,
        p2_holder_push.push_ring_slot_counts().unwrap().0,
        p2_holder_push.push_ring_slot_counts().unwrap().1,
        payload.len(),
    );
    println!("TS-A3 PASS: 0 deadlocks, 0 fenced late writes, 20/20 push_ring slots returned, 0 leaked leases/blocks, 0-bit SHA-256 verified, full 8-error degraded -> ClearMedium(\"DISK\") ZMQ loop verified");

    drop(p2_reader_push);
    drop(p2_holder_push);
    rank0_disk.close(Duration::from_secs(5)).unwrap();
    rank1_disk.close(Duration::from_secs(5)).unwrap();
}
