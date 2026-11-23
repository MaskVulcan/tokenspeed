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

//! What the backends share: the jobs in flight, their chunks waiting to be
//! issued, and the pacing of writes.
//!
//! One lock covers it all. A chunk is one request to the kernel; a job
//! reports once its last chunk retired. Reads are issued ahead of writes;
//! writes first wait in a backlog until the byte budget lets them through,
//! then take at most `write_depth` of the `depth` requests in flight.

use std::{
    collections::{BTreeMap, VecDeque},
    os::fd::{AsRawFd, OwnedFd},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use parking_lot::{Condvar, Mutex};

use crate::{
    checksum::{crc32c_combine, ChecksumTable},
    executor::{Direction, IoError, JobReport, Pressure, Stats},
};

/// Consecutive OS-level I/O errors before tripping the disk circuit breaker.
pub(crate) const MAX_CONSECUTIVE_DISK_ERRORS: u32 = 8;

/// Read-latency AIMD sampling window duration.
pub const AIMD_WINDOW: Duration = Duration::from_millis(100);
/// Minimum read chunk samples required before an AIMD window adjustment.
pub const AIMD_MIN_SAMPLES: usize = 8;
const AIMD_MAX_SAMPLES: usize = 64;
/// Read p99 threshold (microseconds) above which AIMD halves write concurrency.
pub const AIMD_HIGH_US: u32 = 8_000;
/// Read p99 threshold (microseconds) below which AIMD increments write concurrency.
pub const AIMD_LOW_US: u32 = 4_000;

/// What a job runs once it ended: its report.
pub(crate) type Done = Box<dyn FnOnce(JobReport) + Send>;

/// Jobs that ended under the lock, reported after it is released.
pub(crate) type Finished = Vec<(Done, JobReport)>;

pub(crate) fn report(finished: Finished) {
    for (done, report) in finished {
        done(report);
    }
}

/// One request to the kernel: `len` bytes between `file_offset` and
/// `address`, all inside the fixed buffer `buffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub job: u64,
    pub direction: Direction,
    pub file_offset: u64,
    pub address: usize,
    pub len: usize,
    pub buffer: u16,
    /// Index of the logical `Io` in the owning job.
    pub io_index: usize,
    /// Sequential chunk index within that logical `Io` (for CRC32C combine).
    pub chunk_index: usize,
    /// When the chunk was dispatched to the backend (for read-latency AIMD).
    pub issued_at: Option<Instant>,
}

/// Logical `Io` bounds before `Pieces::split`, used to combine per-chunk
/// CRC32Cs back to the logical Disk block range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IoPlan {
    pub file_offset: u64,
    pub len: usize,
    pub chunks: usize,
}

struct IoChecksumState {
    file_offset: u64,
    len: usize,
    chunk_crcs: Vec<Option<(u32, usize)>>,
    remaining: usize,
    failed: bool,
}

impl IoChecksumState {
    fn new(plan: IoPlan) -> Self {
        Self {
            file_offset: plan.file_offset,
            len: plan.len,
            chunk_crcs: vec![None; plan.chunks],
            remaining: plan.chunks,
            failed: false,
        }
    }

    fn combined_crc(&self) -> Option<u32> {
        let mut combined = 0u32;
        let mut first = true;
        for entry in &self.chunk_crcs {
            let (crc, len) = (*entry)?;
            if first {
                combined = crc;
                first = false;
            } else {
                combined = crc32c_combine(combined, crc, len);
            }
        }
        (!first).then_some(combined)
    }
}

struct JobEntry {
    direction: Direction,
    bytes: u64,
    submitted_at: Instant,
    /// When its first chunk went to the kernel.
    issued_at: Option<Instant>,
    queued: usize,
    inflight: usize,
    ios: Vec<IoChecksumState>,
    error: Option<IoError>,
    cancelled: bool,
    done: Option<Done>,
}

impl JobEntry {
    /// The job's outcome once nothing of it is queued or in flight.
    fn outcome(&mut self) -> Result<(), IoError> {
        match self.error.take() {
            Some(error) => Err(error),
            None if self.cancelled => Err(IoError::Cancelled),
            None => Ok(()),
        }
    }
}

/// The requests in flight at once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    pub depth: usize,
    pub write_depth: usize,
}

/// Closed-loop AIMD controller that halves background write concurrency when
/// foreground read p99 exceeds [`AIMD_HIGH_US`] and recovers additively once
/// read p99 drops below [`AIMD_LOW_US`].
pub struct ReadLatencyAimd {
    samples_us: [u32; AIMD_MAX_SAMPLES],
    len: usize,
    cursor: usize,
    window_start: Option<Instant>,
    effective_write_depth: Option<usize>,
}

impl Default for ReadLatencyAimd {
    fn default() -> Self {
        Self {
            samples_us: [0; AIMD_MAX_SAMPLES],
            len: 0,
            cursor: 0,
            window_start: None,
            effective_write_depth: None,
        }
    }
}

impl ReadLatencyAimd {
    /// Construct a fresh AIMD controller starting at full configured write depth.
    pub fn new() -> Self {
        Self::default()
    }

    /// Effective background write concurrency for `max_write_depth`.
    /// When foreground reads are completely idle (`reads_pending == false`),
    /// returns the full `max_write_depth` (background writes remain rate-paced
    /// by `TokenBucket` and `IOPRIO_WRITE` idle priority).
    pub fn effective_write_depth_for(&self, max_write_depth: usize, reads_pending: bool) -> usize {
        if !reads_pending {
            return max_write_depth;
        }
        self.effective_write_depth
            .unwrap_or(max_write_depth)
            .min(max_write_depth)
    }

    pub(crate) fn effective_write_depth(&self, limits: Limits, reads_pending: bool) -> usize {
        self.effective_write_depth_for(limits.write_depth, reads_pending)
    }

    /// Record one completed read chunk latency sample at timestamp `now`.
    pub fn record_read(&mut self, latency: Duration, now: Instant, max_write_depth: usize) {
        let window_start = *self.window_start.get_or_insert(now);
        let us = u32::try_from(latency.as_micros()).unwrap_or(u32::MAX);
        if self.len < AIMD_MAX_SAMPLES {
            self.samples_us[self.len] = us;
            self.len += 1;
        } else {
            self.samples_us[self.cursor] = us;
            self.cursor = (self.cursor + 1) % AIMD_MAX_SAMPLES;
        }
        if now.saturating_duration_since(window_start) < AIMD_WINDOW || self.len < AIMD_MIN_SAMPLES
        {
            return;
        }
        self.samples_us[..self.len].sort_unstable();
        let p99_idx = (self.len * 99) / 100;
        let p99_us = self.samples_us[p99_idx];
        let cap = max_write_depth.max(1);
        let cur = self.effective_write_depth.unwrap_or(cap).min(cap);
        let next = if p99_us > AIMD_HIGH_US {
            cur / 2
        } else if p99_us < AIMD_LOW_US {
            (cur + 1).min(cap)
        } else {
            cur
        };
        if next != cur {
            tracing::debug!(
                read_p99_us = p99_us,
                prev_write_depth = cur,
                next_write_depth = next,
                "adjusted disk write concurrency via read-p99 AIMD"
            );
        }
        self.effective_write_depth = Some(next);
        self.len = 0;
        self.cursor = 0;
        self.window_start = Some(now);
    }
}

#[derive(Default)]
pub(crate) struct Queues {
    jobs: BTreeMap<u64, JobEntry>,
    reads: VecDeque<Chunk>,
    writes: VecDeque<Chunk>,
    backlog: VecDeque<Chunk>,
    inflight_reads: usize,
    inflight_writes: usize,
    inflight_bytes: u64,
    next_job: u64,
    closing: bool,
    broken: bool,
    degraded: bool,
    consecutive_os_errors: u32,
    last_write_depth: usize,
    aimd: ReadLatencyAimd,
    stats: Stats,
}

impl Queues {
    /// Queue a job's chunks and logical `Io` plans; returns its id.
    pub(crate) fn admit(
        &mut self,
        direction: Direction,
        ios: Vec<IoPlan>,
        chunks: impl IntoIterator<Item = Chunk>,
        done: Done,
        now: Instant,
    ) -> u64 {
        self.next_job += 1;
        let job = self.next_job;
        let mut queued = 0;
        let mut bytes = 0;
        for chunk in chunks {
            let chunk = Chunk { job, ..chunk };
            queued += 1;
            bytes += chunk.len as u64;
            match direction {
                Direction::Read => self.reads.push_back(chunk),
                Direction::Write => self.backlog.push_back(chunk),
            }
        }
        self.jobs.insert(
            job,
            JobEntry {
                direction,
                bytes,
                submitted_at: now,
                issued_at: None,
                queued,
                inflight: 0,
                ios: ios.into_iter().map(IoChecksumState::new).collect(),
                error: None,
                cancelled: false,
                done: Some(done),
            },
        );
        job
    }

    /// The next chunk to issue within `limits`, now counted in flight.
    pub(crate) fn next(&mut self, limits: Limits) -> Option<Chunk> {
        if self.broken
            || self.degraded
            || self.inflight_reads + self.inflight_writes >= limits.depth
        {
            return None;
        }
        self.last_write_depth = limits.write_depth;
        let reads_pending = !self.reads.is_empty() || self.inflight_reads > 0;
        let write_depth = self.aimd.effective_write_depth(limits, reads_pending);
        let mut chunk = match self.reads.pop_front() {
            Some(chunk) => chunk,
            None if self.inflight_writes < write_depth => self.writes.pop_front()?,
            None => return None,
        };
        let now = Instant::now();
        chunk.issued_at = Some(now);
        match chunk.direction {
            Direction::Read => self.inflight_reads += 1,
            Direction::Write => self.inflight_writes += 1,
        }
        self.inflight_bytes += chunk.len as u64;
        let entry = self.jobs.get_mut(&chunk.job).expect("a queued chunk's job");
        entry.queued -= 1;
        entry.inflight += 1;
        entry.issued_at.get_or_insert(now);
        Some(chunk)
    }

    /// A chunk retired with `result` and optional chunk CRC32C; combines chunk
    /// CRCs per logical `Io` into `checksums` and returns all jobs that ended
    /// (usually 0 or 1, or more if this error tripped the circuit breaker).
    pub(crate) fn finish_with_checksum(
        &mut self,
        chunk: Chunk,
        result: Result<(), IoError>,
        chunk_crc: Option<u32>,
        checksums: &ChecksumTable,
    ) -> Finished {
        self.finish_at(chunk, result, chunk_crc, Some(checksums), Instant::now())
    }

    #[cfg(test)]
    pub(crate) fn finish(&mut self, chunk: Chunk, result: Result<(), IoError>) -> Finished {
        self.finish_at(chunk, result, None, None, Instant::now())
    }

    fn finish_at(
        &mut self,
        chunk: Chunk,
        mut result: Result<(), IoError>,
        chunk_crc: Option<u32>,
        checksums: Option<&ChecksumTable>,
        now: Instant,
    ) -> Finished {
        match chunk.direction {
            Direction::Read => self.inflight_reads -= 1,
            Direction::Write => self.inflight_writes -= 1,
        }
        self.inflight_bytes -= chunk.len as u64;
        let mut tripped_errno = None;
        match &result {
            Ok(()) => {
                self.consecutive_os_errors = 0;
                if chunk.direction == Direction::Read {
                    if let Some(issued) = chunk.issued_at {
                        self.aimd.record_read(
                            now.saturating_duration_since(issued),
                            now,
                            self.last_write_depth,
                        );
                    }
                }
            }
            Err(IoError::Os(errno)) => {
                self.consecutive_os_errors = self.consecutive_os_errors.saturating_add(1);
                if !self.degraded && self.consecutive_os_errors >= MAX_CONSECUTIVE_DISK_ERRORS {
                    tripped_errno = Some(*errno);
                }
            }
            Err(_) => {}
        }
        let entry = self
            .jobs
            .get_mut(&chunk.job)
            .expect("an issued chunk's job");
        entry.inflight -= 1;
        if let Some(io_state) = entry.ios.get_mut(chunk.io_index) {
            if result.is_ok() {
                if let (Some(crc), Some(slot)) =
                    (chunk_crc, io_state.chunk_crcs.get_mut(chunk.chunk_index))
                {
                    *slot = Some((crc, chunk.len));
                }
                io_state.remaining = io_state.remaining.saturating_sub(1);
                if io_state.remaining == 0 && !io_state.failed {
                    if let (Some(table), Some(combined)) = (checksums, io_state.combined_crc()) {
                        match chunk.direction {
                            Direction::Write => {
                                if !entry.cancelled && entry.error.is_none() {
                                    table.record_crc(io_state.file_offset, io_state.len, combined);
                                }
                            }
                            Direction::Read => {
                                if let Err((expected, actual)) =
                                    table.verify_crc(io_state.file_offset, io_state.len, combined)
                                {
                                    tracing::error!(
                                        file_offset = io_state.file_offset,
                                        len = io_state.len,
                                        expected = format_args!("{expected:#010x}"),
                                        actual = format_args!("{actual:#010x}"),
                                        "disk read CRC32C mismatch; dropping corrupted block"
                                    );
                                    result = Err(IoError::ChecksumMismatch { expected, actual });
                                }
                            }
                        }
                    }
                }
            } else {
                io_state.failed = true;
                if chunk.direction == Direction::Write {
                    if let Some(table) = checksums {
                        table.invalidate(io_state.file_offset, io_state.len);
                    }
                }
            }
        }
        if let Err(error) = result {
            entry.error.get_or_insert(error);
            // Its other chunks are not issued: the job failed already.
            if entry.queued > 0 {
                entry.queued = 0;
                self.drop_queued(chunk.job);
            }
        }
        let mut finished = Vec::new();
        if let Some(done) = self.end_if_retired(chunk.job) {
            finished.push(done);
        }
        if let Some(errno) = tripped_errno {
            finished.extend(self.trip_degraded(errno));
        }
        finished
    }

    /// Trip the circuit breaker after consecutive OS I/O errors: stop issuing
    /// new work and fail all queued unissued chunks while letting in-flight
    /// requests retire safely.
    fn trip_degraded(&mut self, errno: i32) -> Finished {
        self.degraded = true;
        tracing::error!(
            consecutive_errors = self.consecutive_os_errors,
            errno,
            "disk tier hit consecutive OS I/O error threshold; tripping circuit breaker into degraded mode"
        );
        let jobs: Vec<u64> = self.jobs.keys().copied().collect();
        let mut finished = Vec::new();
        for job in jobs {
            if let Some(entry) = self.jobs.get_mut(&job) {
                entry.error.get_or_insert(IoError::Os(errno));
                if entry.queued > 0 {
                    entry.queued = 0;
                    self.drop_queued(job);
                }
            }
            if let Some(done) = self.end_if_retired(job) {
                finished.push(done);
            }
        }
        finished
    }

    /// Stop issuing `job`'s chunks; returns it if nothing of it is in flight.
    pub(crate) fn cancel(&mut self, job: u64) -> Option<(Done, JobReport)> {
        let entry = self.jobs.get_mut(&job)?;
        entry.cancelled = true;
        if entry.queued > 0 {
            entry.queued = 0;
            self.drop_queued(job);
        }
        self.end_if_retired(job)
    }

    /// Refuse new jobs and cancel every queued chunk; what is in flight
    /// still retires.
    pub(crate) fn close(&mut self) -> Finished {
        self.closing = true;
        let jobs: Vec<u64> = self.jobs.keys().copied().collect();
        jobs.into_iter()
            .filter_map(|job| self.cancel(job))
            .collect()
    }

    /// The backend can no longer reap what it issued: issue nothing more.
    /// The jobs in flight never report, so whatever they name stays held.
    pub(crate) fn break_down(&mut self) -> Finished {
        self.broken = true;
        self.close()
    }

    pub(crate) fn closing(&self) -> bool {
        self.closing
    }

    pub(crate) fn broken(&self) -> bool {
        self.broken
    }

    pub(crate) fn degraded(&self) -> bool {
        self.degraded
    }

    /// No chunk waits to be issued.
    pub(crate) fn drained(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty() && self.backlog.is_empty()
    }

    /// The size of the next write waiting for the byte budget.
    pub(crate) fn backlog_front(&self) -> Option<usize> {
        self.backlog.front().map(|chunk| chunk.len)
    }

    /// Let the next write through the budget.
    pub(crate) fn release_write(&mut self) {
        if let Some(chunk) = self.backlog.pop_front() {
            self.writes.push_back(chunk);
        }
    }

    pub(crate) fn pressure(&self, now: Instant) -> Pressure {
        let bytes = |queue: &VecDeque<Chunk>| queue.iter().map(|chunk| chunk.len as u64).sum();
        Pressure {
            queued_read_bytes: bytes(&self.reads),
            queued_write_bytes: bytes(&self.writes) + bytes(&self.backlog),
            inflight_bytes: self.inflight_bytes,
            jobs: self.jobs.len(),
            oldest_job: self
                .jobs
                .values()
                .next()
                .map_or(Duration::ZERO, |entry| now - entry.submitted_at),
        }
    }

    pub(crate) fn stats(&self) -> Stats {
        self.stats
    }

    pub(crate) fn effective_write_depth(&self, limits: Limits, reads_pending: bool) -> usize {
        self.aimd.effective_write_depth(limits, reads_pending)
    }

    fn drop_queued(&mut self, job: u64) {
        for queue in [&mut self.reads, &mut self.writes, &mut self.backlog] {
            queue.retain(|chunk| chunk.job != job);
        }
    }

    fn end_if_retired(&mut self, job: u64) -> Option<(Done, JobReport)> {
        let entry = self.jobs.get(&job)?;
        if entry.queued > 0 || entry.inflight > 0 {
            return None;
        }
        let mut entry = self.jobs.remove(&job).expect("the job just found");
        let outcome = entry.outcome();
        match &outcome {
            Ok(()) => match entry.direction {
                Direction::Read => {
                    self.stats.read_jobs += 1;
                    self.stats.read_bytes += entry.bytes;
                }
                Direction::Write => {
                    self.stats.write_jobs += 1;
                    self.stats.write_bytes += entry.bytes;
                }
            },
            Err(IoError::Cancelled) => self.stats.cancelled_jobs += 1,
            Err(_) => self.stats.failed_jobs += 1,
        }
        let report = JobReport {
            outcome,
            queued: entry
                .issued_at
                .map(|issued| issued.duration_since(entry.submitted_at)),
            elapsed: entry.submitted_at.elapsed(),
        };
        Some((entry.done.take().expect("a job reports once"), report))
    }
}

/// What the executor, the pacer and the backend threads share.
pub(crate) struct Shared {
    pub(crate) queues: Mutex<Queues>,
    /// Signalled whenever work arrives or a request retires.
    pub(crate) work: Condvar,
    pub(crate) limits: Limits,
    /// The io_uring thread's wake-up: it waits in the kernel, not on `work`.
    pub(crate) eventfd: Option<OwnedFd>,
    /// In-memory CRC32C table for end-to-end chunk integrity verification.
    pub(crate) checksums: ChecksumTable,
    /// Whether the active backend is staging O_DIRECT through an anonymous mmap bounce pool.
    pub(crate) bounce_active: AtomicBool,
}

impl Shared {
    pub(crate) fn new(limits: Limits, eventfd: Option<OwnedFd>) -> Self {
        Self {
            queues: Mutex::new(Queues::default()),
            work: Condvar::new(),
            limits,
            eventfd,
            checksums: ChecksumTable::new(),
            bounce_active: AtomicBool::new(false),
        }
    }

    pub(crate) fn set_bounce_active(&self) {
        self.bounce_active.store(true, Ordering::Release);
    }

    pub(crate) fn is_bounce_active(&self) -> bool {
        self.bounce_active.load(Ordering::Acquire)
    }

    /// Tell every thread the queues changed.
    pub(crate) fn wake(&self) {
        self.work.notify_all();
        if let Some(eventfd) = &self.eventfd {
            let one = 1u64;
            // SAFETY: writes eight bytes from a local into an eventfd this
            // value owns. A failure only means the counter is saturated, so
            // a wake-up is pending anyway.
            unsafe {
                libc::write(
                    eventfd.as_raw_fd(),
                    (&one as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                );
            }
        }
    }
}

/// Bytes per second, with a burst of at least one chunk.
struct TokenBucket {
    rate: f64,
    burst: f64,
    tokens: f64,
    last: Instant,
}

impl TokenBucket {
    fn new(rate: u64, max_chunk: usize, now: Instant) -> Self {
        // A twentieth of a second of budget smooths the stream without
        // letting a backlog burst past the rate.
        let burst = (rate as f64 / 20.0).max(max_chunk as f64);
        Self {
            rate: rate as f64,
            burst,
            tokens: burst,
            last: now,
        }
    }

    /// Spend `bytes`, or say how long until they are available.
    fn take(&mut self, bytes: usize, now: Instant) -> Result<(), Duration> {
        let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.rate).min(self.burst);
        self.last = now;
        let bytes = bytes as f64;
        if self.tokens >= bytes {
            self.tokens -= bytes;
            return Ok(());
        }
        Err(Duration::from_secs_f64((bytes - self.tokens) / self.rate))
    }
}

/// The pacer thread: moves writes from the backlog to the issue queue at
/// `rate` bytes per second until the queues close.
pub(crate) fn pace(shared: &Shared, rate: u64, max_chunk: usize) {
    let mut bucket = TokenBucket::new(rate, max_chunk, Instant::now());
    let mut queues = shared.queues.lock();
    loop {
        if queues.closing() {
            return;
        }
        let Some(bytes) = queues.backlog_front() else {
            shared.work.wait(&mut queues);
            continue;
        };
        match bucket.take(bytes, Instant::now()) {
            Ok(()) => {
                queues.release_write();
                drop(queues);
                shared.wake();
                queues = shared.queues.lock();
            }
            Err(wait) => {
                shared.work.wait_for(&mut queues, wait);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(direction: Direction, len: usize) -> Chunk {
        Chunk {
            job: 0,
            direction,
            file_offset: 0,
            address: 0,
            len,
            buffer: 0,
            io_index: 0,
            chunk_index: 0,
            issued_at: None,
        }
    }

    fn admit_chunks(
        queues: &mut Queues,
        direction: Direction,
        chunks: impl IntoIterator<Item = Chunk>,
        done: Done,
        now: Instant,
    ) -> u64 {
        queues.admit(direction, Vec::new(), chunks, done, now)
    }

    type Outcomes = std::sync::Arc<Mutex<Vec<Result<(), IoError>>>>;

    fn recorder() -> (Done, Outcomes) {
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        (
            Box::new(move |report: JobReport| sink.lock().push(report.outcome)),
            seen,
        )
    }

    const LIMITS: Limits = Limits {
        depth: 3,
        write_depth: 1,
    };

    #[test]
    fn reads_go_first_and_writes_keep_to_their_share() {
        let mut queues = Queues::default();
        let now = Instant::now();
        let (done, _) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Write,
            [chunk(Direction::Write, 1), chunk(Direction::Write, 1)],
            done,
            now,
        );
        queues.release_write();
        queues.release_write();
        let (done, _) = recorder();
        admit_chunks(&mut queues, Direction::Read, [chunk(Direction::Read, 1)], done, now);
        assert_eq!(queues.next(LIMITS).unwrap().direction, Direction::Read);
        assert_eq!(queues.next(LIMITS).unwrap().direction, Direction::Write);
        // One write in flight is the share, though a slot is free.
        assert_eq!(queues.next(LIMITS), None);
    }

    #[test]
    fn a_job_reports_once_after_its_last_chunk_and_keeps_the_first_error() {
        let mut queues = Queues::default();
        let (done, seen) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Read,
            [
                chunk(Direction::Read, 1),
                chunk(Direction::Read, 1),
                chunk(Direction::Read, 1),
            ],
            done,
            Instant::now(),
        );
        let first = queues.next(LIMITS).unwrap();
        let second = queues.next(LIMITS).unwrap();
        assert!(queues.finish(first, Err(IoError::Os(libc::EIO))).is_empty());
        // The third chunk is no longer issued: the job failed.
        assert_eq!(queues.next(LIMITS), None);
        let mut finished = queues.finish(second, Err(IoError::Os(libc::ENOSPC)));
        assert_eq!(finished.len(), 1);
        let (done, report) = finished.pop().unwrap();
        done(report);
        assert_eq!(*seen.lock(), vec![Err(IoError::Os(libc::EIO))]);
        assert_eq!(queues.stats().failed_jobs, 1);
    }

    #[test]
    fn a_cancelled_job_waits_for_its_chunks_in_flight() {
        let mut queues = Queues::default();
        let (done, _) = recorder();
        let job = admit_chunks(
            &mut queues,
            Direction::Read,
            [chunk(Direction::Read, 1), chunk(Direction::Read, 1)],
            done,
            Instant::now(),
        );
        let issued = queues.next(LIMITS).unwrap();
        assert!(queues.cancel(job).is_none());
        assert!(queues.drained());
        let mut finished = queues.finish(issued, Ok(()));
        assert_eq!(finished.len(), 1);
        let (_, report) = finished.pop().unwrap();
        assert_eq!(report.outcome, Err(IoError::Cancelled));
        assert!(report.queued.is_some_and(|queued| queued <= report.elapsed));
    }

    #[test]
    fn closing_cancels_what_is_queued() {
        let mut queues = Queues::default();
        let (done, _) = recorder();
        admit_chunks(
            &mut queues,
            Direction::Write,
            [chunk(Direction::Write, 1)],
            done,
            Instant::now(),
        );
        let finished = queues.close();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].1.outcome, Err(IoError::Cancelled));
        // Nothing of it reached the kernel.
        assert_eq!(finished[0].1.queued, None);
        assert!(queues.closing() && queues.drained());
    }

    #[test]
    fn the_bucket_paces_bytes_to_its_rate() {
        let start = Instant::now();
        let mut bucket = TokenBucket::new(1000, 100, start);
        // The burst is one chunk: the first passes, the next waits 0.1 s.
        assert_eq!(bucket.take(100, start), Ok(()));
        let wait = bucket.take(100, start).unwrap_err();
        assert!((wait.as_secs_f64() - 0.1).abs() < 1e-9);
        assert_eq!(bucket.take(100, start + Duration::from_millis(100)), Ok(()));
    }

    #[test]
    fn aimd_throttles_writes_on_high_read_p99_and_recovers_on_low_p99() {
        let limits = Limits {
            depth: 8,
            write_depth: 4,
        };
        let mut aimd = ReadLatencyAimd::default();
        let start = Instant::now();
        assert_eq!(aimd.effective_write_depth(limits, true), 4);
        assert_eq!(aimd.effective_write_depth(limits, false), 4);

        // Feed 8 slow reads (12 ms > AIMD_HIGH_US=8_000) spanning 100 ms -> halve 4 -> 2.
        for i in 0..AIMD_MIN_SAMPLES {
            let t = start + Duration::from_millis((i as u64 + 1) * 15);
            aimd.record_read(Duration::from_micros(12_000), t, limits.write_depth);
        }
        assert_eq!(aimd.effective_write_depth(limits, true), 2);
        assert_eq!(aimd.effective_write_depth(limits, false), 4);

        // Another slow window (10 ms > 8_000) -> halve 2 -> 1.
        let t2 = start + Duration::from_millis(200);
        for i in 0..AIMD_MIN_SAMPLES {
            let t = t2 + Duration::from_millis((i as u64 + 1) * 15);
            aimd.record_read(Duration::from_micros(10_000), t, limits.write_depth);
        }
        assert_eq!(aimd.effective_write_depth(limits, true), 1);

        // Another slow window -> halve 1 -> 0 while reads are pending, while
        // returning full `max_write_depth` (4) when reads are completely idle!
        let t3 = start + Duration::from_millis(400);
        for i in 0..AIMD_MIN_SAMPLES {
            let t = t3 + Duration::from_millis((i as u64 + 1) * 15);
            aimd.record_read(Duration::from_micros(10_000), t, limits.write_depth);
        }
        assert_eq!(aimd.effective_write_depth(limits, true), 0);
        assert_eq!(aimd.effective_write_depth(limits, false), 4);

        // Hysteresis window (6 ms within [4_000, 8_000] us) holds steady.
        let t4 = start + Duration::from_millis(600);
        for i in 0..AIMD_MIN_SAMPLES {
            let t = t4 + Duration::from_millis((i as u64 + 1) * 15);
            aimd.record_read(Duration::from_micros(6_000), t, limits.write_depth);
        }
        assert_eq!(aimd.effective_write_depth(limits, true), 0);
        assert_eq!(aimd.effective_write_depth(limits, false), 4);

        // Healthy 4 MiB read window (1.5 ms < AIMD_LOW_US=4_000) -> additive recovery 0 -> 1.
        let t5 = start + Duration::from_millis(800);
        for i in 0..AIMD_MIN_SAMPLES {
            let t = t5 + Duration::from_millis((i as u64 + 1) * 15);
            aimd.record_read(Duration::from_micros(1_500), t, limits.write_depth);
        }
        assert_eq!(aimd.effective_write_depth(limits, true), 1);
    }

    #[test]
    fn circuit_breaker_trips_after_consecutive_os_errors_and_fails_queued_jobs() {
        let mut queues = Queues::default();
        let now = Instant::now();
        // Queue one extra job that will remain unissued when the breaker trips.
        let (done_queued, seen_queued) = recorder();
        admit_chunks(&mut queues, Direction::Write, [chunk(Direction::Write, 1)], done_queued, now);
        queues.release_write();

        // Fail MAX_CONSECUTIVE_DISK_ERRORS reads consecutively.
        let mut last_finished = Vec::new();
        for _ in 0..MAX_CONSECUTIVE_DISK_ERRORS {
            let (done, _) = recorder();
            admit_chunks(&mut queues, Direction::Read, [chunk(Direction::Read, 1)], done, now);
            let issued = queues.next(LIMITS).unwrap();
            last_finished = queues.finish(issued, Err(IoError::Os(libc::EIO)));
        }
        assert!(queues.degraded());
        assert!(queues.drained());
        // Both the triggering read and the queued write were reported.
        assert_eq!(last_finished.len(), 2);
        report(last_finished);
        assert_eq!(*seen_queued.lock(), vec![Err(IoError::Os(libc::EIO))]);
        // No more work is dispatched once degraded.
        let (done_after, _) = recorder();
        admit_chunks(&mut queues, Direction::Read, [chunk(Direction::Read, 1)], done_after, now);
        assert_eq!(queues.next(LIMITS), None);
    }

    #[test]
    fn successful_io_resets_consecutive_os_error_counter() {
        let mut queues = Queues::default();
        let now = Instant::now();
        for _ in 0..(MAX_CONSECUTIVE_DISK_ERRORS - 1) {
            let (done, _) = recorder();
            admit_chunks(&mut queues, Direction::Read, [chunk(Direction::Read, 1)], done, now);
            let issued = queues.next(LIMITS).unwrap();
            let _ = queues.finish(issued, Err(IoError::Os(libc::EIO)));
        }
        assert!(!queues.degraded());
        // One healthy completion resets the streak.
        let (done, _) = recorder();
        admit_chunks(&mut queues, Direction::Read, [chunk(Direction::Read, 1)], done, now);
        let issued = queues.next(LIMITS).unwrap();
        let _ = queues.finish(issued, Ok(()));
        // Another error does not trip the breaker.
        let (done, _) = recorder();
        admit_chunks(&mut queues, Direction::Read, [chunk(Direction::Read, 1)], done, now);
        let issued = queues.next(LIMITS).unwrap();
        let _ = queues.finish(issued, Err(IoError::Os(libc::EIO)));
        assert!(!queues.degraded());
    }
}
