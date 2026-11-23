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

//! The io_uring backend: one thread owns a ring with the memory registered
//! as fixed buffers, keeps up to `depth` requests in flight, and waits in
//! the kernel for the next completion. A read of the executor's eventfd
//! stays armed in the ring, so new work wakes the thread like a completion
//! does. The file is not registered and there is no SQPOLL: the kernels and
//! container policies in use refuse fixed files (EBADF) and unprivileged
//! SQPOLL.

use std::{
    collections::VecDeque,
    io,
    os::fd::{AsRawFd, RawFd},
};

use io_uring::{opcode, squeue, types, IoUring};

use crate::{
    checksum::crc32c,
    dispatch::{self, Chunk, Shared},
    executor::{Direction, IoError, Pieces},
    file::{DiskFile, ALIGNMENT},
    memory::Memory,
};

/// Maximum single-iovec length registered with `io_uring_register_buffers`.
const MAX_FIXED_BOUNCE_BYTES: usize = 1 << 30;

/// The user data of the eventfd read; requests use their slot index.
const WAKE_UP: u64 = u64::MAX;

/// Upper bound on transient `EINTR`/`EAGAIN` retries per chunk so a broken
/// device cannot livelock `Executor::close`.
const MAX_TRANSIENT_RETRIES: u8 = 16;

/// Linux `ioprio` encoding (`class << 13 | level`): foreground reads run at
/// best-effort level 0, background cleaning writes at idle class.
const IOPRIO_READ: u16 = 2 << 13;
const IOPRIO_WRITE: u16 = 3 << 13;

/// One in-flight request in a ring slot, tracking progress across short I/Os.
#[derive(Clone, Copy)]
struct InFlight {
    chunk: Chunk,
    done: usize,
    retries: u8,
    write_crc: Option<u32>,
    bounced: bool,
}

enum BufferMode {
    /// Caller's registered memory pieces are pinned directly in io_uring fixed buffers.
    FixedCaller,
    /// Caller's memory is usable by O_DIRECT directly, without fixed-buffer registration.
    NonFixedCaller,
    /// Caller's memory is VM_PFNMAP/VM_IO (e.g. Ascend `torch_npu` `/dev/davinci_manager`)
    /// which Linux `O_DIRECT` rejects with `EFAULT`. Stage each slot through a page-aligned
    /// anonymous `mmap` bounce pool (registered as a single fixed buffer when possible).
    Bounce {
        pool: Memory,
        slot_size: usize,
        fixed: bool,
    },
}

impl BufferMode {
    fn bounce_slot_addr(&self, slot: usize) -> Option<usize> {
        match self {
            Self::Bounce {
                pool, slot_size, ..
            } => Some(pool.address() + slot * slot_size),
            _ => None,
        }
    }

    fn is_bounce(&self) -> bool {
        matches!(self, Self::Bounce { .. })
    }
}

pub(crate) struct Ring {
    ring: IoUring,
    fd: RawFd,
    depth: usize,
    max_io_bytes: usize,
    mode: BufferMode,
    r_disabled: bool,
    #[cfg(test)]
    async_from_slot: Option<usize>,
    #[cfg(test)]
    on_efault_retry: Option<Box<dyn FnMut(Chunk) + Send>>,
}

fn allocate_bounce_mode(ring: &IoUring, depth: usize, max_io_bytes: usize) -> io::Result<BufferMode> {
    let total = depth
        .checked_mul(max_io_bytes)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    let pool = Memory::map(total)?;
    let fixed = if total <= MAX_FIXED_BOUNCE_BYTES {
        let iov = libc::iovec {
            iov_base: pool.address() as *mut libc::c_void,
            iov_len: pool.size(),
        };
        // SAFETY: `pool` stays mapped until after `ring` is dropped (or both are forgotten on breakdown).
        match unsafe { ring.submitter().register_buffers(&[iov]) } {
            Ok(()) => true,
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(libc::ENOMEM | libc::EFAULT | libc::EINVAL | libc::E2BIG | libc::EPERM)
                ) =>
            {
                tracing::warn!(
                    %error,
                    bounce_bytes = total,
                    "io_uring register_buffers failed on bounce pool; using non-fixed O_DIRECT on bounce pool"
                );
                false
            }
            Err(error) => return Err(error),
        }
    } else {
        false
    };
    Ok(BufferMode::Bounce {
        pool,
        slot_size: max_io_bytes,
        fixed,
    })
}

fn build_ring(entries: u32) -> io::Result<(IoUring, bool)> {
    // Linux >= 6.1: bind single-issuer ownership when `Ring::run` activates on
    // its dedicated worker thread, avoiding cross-thread kernel ring locks.
    match IoUring::builder()
        .setup_single_issuer()
        .setup_defer_taskrun()
        .setup_r_disabled()
        .build(entries)
    {
        Ok(ring) => return Ok((ring, true)),
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {}
        Err(error) => return Err(error),
    }
    // Linux 5.19..6.0 fallback: cooperative task_work without IPI interrupts.
    match IoUring::builder().setup_coop_taskrun().build(entries) {
        Ok(ring) => return Ok((ring, false)),
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {}
        Err(error) => return Err(error),
    }
    Ok((IoUring::new(entries)?, false))
}

impl Ring {
    #[cfg(test)]
    pub(crate) fn new(
        file: &DiskFile,
        pieces: &Pieces,
        depth: usize,
        max_io_bytes: usize,
    ) -> io::Result<Self> {
        Self::new_with_mode(file, pieces, depth, max_io_bytes, false)
    }

    pub(crate) fn new_with_mode(
        file: &DiskFile,
        pieces: &Pieces,
        depth: usize,
        max_io_bytes: usize,
        force_bounce: bool,
    ) -> io::Result<Self> {
        // One entry beyond the requests for the armed wake-up read.
        let entries = u32::try_from((depth + 1).next_power_of_two())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let (ring, r_disabled) = build_ring(entries)?;
        let mode = if force_bounce {
            allocate_bounce_mode(&ring, depth, max_io_bytes)?
        } else {
            let buffers: Vec<libc::iovec> = pieces
                .iter()
                .map(|(start, end)| libc::iovec {
                    iov_base: start as *mut libc::c_void,
                    iov_len: end - start,
                })
                .collect();
            // SAFETY: the executor's caller keeps the memory mapped until close
            // succeeds (`Executor::start`), and a failed close leaks this ring.
            // - `EFAULT` signals VM_PFNMAP/VM_IO device-pinned memory (e.g. Ascend
            //   `torch_npu` `/dev/davinci_manager`), which Linux `O_DIRECT` also
            //   rejects on non-fixed Read/Write -> stage through an anonymous mmap
            //   bounce pool (registered as fixed buffers when possible).
            // - `ENOMEM`/`EINVAL`/`E2BIG`/`EPERM` fall back to non-fixed O_DIRECT
            //   on the caller's memory (with lazy promotion to a bounce pool if a
            //   runtime CQE still returns `-EFAULT`).
            match unsafe { ring.submitter().register_buffers(&buffers) } {
                Ok(()) => BufferMode::FixedCaller,
                Err(error) if error.raw_os_error() == Some(libc::EFAULT) => {
                    let bounce = allocate_bounce_mode(&ring, depth, max_io_bytes)?;
                    let bounce_fixed = matches!(bounce, BufferMode::Bounce { fixed: true, .. });
                    tracing::warn!(
                        %error,
                        pieces = buffers.len(),
                        bounce_bytes = depth * max_io_bytes,
                        bounce_fixed,
                        "io_uring register_buffers failed with EFAULT (device-pinned VM_PFNMAP/VM_IO memory); \
                         staging O_DIRECT through anonymous mmap bounce pool"
                    );
                    bounce
                }
                Err(error)
                    if matches!(
                        error.raw_os_error(),
                        Some(libc::ENOMEM | libc::EINVAL | libc::E2BIG | libc::EPERM)
                    ) =>
                {
                    tracing::warn!(
                        %error,
                        pieces = buffers.len(),
                        "io_uring register_buffers failed (check RLIMIT_MEMLOCK / THP); \
                         falling back to non-fixed O_DIRECT ops"
                    );
                    BufferMode::NonFixedCaller
                }
                Err(error) => return Err(error),
            }
        };
        Ok(Self {
            ring,
            fd: file.raw_fd(),
            depth,
            max_io_bytes,
            mode,
            r_disabled,
            #[cfg(test)]
            async_from_slot: None,
            #[cfg(test)]
            on_efault_retry: None,
        })
    }

    pub(crate) fn is_bounce(&self) -> bool {
        self.mode.is_bounce()
    }

    #[cfg(test)]
    fn new_non_fixed(file: &DiskFile, depth: usize, max_io_bytes: usize) -> io::Result<Self> {
        let entries = u32::try_from((depth + 1).next_power_of_two())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let (ring, r_disabled) = build_ring(entries)?;
        Ok(Self {
            ring,
            fd: file.raw_fd(),
            depth,
            max_io_bytes,
            mode: BufferMode::NonFixedCaller,
            r_disabled,
            async_from_slot: None,
            on_efault_retry: None,
        })
    }

    #[cfg(test)]
    fn new_bounce(file: &DiskFile, depth: usize, max_io_bytes: usize) -> io::Result<Self> {
        let entries = u32::try_from((depth + 1).next_power_of_two())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let (ring, r_disabled) = build_ring(entries)?;
        let mode = allocate_bounce_mode(&ring, depth, max_io_bytes)?;
        Ok(Self {
            ring,
            fd: file.raw_fd(),
            depth,
            max_io_bytes,
            mode,
            r_disabled,
            async_from_slot: None,
            on_efault_retry: None,
        })
    }

    pub(crate) fn run(mut self, shared: &Shared) {
        if self.r_disabled {
            if let Err(error) = self.ring.submitter().register_enable_rings() {
                tracing::error!(%error, "io_uring register_enable_rings failed");
                let finished = shared.queues.lock().break_down();
                shared.wake();
                dispatch::report(finished);
                return;
            }
        }
        let eventfd = shared
            .eventfd
            .as_ref()
            .expect("the io_uring backend wakes through an eventfd")
            .as_raw_fd();
        // The wake-up read's target. Dropped after the ring, whose teardown
        // cancels the read.
        let mut counter = Box::new(0u64);
        let mut slots: Vec<Option<InFlight>> = vec![None; self.depth];
        let mut free: Vec<usize> = (0..self.depth).rev().collect();
        let mut retries: VecDeque<InFlight> = VecDeque::new();
        let mut armed = false;
        let mut to_issue: Vec<InFlight> = Vec::with_capacity(self.depth);
        let mut outcomes: Vec<(Chunk, Result<(), IoError>, Option<u32>)> =
            Vec::with_capacity(self.depth);
        loop {
            let sq_space = {
                let submission = self.ring.submission();
                let avail = submission.capacity().saturating_sub(submission.len());
                if armed {
                    avail
                } else {
                    avail.saturating_sub(1)
                }
            };
            let max_issue = free.len().min(sq_space);
            to_issue.clear();
            {
                let mut queues = shared.queues.lock();
                while to_issue.len() < max_issue {
                    let inflight = match retries.pop_front() {
                        Some(inflight) => inflight,
                        None => match queues.next(shared.limits) {
                            Some(chunk) => InFlight {
                                chunk,
                                done: 0,
                                retries: 0,
                                write_crc: None,
                                bounced: false,
                            },
                            None => break,
                        },
                    };
                    to_issue.push(inflight);
                }
                if queues.closing()
                    && queues.drained()
                    && retries.is_empty()
                    && to_issue.is_empty()
                    && free.len() == self.depth
                {
                    break;
                }
            }
            // Compute write CRC32C outside `shared.queues` to avoid holding
            // the dispatch mutex during hashing. In bounce mode we hash the
            // cache-hot bounce buffer right after copying into the slot below.
            if shared.checksums.enabled() && !self.mode.is_bounce() {
                for inflight in &mut to_issue {
                    if inflight.chunk.direction == Direction::Write
                        && inflight.done == 0
                        && inflight.retries == 0
                        && inflight.write_crc.is_none()
                    {
                        // SAFETY: the chunk lies in registered memory that stays
                        // mapped until the executor closes.
                        let slice = unsafe {
                            std::slice::from_raw_parts(
                                inflight.chunk.address as *const u8,
                                inflight.chunk.len,
                            )
                        };
                        inflight.write_crc = Some(crc32c(slice));
                    }
                }
            }
            {
                let mut submission = self.ring.submission();
                if !armed && !submission.is_full() {
                    let read = opcode::Read::new(
                        types::Fd(eventfd),
                        (&mut *counter as *mut u64).cast(),
                        8,
                    )
                    .build()
                    .user_data(WAKE_UP);
                    // SAFETY: `counter` outlives the ring.
                    unsafe { submission.push(&read) }.expect("the ring has room for the wake-up");
                    armed = true;
                }
                for mut inflight in to_issue.drain(..) {
                    let slot = free.pop().expect("free slot reserved for to_issue");
                    inflight.bounced = self.mode.is_bounce();
                    if let Some(bounce_addr) = self.mode.bounce_slot_addr(slot) {
                        let remaining = inflight.chunk.len - inflight.done;
                        debug_assert!(remaining <= self.max_io_bytes);
                        if inflight.chunk.direction == Direction::Write {
                            // SAFETY: source is caller's registered memory (stays mapped
                            // until close) and destination is this slot's owned bounce buffer.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    (inflight.chunk.address + inflight.done) as *const u8,
                                    bounce_addr as *mut u8,
                                    remaining,
                                );
                            }
                            if shared.checksums.enabled()
                                && inflight.done == 0
                                && inflight.retries == 0
                                && inflight.write_crc.is_none()
                            {
                                let slice = unsafe {
                                    std::slice::from_raw_parts(
                                        bounce_addr as *const u8,
                                        inflight.chunk.len,
                                    )
                                };
                                inflight.write_crc = Some(crc32c(slice));
                            }
                        }
                    }
                    let req =
                        request(self.fd, inflight, slot, &self.mode).user_data(slot as u64);
                    #[cfg(test)]
                    let req =
                        if !inflight.bounced && self.async_from_slot.is_some_and(|min| slot >= min)
                        {
                            req.flags(squeue::Flags::ASYNC)
                        } else {
                            req
                        };
                    // SAFETY: the chunk (or bounce slot) stays mapped; its slot
                    // keeps it until it completes.
                    unsafe { submission.push(&req) }
                        .expect("the ring has an entry for every slot");
                    slots[slot] = Some(inflight);
                }
            }
            if let Err(error) = self.ring.submit_and_wait(1) {
                // Interrupted, or the completion queue is full: reap first.
                if !matches!(error.raw_os_error(), Some(libc::EINTR | libc::EBUSY)) {
                    tracing::error!(%error, "io_uring broke down; requests it holds never report");
                    let finished = shared.queues.lock().break_down();
                    shared.wake();
                    dispatch::report(finished);
                    // The kernel may still write into the memory, bounce pool,
                    // and counter: keep them, and the ring, for the process.
                    std::mem::forget(self.ring);
                    std::mem::forget(self.mode);
                    std::mem::forget(counter);
                    return;
                }
            }
            let completions: Vec<(u64, i32)> = self
                .ring
                .completion()
                .map(|completion| (completion.user_data(), completion.result()))
                .collect();
            outcomes.clear();
            let mut promote_to_bounce = false;
            let mut efault_retries: Vec<InFlight> = Vec::new();
            for (user_data, result) in completions {
                if user_data == WAKE_UP {
                    armed = false;
                    continue;
                }
                let slot = usize::try_from(user_data).expect("a slot index");
                let mut inflight = slots[slot]
                    .take()
                    .expect("a completion of an issued request");
                let bounce_addr = if inflight.bounced {
                    self.mode.bounce_slot_addr(slot)
                } else {
                    None
                };
                // Catch every pre-promotion NonFixedCaller SQE that hit -EFAULT
                // (e.g. ENOMEM hid a VM_PFNMAP region at setup), even if an
                // earlier CQE already promoted `self.mode` to `BufferMode::Bounce`.
                if result == -libc::EFAULT && !inflight.bounced && inflight.done == 0 {
                    free.push(slot);
                    #[cfg(test)]
                    if let Some(hook) = self.on_efault_retry.as_mut() {
                        hook(inflight.chunk);
                    }
                    if self.mode.is_bounce() {
                        retries.push_back(inflight);
                    } else {
                        promote_to_bounce = true;
                        efault_retries.push(inflight);
                    }
                    continue;
                }
                if result == -libc::EINTR || result == -libc::EAGAIN {
                    free.push(slot);
                    if inflight.retries < MAX_TRANSIENT_RETRIES {
                        inflight.retries += 1;
                        retries.push_back(inflight);
                    } else {
                        outcomes.push((inflight.chunk, Err(IoError::Os(-result)), None));
                    }
                    continue;
                }
                let remaining = inflight.chunk.len - inflight.done;
                let mut chunk_crc = None;
                let outcome = match usize::try_from(result) {
                    Err(_) => Err(IoError::Os(-result)),
                    Ok(0) => Err(IoError::Short {
                        expected: inflight.chunk.len,
                        done: inflight.done,
                    }),
                    Ok(n) if n < remaining => {
                        if let Some(b_addr) = bounce_addr {
                            if inflight.chunk.direction == Direction::Read {
                                // SAFETY: kernel wrote `n` bytes into slot's bounce buffer;
                                // copy into caller's registered memory before freeing slot.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        b_addr as *const u8,
                                        (inflight.chunk.address + inflight.done) as *mut u8,
                                        n,
                                    );
                                }
                            }
                        }
                        let next_done = inflight.done + n;
                        if n % ALIGNMENT != 0 {
                            // O_DIRECT cannot resume from an unaligned offset.
                            Err(IoError::Short {
                                expected: inflight.chunk.len,
                                done: next_done,
                            })
                        } else {
                            free.push(slot);
                            inflight.done = next_done;
                            inflight.retries = 0;
                            retries.push_back(inflight);
                            continue;
                        }
                    }
                    Ok(n) if n == remaining => {
                        if inflight.chunk.direction == Direction::Read {
                            if let Some(b_addr) = bounce_addr {
                                // SAFETY: kernel wrote `n` bytes into slot's bounce buffer;
                                // hash cache-hot bounce buffer when full chunk landed in one
                                // op, then copy into caller's registered memory.
                                if shared.checksums.enabled() && inflight.done == 0 {
                                    let slice = unsafe {
                                        std::slice::from_raw_parts(
                                            b_addr as *const u8,
                                            inflight.chunk.len,
                                        )
                                    };
                                    chunk_crc = Some(crc32c(slice));
                                }
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        b_addr as *const u8,
                                        (inflight.chunk.address + inflight.done) as *mut u8,
                                        n,
                                    );
                                }
                                if shared.checksums.enabled() && inflight.done > 0 {
                                    let slice = unsafe {
                                        std::slice::from_raw_parts(
                                            inflight.chunk.address as *const u8,
                                            inflight.chunk.len,
                                        )
                                    };
                                    chunk_crc = Some(crc32c(slice));
                                }
                            } else if shared.checksums.enabled() {
                                // SAFETY: the read just completed into registered
                                // memory that stays mapped until close.
                                let slice = unsafe {
                                    std::slice::from_raw_parts(
                                        inflight.chunk.address as *const u8,
                                        inflight.chunk.len,
                                    )
                                };
                                chunk_crc = Some(crc32c(slice));
                            }
                        } else {
                            chunk_crc = inflight.write_crc;
                        }
                        Ok(())
                    }
                    Ok(_) => Err(IoError::Os(libc::EIO)),
                };
                free.push(slot);
                outcomes.push((inflight.chunk, outcome, chunk_crc));
            }
            if promote_to_bounce && matches!(self.mode, BufferMode::NonFixedCaller) {
                match allocate_bounce_mode(&self.ring, self.depth, self.max_io_bytes) {
                    Ok(bounce_mode) => {
                        let bounce_fixed =
                            matches!(bounce_mode, BufferMode::Bounce { fixed: true, .. });
                        tracing::warn!(
                            bounce_bytes = self.depth * self.max_io_bytes,
                            bounce_fixed,
                            "io_uring non-fixed O_DIRECT hit EFAULT at runtime; switching to anonymous mmap bounce pool"
                        );
                        self.mode = bounce_mode;
                        shared.set_bounce_active();
                        retries.extend(efault_retries);
                    }
                    Err(error) => {
                        let errno = error.raw_os_error().unwrap_or(libc::ENOMEM);
                        tracing::error!(%error, "failed to allocate io_uring runtime bounce pool");
                        for pending in efault_retries {
                            outcomes.push((pending.chunk, Err(IoError::Os(errno)), None));
                        }
                    }
                }
            }
            let mut finished = Vec::new();
            if !outcomes.is_empty() {
                let mut queues = shared.queues.lock();
                for (chunk, outcome, chunk_crc) in outcomes.drain(..) {
                    finished.extend(queues.finish_with_checksum(
                        chunk,
                        outcome,
                        chunk_crc,
                        &shared.checksums,
                    ));
                }
            }
            shared.work.notify_all();
            dispatch::report(finished);
        }
        drop(self.ring);
        drop(self.mode);
        drop(counter);
    }
}

fn request(fd: RawFd, inflight: InFlight, slot: usize, mode: &BufferMode) -> squeue::Entry {
    let chunk = inflight.chunk;
    let remaining = chunk.len - inflight.done;
    let len = u32::try_from(remaining).expect("requests are at most 1 GiB");
    let file_offset = chunk.file_offset + inflight.done as u64;
    match mode {
        BufferMode::FixedCaller => {
            let address = chunk.address + inflight.done;
            match chunk.direction {
                Direction::Read => {
                    opcode::ReadFixed::new(types::Fd(fd), address as *mut u8, len, chunk.buffer)
                        .offset(file_offset)
                        .ioprio(IOPRIO_READ)
                        .build()
                }
                Direction::Write => {
                    opcode::WriteFixed::new(types::Fd(fd), address as *const u8, len, chunk.buffer)
                        .offset(file_offset)
                        .ioprio(IOPRIO_WRITE)
                        .build()
                }
            }
        }
        BufferMode::NonFixedCaller => {
            let address = chunk.address + inflight.done;
            match chunk.direction {
                Direction::Read => opcode::Read::new(types::Fd(fd), address as *mut u8, len)
                    .offset(file_offset)
                    .ioprio(IOPRIO_READ)
                    .build(),
                Direction::Write => opcode::Write::new(types::Fd(fd), address as *const u8, len)
                    .offset(file_offset)
                    .ioprio(IOPRIO_WRITE)
                    .build(),
            }
        }
        BufferMode::Bounce {
            pool,
            slot_size,
            fixed,
        } => {
            let address = pool.address() + slot * slot_size;
            match (chunk.direction, *fixed) {
                (Direction::Read, true) => {
                    opcode::ReadFixed::new(types::Fd(fd), address as *mut u8, len, 0)
                        .offset(file_offset)
                        .ioprio(IOPRIO_READ)
                        .build()
                }
                (Direction::Write, true) => {
                    opcode::WriteFixed::new(types::Fd(fd), address as *const u8, len, 0)
                        .offset(file_offset)
                        .ioprio(IOPRIO_WRITE)
                        .build()
                }
                (Direction::Read, false) => opcode::Read::new(types::Fd(fd), address as *mut u8, len)
                    .offset(file_offset)
                    .ioprio(IOPRIO_READ)
                    .build(),
                (Direction::Write, false) => {
                    opcode::Write::new(types::Fd(fd), address as *const u8, len)
                        .offset(file_offset)
                        .ioprio(IOPRIO_WRITE)
                        .build()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        os::fd::{FromRawFd, OwnedFd},
        path::PathBuf,
        sync::{mpsc, Arc},
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::{
        dispatch::{Chunk, IoPlan, Limits, Shared},
        executor::JobReport,
        memory::Memory,
    };

    struct ScratchFile(PathBuf);

    impl ScratchFile {
        fn new(name: &str) -> Self {
            let dir = std::env::var_os("TS_DISK_TEST_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            Self(dir.join(format!("ts-disk-uring-{}-{name}", std::process::id())))
        }
    }

    impl Drop for ScratchFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn register_buffers_efault_falls_back_to_bounce_pool_without_failing() {
        let scratch = ScratchFile::new("efault-fallback");
        let file = DiskFile::create(&scratch.0, 16 * ALIGNMENT as u64).unwrap();
        // A PROT_READ mapping triggers EFAULT (os error 14) on
        // io_uring_register_buffers (FOLL_WRITE | FOLL_LONGTERM), mirroring
        // device-pinned VM_PFNMAP/VM_IO memory (e.g. Ascend torch_npu) that
        // Linux O_DIRECT also rejects with EFAULT.
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
        assert_ne!(ptr, libc::MAP_FAILED);
        let pieces = Pieces::from_raw_for_test(vec![(ptr as usize, ptr as usize + ALIGNMENT)]);
        let ring = Ring::new(&file, &pieces, 4, 2 * ALIGNMENT)
            .expect("EFAULT should fall back cleanly to bounce pool");
        assert!(ring.mode.is_bounce());
        drop(ring);
        unsafe {
            libc::munmap(ptr, ALIGNMENT);
        }
    }

    fn run_round_trip_with_ring(name: &str, make_ring: impl FnOnce(&DiskFile) -> Ring) {
        let scratch = ScratchFile::new(name);
        let file = DiskFile::create(&scratch.0, 16 * ALIGNMENT as u64).unwrap();
        let mut memory = Memory::map(4 * ALIGNMENT).unwrap();
        let payload: Vec<u8> = (0..2 * ALIGNMENT)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(5))
            .collect();
        memory.write_at(0, &payload);

        let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
        assert!(efd >= 0);
        let eventfd = unsafe { OwnedFd::from_raw_fd(efd) };
        let shared = Arc::new(Shared::new(
            Limits {
                depth: 4,
                write_depth: 2,
            },
            Some(eventfd),
        ));
        let ring = make_ring(&file);
        let worker_shared = shared.clone();
        let worker = thread::spawn(move || ring.run(&worker_shared));

        let (tx, rx) = mpsc::channel();
        {
            let mut queues = shared.queues.lock();
            queues.admit(
                Direction::Write,
                vec![IoPlan {
                    file_offset: 0,
                    len: 2 * ALIGNMENT,
                    chunks: 2,
                }],
                [
                    Chunk {
                        job: 0,
                        direction: Direction::Write,
                        file_offset: 0,
                        address: memory.address(),
                        len: ALIGNMENT,
                        buffer: 0,
                        io_index: 0,
                        chunk_index: 0,
                        issued_at: None,
                    },
                    Chunk {
                        job: 0,
                        direction: Direction::Write,
                        file_offset: ALIGNMENT as u64,
                        address: memory.address() + ALIGNMENT,
                        len: ALIGNMENT,
                        buffer: 0,
                        io_index: 0,
                        chunk_index: 1,
                        issued_at: None,
                    },
                ],
                Box::new(move |report: JobReport| tx.send(report.outcome).unwrap()),
                Instant::now(),
            );
            queues.release_write();
            queues.release_write();
        }
        shared.wake();
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Ok(()));

        // Read back into a separate slot as ONE single chunk (different split!) and verify bytes + CRC32C.
        let (tx, rx) = mpsc::channel();
        {
            let mut queues = shared.queues.lock();
            queues.admit(
                Direction::Read,
                vec![IoPlan {
                    file_offset: 0,
                    len: 2 * ALIGNMENT,
                    chunks: 1,
                }],
                [Chunk {
                    job: 0,
                    direction: Direction::Read,
                    file_offset: 0,
                    address: memory.address() + 2 * ALIGNMENT,
                    len: 2 * ALIGNMENT,
                    buffer: 0,
                    io_index: 0,
                    chunk_index: 0,
                    issued_at: None,
                }],
                Box::new(move |report: JobReport| tx.send(report.outcome).unwrap()),
                Instant::now(),
            );
        }
        shared.wake();
        assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), Ok(()));

        let mut actual = vec![0u8; 2 * ALIGNMENT];
        memory.read_at(2 * ALIGNMENT, &mut actual);
        assert_eq!(actual, payload);

        let _ = shared.queues.lock().close();
        shared.wake();
        worker.join().unwrap();
    }

    #[test]
    fn non_fixed_direct_io_round_trips_with_crc32c() {
        run_round_trip_with_ring("non-fixed-round-trip", |file| {
            let ring = Ring::new_non_fixed(file, 4, 2 * ALIGNMENT).unwrap();
            assert!(matches!(ring.mode, BufferMode::NonFixedCaller));
            ring
        });
    }

    #[test]
    fn bounce_pool_direct_io_round_trips_with_crc32c() {
        run_round_trip_with_ring("bounce-round-trip", |file| {
            let ring = Ring::new_bounce(file, 4, 2 * ALIGNMENT).unwrap();
            assert!(ring.mode.is_bounce());
            ring
        });
    }

    #[test]
    fn runtime_efault_promotion_preserves_inflight_non_bounced_reads_and_retries_all_siblings() {
        // Reproduces TS-P0-1 (`RLIMIT_MEMLOCK < 75.5 GiB` -> `NonFixedCaller` startup
        // with mixed anonymous `mmap` `push_ring` + `VM_PFNMAP` Host L2):
        // (a) Non-bounced anonymous `mmap` reads in flight across runtime `Bounce`
        //     promotion MUST NOT be zero-corrupted by the freshly mapped bounce pool.
        // (b) `>= 8` (`MAX_CONSECUTIVE_DISK_ERRORS`) sibling `-EFAULT` SQEs — including
        //     ones harvested *after* an earlier CQE already promoted `self.mode` to
        //     `Bounce` — MUST retry cleanly without tripping `queues.degraded()`.
        const DEPTH: usize = 16;
        const ANON_CHUNKS: usize = 4;
        const ANON_CHUNK_PAGES: usize = 64; // 256 KiB each (1 MiB total real NVMe DMA)
        const EFAULT_CHUNKS: usize = 10; // >= MAX_CONSECUTIVE_DISK_ERRORS (8)
        const MAX_CHUNK_BYTES: usize = ANON_CHUNK_PAGES * ALIGNMENT;

        let total_pages = ANON_CHUNKS * ANON_CHUNK_PAGES + EFAULT_CHUNKS;
        let scratch = ScratchFile::new("runtime-efault-mixed-promotion");
        let file = DiskFile::create(&scratch.0, (total_pages * ALIGNMENT) as u64).unwrap();

        let mut anon_mem = Memory::map(total_pages * ALIGNMENT).unwrap();
        let golden: Vec<u8> = (0..total_pages * ALIGNMENT)
            .map(|i| ((i / 17) as u8).wrapping_mul(37).wrapping_add(0x5B) | 1)
            .collect();
        anon_mem.write_at(0, &golden);

        // Allocate 10 disjoint `PROT_READ` pages simulating `VM_PFNMAP` `-EFAULT` on
        // `O_DIRECT` `Read` (`pin_user_pages_fast(FOLL_WRITE)` -> `-EFAULT`).
        let efault_bytes = EFAULT_CHUNKS * ALIGNMENT;
        let efault_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                efault_bytes,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(efault_ptr, libc::MAP_FAILED);
        let efault_base = efault_ptr as usize;

        let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
        assert!(efd >= 0);
        let eventfd = unsafe { OwnedFd::from_raw_fd(efd) };
        let shared = Arc::new(Shared::new(
            Limits {
                depth: DEPTH,
                write_depth: DEPTH,
            },
            Some(eventfd),
        ));

        let mut ring = Ring::new_non_fixed(&file, DEPTH, MAX_CHUNK_BYTES).unwrap();
        assert!(matches!(ring.mode, BufferMode::NonFixedCaller));
        // Slot 0 fails inline in `io_submit_sqes` and triggers `Bounce` promotion
        // immediately while slots 1..14 (`IOSQE_ASYNC` via `io-wq`) are in flight!
        ring.async_from_slot = Some(1);
        let efault_seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let efault_seen_hook = efault_seen.clone();
        ring.on_efault_retry = Some(Box::new(move |chunk: Chunk| {
            efault_seen_hook.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Make just this chunk's disjoint page writable before its bounced retry.
            let rc = unsafe {
                libc::mprotect(
                    chunk.address as *mut libc::c_void,
                    chunk.len,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            assert_eq!(rc, 0, "mprotect failed on retried EFAULT page");
        }));

        let worker_shared = shared.clone();
        let worker = thread::spawn(move || ring.run(&worker_shared));

        // Step 1: Write golden payload to disk (still in `NonFixedCaller` mode) so
        // `shared.checksums` records valid CRC32Cs for every range.
        let (w_tx, w_rx) = mpsc::channel();
        {
            let mut queues = shared.queues.lock();
            let mut plans = Vec::new();
            let mut chunks = Vec::new();
            let mut io_idx = 0usize;
            for k in 0..EFAULT_CHUNKS {
                let off = k * ALIGNMENT;
                plans.push(IoPlan {
                    file_offset: off as u64,
                    len: ALIGNMENT,
                    chunks: 1,
                });
                chunks.push(Chunk {
                    job: 0,
                    direction: Direction::Write,
                    file_offset: off as u64,
                    address: anon_mem.address() + off,
                    len: ALIGNMENT,
                    buffer: 0,
                    io_index: io_idx,
                    chunk_index: 0,
                    issued_at: None,
                });
                io_idx += 1;
            }
            for a in 0..ANON_CHUNKS {
                let off = EFAULT_CHUNKS * ALIGNMENT + a * MAX_CHUNK_BYTES;
                plans.push(IoPlan {
                    file_offset: off as u64,
                    len: MAX_CHUNK_BYTES,
                    chunks: 1,
                });
                chunks.push(Chunk {
                    job: 0,
                    direction: Direction::Write,
                    file_offset: off as u64,
                    address: anon_mem.address() + off,
                    len: MAX_CHUNK_BYTES,
                    buffer: 0,
                    io_index: io_idx,
                    chunk_index: 0,
                    issued_at: None,
                });
                io_idx += 1;
            }
            let n_writes = chunks.len();
            queues.admit(
                Direction::Write,
                plans,
                chunks,
                Box::new(move |report: JobReport| w_tx.send(report.outcome).unwrap()),
                Instant::now(),
            );
            for _ in 0..n_writes {
                queues.release_write();
            }
        }
        shared.wake();
        assert_eq!(w_rx.recv_timeout(Duration::from_secs(10)).unwrap(), Ok(()));
        assert!(
            !shared.is_bounce_active(),
            "writes from RW anon_mem should keep NonFixedCaller active before mixed read"
        );

        // Step 2: Poison `anon_mem` read destinations with `0xDEADBEEF`, then admit
        // 10 `-EFAULT` reads + 4 `anon_mem` reads in ONE atomic queue batch!
        let anon_read_base = EFAULT_CHUNKS * ALIGNMENT;
        let poison: Vec<u8> = (0..ANON_CHUNKS * MAX_CHUNK_BYTES)
            .map(|i| [0xDE, 0xAD, 0xBE, 0xEF][i % 4])
            .collect();
        anon_mem.write_at(anon_read_base, &poison);

        let (r_efault_tx, r_efault_rx) = mpsc::channel();
        let (r_anon_tx, r_anon_rx) = mpsc::channel();
        {
            let mut queues = shared.queues.lock();
            let mut efault_plans = Vec::new();
            let mut efault_chunks = Vec::new();
            for k in 0..EFAULT_CHUNKS {
                let off = k * ALIGNMENT;
                efault_plans.push(IoPlan {
                    file_offset: off as u64,
                    len: ALIGNMENT,
                    chunks: 1,
                });
                efault_chunks.push(Chunk {
                    job: 0,
                    direction: Direction::Read,
                    file_offset: off as u64,
                    address: efault_base + off,
                    len: ALIGNMENT,
                    buffer: 0,
                    io_index: k,
                    chunk_index: 0,
                    issued_at: None,
                });
            }
            queues.admit(
                Direction::Read,
                efault_plans,
                efault_chunks,
                Box::new(move |report: JobReport| r_efault_tx.send(report.outcome).unwrap()),
                Instant::now(),
            );

            let mut anon_plans = Vec::new();
            let mut anon_chunks = Vec::new();
            for a in 0..ANON_CHUNKS {
                let off = anon_read_base + a * MAX_CHUNK_BYTES;
                anon_plans.push(IoPlan {
                    file_offset: off as u64,
                    len: MAX_CHUNK_BYTES,
                    chunks: 1,
                });
                anon_chunks.push(Chunk {
                    job: 0,
                    direction: Direction::Read,
                    file_offset: off as u64,
                    address: anon_mem.address() + off,
                    len: MAX_CHUNK_BYTES,
                    buffer: 0,
                    io_index: a,
                    chunk_index: 0,
                    issued_at: None,
                });
            }
            queues.admit(
                Direction::Read,
                anon_plans,
                anon_chunks,
                Box::new(move |report: JobReport| r_anon_tx.send(report.outcome).unwrap()),
                Instant::now(),
            );
        }
        shared.wake();

        assert_eq!(
            r_efault_rx.recv_timeout(Duration::from_secs(10)).unwrap(),
            Ok(()),
            "all 10 -EFAULT sibling SQEs must retry cleanly via promoted bounce pool"
        );
        assert_eq!(
            r_anon_rx.recv_timeout(Duration::from_secs(10)).unwrap(),
            Ok(()),
            "in-flight non-bounced anon_mem reads must not be zero-corrupted or fail CRC32C"
        );
        assert!(shared.is_bounce_active());
        assert_eq!(
            efault_seen.load(std::sync::atomic::Ordering::Relaxed),
            EFAULT_CHUNKS
        );
        {
            let queues = shared.queues.lock();
            assert!(!queues.degraded(), "runtime EFAULT promotion must not trip degraded()");
            assert_eq!(queues.stats().failed_jobs, 0);
        }

        // Verify 0-bit match on both `anon_mem` and `efault_mem`!
        let mut actual_anon = vec![0u8; ANON_CHUNKS * MAX_CHUNK_BYTES];
        anon_mem.read_at(anon_read_base, &mut actual_anon);
        assert_eq!(actual_anon, &golden[anon_read_base..]);

        let actual_efault =
            unsafe { std::slice::from_raw_parts(efault_ptr as *const u8, efault_bytes) };
        assert_eq!(actual_efault, &golden[..efault_bytes]);

        let _ = shared.queues.lock().close();
        shared.wake();
        worker.join().unwrap();
        unsafe {
            libc::munmap(efault_ptr, efault_bytes);
        }
    }
}
