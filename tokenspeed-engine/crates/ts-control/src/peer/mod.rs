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

//! Sharing cached prefixes between engine replicas, as the control loop runs
//! it (`docs/rfcs/0004-peer-prefix-sharing.md`).
//!
//! The replica's first member owns the endpoint. As a holder it answers
//! other replicas' requests on a servicer thread, with reads of the scheduler
//! between the control thread's rounds ([`servicer::Holder`]); as a reader it
//! sends its scheduler's lookups and drives each landing's pin and release.
//! Every round the members share what they saw -- the first member's clock,
//! the holder's queued changes and the replies it received, each member's
//! read outcomes -- and step every landing alike from that agreement
//! ([`landing::Landing`]). Each member reads its own shard of the holder,
//! from the holder's member of the same index, once its plan's device work
//! has run.

pub mod geometry;
pub mod landing;
pub mod net;
pub mod push;
pub mod read;
pub mod servicer;
pub mod wire;

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    num::NonZeroUsize,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use parking_lot::Mutex;
use ts_collective::{Group, Mesh};
use ts_scheduler::{
    PeerAnswerKind, PeerEventKind, PeerEventView, PeerFetchView, PeerLookupView, PeerTier,
    Scheduler,
};

use crate::{
    codec::{Reader, Writer},
    peer::{
        geometry::{push_items, read_batches, RankGeometry},
        landing::{Action, HealthOutcome, Landing, ReadOutcome, Timeouts},
        net::{Inbound, PeerNet},
        push::PagePusher,
        read::{PageReader, PushJob, ReadJob, Reads},
        servicer::{empty_event, Holder},
        wire::{read_change, write_change, Prompt, Reply, Request, Stamp},
    },
};

const READ_CONCURRENCY: NonZeroUsize = NonZeroUsize::new(4).unwrap();
/// Default consecutive holder faults before tripping lookup cooldown.
pub const DEFAULT_HOLDER_FAIL_THRESHOLD: u32 = 3;
/// Default holder lookup cooldown window in microseconds (`30.0s`).
pub const DEFAULT_HOLDER_COOLDOWN_US: i64 = 30_000_000;
const DEFAULT_HOLDER_COOLDOWN_S: f64 = 30.0;
const COOLDOWN_WARN_INTERVAL_US: i64 = 5_000_000;
/// Additional margin beyond `push_us` (`1s` RTT + `5s` `CANCEL_WAIT`) for half-open Disk probes.
const PROBE_PUSH_MARGIN_US: i64 = 6_000_000;

/// Parse `(holder_fail_threshold, holder_cooldown_us)` from environment strings.
pub fn parse_holder_circuit_breaker_values(
    threshold: Option<&str>,
    cooldown_s: Option<&str>,
) -> (u32, i64) {
    let threshold = threshold
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(DEFAULT_HOLDER_FAIL_THRESHOLD);
    let cooldown_s = cooldown_s
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(DEFAULT_HOLDER_COOLDOWN_S);
    (threshold, (cooldown_s * 1e6) as i64)
}

/// Parse `(holder_fail_threshold, holder_cooldown_us)` from process env.
pub fn parse_holder_circuit_breaker_env() -> (u32, i64) {
    let threshold = std::env::var("TOKENSPEED_PEER_HOLDER_FAIL_THRESHOLD").ok();
    let cooldown_s = std::env::var("TOKENSPEED_PEER_HOLDER_COOLDOWN_S").ok();
    parse_holder_circuit_breaker_values(threshold.as_deref(), cooldown_s.as_deref())
}

#[derive(Debug, Clone, Default)]
struct HolderHealth {
    consecutive_failures: u32,
    cooldown_until_us: Option<i64>,
    cooldown_started_us: Option<i64>,
    last_recovered_us: Option<i64>,
    last_warn_us: Option<i64>,
    suppressed_since_warn: u64,
}

/// Replica-agreed policy parameters exchanged at startup (`ReplicaSettings`)
/// so every TP/PP member enforces identical quarantine and circuit-breaker
/// rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicaSettings {
    /// Failed-read destination block quarantine window in microseconds.
    pub quarantine_us: i64,
    /// Consecutive holder faults before entering lookup cooldown (`0` disables).
    pub holder_fail_threshold: u32,
    /// Holder lookup cooldown duration in microseconds.
    pub holder_cooldown_us: i64,
}

impl ReplicaSettings {
    /// Encode for intra-replica `all_gather` / `broadcast`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.i64(self.quarantine_us);
        out.u32(self.holder_fail_threshold);
        out.i64(self.holder_cooldown_us);
        out.finish()
    }

    /// Decode from intra-replica `all_gather` / `broadcast`.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        let settings = Self {
            quarantine_us: reader.i64()?,
            holder_fail_threshold: reader.u32()?,
            holder_cooldown_us: reader.i64()?,
        };
        reader.finish(settings)
    }
}

/// What a replica's peer sharing runs with.
#[derive(Debug, Clone)]
pub struct PeerSettings {
    /// The replica's name: its endpoint, `host:port`.
    pub name: String,
    /// The process's incarnation.
    pub incarnation: u64,
    /// The cache namespace's deployment id: everything that shapes a
    /// cached page's bytes but the weights.
    pub deployment_id: String,
    /// The weight version the namespace starts at.
    pub weight_version: String,
    /// The waits a landing allows itself.
    pub timeouts: Timeouts,
    /// How long a holder keeps a lease's record past its last request.
    pub record_lifetime: Duration,
    /// Consecutive holder faults before entering cooldown (`0` disables).
    pub holder_fail_threshold: u32,
    /// Holder cooldown window in microseconds.
    pub holder_cooldown_us: i64,
}

impl PeerSettings {
    /// The namespace: only a holder of the same identity is read.
    fn identity(&self) -> String {
        namespace(&self.deployment_id, &self.weight_version)
    }
}

fn namespace(deployment_id: &str, weight_version: &str) -> String {
    format!("{deployment_id}/{weight_version}")
}

/// A round's agreement could not be reached.
#[derive(Debug, thiserror::Error)]
pub enum PeerError {
    /// A member's report did not decode.
    #[error("a malformed peer report")]
    Report,
    /// The endpoint did not start.
    #[error("the peer endpoint: {0}")]
    Net(#[from] ts_pd::NetError),
    /// Only the first member owns the replica's listener.
    #[error("the first peer member must own a listener, and other members must not")]
    Listener,
    /// The local replica leaders could not exchange endpoints.
    #[error("peer node listener setup: {0}")]
    Mesh(#[from] ts_collective::Error),
    /// A thread did not start.
    #[error("starting a peer thread: {0}")]
    Thread(#[from] std::io::Error),
}

/// What one member shares each round.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Report {
    now_us: i64,
    /// The first member's: the holder's changes since the last round.
    changes: Vec<PeerEventView>,
    /// The first member's: the replies it received.
    replies: Vec<Reply>,
    /// This member's reads that ended, by landing request.
    outcomes: Vec<(String, ReadOutcome)>,
}

impl Report {
    fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.i64(self.now_us);
        out.count(self.changes.len());
        for change in &self.changes {
            write_change(&mut out, change);
        }
        out.count(self.replies.len());
        for reply in &self.replies {
            out.bytes(&reply.encode());
        }
        out.count(self.outcomes.len());
        for (request, outcome) in &self.outcomes {
            out.bytes(request.as_bytes());
            out.u8(match outcome {
                ReadOutcome::Failed => 0,
                ReadOutcome::Ok => 1,
                ReadOutcome::Busy => 2,
            });
        }
        out.finish()
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        let now_us = reader.i64()?;
        let count = reader.count()?;
        let mut changes = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            changes.push(read_change(&mut reader)?);
        }
        let count = reader.count()?;
        let mut replies = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            replies.push(Reply::decode(reader.bytes()?).ok()?);
        }
        let count = reader.count()?;
        let mut outcomes = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            let request = String::from_utf8(reader.bytes()?.to_vec()).ok()?;
            let outcome = match reader.u8()? {
                0 => ReadOutcome::Failed,
                1 => ReadOutcome::Ok,
                2 => ReadOutcome::Busy,
                _ => return None,
            };
            outcomes.push((request, outcome));
        }
        reader.finish(Self {
            now_us,
            changes,
            replies,
            outcomes,
        })
    }
}

/// What a round's agreement feeds the scheduler: the agreed clock, then the
/// events in order (`Scheduler::advance_peer`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerRound {
    /// The agreed peer clock.
    pub now_us: i64,
    /// The holder's changes first, then the reader's answers and landings.
    pub events: Vec<PeerEventView>,
    /// Confirmed remote intervals for per-request usage attribution.
    pub landed: Vec<(String, i32, i32)>,
    /// Attribution for lookup decisions and failed landings on the reporting owner.
    pub decisions: Vec<(String, &'static str)>,
}

/// A holder's layout, as a reader knows it.
#[derive(Debug, Clone)]
struct Layout {
    incarnation: u64,
    ranks: Vec<RankGeometry>,
}

/// A bound peer endpoint, owned continuously from binding through startup.
pub struct PeerListener {
    net: Arc<PeerNet>,
    received: mpsc::Receiver<Vec<Bytes>>,
}

/// The public listener dispatches both holder requests and reader replies.
struct NodeRoutes {
    own_rank: u32,
    local_ports: HashMap<u32, u16>,
}

impl NodeRoutes {
    fn dispatch(&self, net: &PeerNet, mut frames: Vec<Bytes>) -> Option<Vec<Bytes>> {
        let rank = match frames.as_slice() {
            // The original two-frame address names DP rank zero.
            [_, _] => 0,
            [destination, _, _] => {
                let Ok(bytes) = destination.as_ref().try_into() else {
                    tracing::warn!(
                        bytes = destination.len(),
                        "invalid peer destination rank frame"
                    );
                    return None;
                };
                let rank = u32::from_be_bytes(bytes);
                frames.remove(0);
                rank
            }
            _ => {
                tracing::warn!(frames = frames.len(), "unexpected peer message frame count");
                return None;
            }
        };
        if rank == self.own_rank {
            return Some(frames);
        }
        if let Some(port) = self.local_ports.get(&rank) {
            if let Err(error) = net.forward(*port, frames) {
                tracing::warn!(rank, %error, "peer message could not be forwarded");
            }
        } else {
            tracing::warn!(
                rank,
                own_rank = self.own_rank,
                "peer destination rank is not on this node"
            );
        }
        None
    }

    /// Route public traffic independently of the local holder's scheduler lock.
    fn start(
        self,
        net: &Arc<PeerNet>,
        incoming: mpsc::Receiver<Vec<Bytes>>,
    ) -> std::io::Result<mpsc::Receiver<Vec<Bytes>>> {
        let net = Arc::downgrade(net);
        let (local, received) = mpsc::channel();
        thread::Builder::new()
            .name("ts-peer-router".into())
            .spawn(move || {
                for frames in incoming {
                    let Some(net) = net.upgrade() else { break };
                    if let Some(frames) = self.dispatch(&net, frames) {
                        if local.send(frames).is_err() {
                            break;
                        }
                    }
                }
            })?;
        Ok(received)
    }
}

impl PeerListener {
    /// Bind on `host` at `port` (0 lets the OS choose) and return the listener.
    /// Its port stays reserved while the caller prepares the replica's name
    /// and scheduler, then moves the listener into `Peer::start`.
    pub fn bind(host: &str, port: u16) -> Result<Self, PeerError> {
        let (incoming, received) = mpsc::channel();
        let net = Arc::new(PeerNet::start(host, port, incoming)?);
        Ok(Self { net, received })
    }

    /// Bind one public control endpoint for the node. Its replica leaders
    /// exchange private loopback ports once, before they advertise readiness.
    pub fn bind_node(
        mesh: &mut Mesh,
        group_id: u32,
        leaders: &[(usize, u32)],
        host: &str,
        port: u16,
    ) -> Result<Self, PeerError> {
        let ranks = leaders.iter().map(|&(rank, _)| rank).collect();
        let group = Group::new(group_id, ranks, mesh.rank())?;
        // Reserve the public port before the OS assigns any loopback ports.
        let public = if group.index() == 0 {
            Some(Self::bind(host, port)?)
        } else {
            None
        };
        mesh.broadcast(&group, 0, Vec::new())?;
        let mut listener = match public {
            Some(listener) => listener,
            None => Self::bind("127.0.0.1", 0)?,
        };
        let ports = mesh.all_gather_i64(&group, &[i64::from(listener.port())])?;
        if group.index() == 0 {
            let routes = NodeRoutes {
                own_rank: leaders[0].1,
                local_ports: leaders
                    .iter()
                    .zip(ports)
                    .skip(1)
                    .map(|(&(_, rank), row)| (rank, row[0] as u16))
                    .collect(),
            };
            listener.received = routes.start(&listener.net, listener.received)?;
        }
        Ok(listener)
    }

    /// Return the bound port, including the OS-selected port when binding to 0.
    pub fn port(&self) -> u16 {
        self.net.port()
    }
}

/// The first member's endpoint and the threads around it.
struct Endpoint {
    net: Arc<PeerNet>,
    holder: Arc<Mutex<Holder>>,
    replies: mpsc::Receiver<Reply>,
    _servicer: thread::JoinHandle<()>,
}

/// Where a member's data plane reports, in plan order, that a plan with
/// landings ran its device prerequisites: the stores reading blocks the plan
/// granted, its copies and zeroing, the forwards in flight. Only then may the
/// holder's pages be read into the plan's blocks.
#[derive(Debug, Clone)]
pub struct PeerFence(mpsc::Sender<()>);

impl PeerFence {
    /// The oldest such plan not yet reported ran its device work.
    pub fn ready(&self) {
        // A stopped loop has nothing left to read.
        let _ = self.0.send(());
    }
}

/// One member's peer sharing.
pub struct Peer {
    settings: PeerSettings,
    /// The namespace this member reads and serves under; a weight update
    /// moves it.
    identity: String,
    member: usize,
    members: usize,
    geometry: RankGeometry,
    endpoint: Option<Endpoint>,
    origin: Instant,
    layouts: HashMap<String, Layout>,
    /// Outstanding layout queries and their deadlines on the agreed clock.
    asked: HashMap<String, i64>,
    landings: BTreeMap<String, Landing>,
    /// This member's plans with landings, oldest first, whose device work
    /// has not reported done.
    fences: VecDeque<Vec<String>>,
    fence: PeerFence,
    fenced_plans: mpsc::Receiver<()>,
    /// Landings whose plan's device work has run on this member.
    fenced: HashSet<String>,
    /// Landings this member started reading.
    reading: HashSet<String>,
    reads: Reads,
    metrics: Option<Arc<ts_metrics::PeerMetrics>>,
    read_metrics: Option<Arc<ts_metrics::PeerMetrics>>,
    // Only the endpoint owner measures logical attempts.
    lookup_times: HashMap<u64, (String, Instant, i64)>,
    /// Unconditional lookup deadline tracker (`lease_id -> (holder, started_us, deadline_us)`),
    /// kept identical across all replica members so lookup timeouts trip `holder_health`.
    pending_lookups: BTreeMap<u64, (String, i64, i64)>,
    decisions: Vec<(String, &'static str)>,
    next_sample_us: i64,
    previous_totals: [u64; 2],
    /// Reads that could not be placed, to report as failed.
    unplaced: Vec<(String, ReadOutcome)>,
    /// Per-holder physical read failure circuit-breaker state (mirrored across members).
    holder_health: HashMap<String, HolderHealth>,
    /// Consecutive physical read failures before entering cooldown (`0` disables).
    holder_fail_threshold: u32,
    /// Holder cooldown window in microseconds.
    holder_cooldown_us: i64,
    /// Synthetic miss answers for lookups short-circuited by holder cooldown.
    cooldown_answers: Vec<PeerEventView>,
}

impl Peer {
    /// Start one member with every member's geometry in `replica`.
    /// The first (`member` 0) must supply the bound `listener` and serves
    /// `scheduler` as a holder; other members must supply `None`. The
    /// member reads through `reader` and has holders' Disk pages pushed
    /// through `pusher` (its rank's push endpoint).
    pub fn start(
        settings: PeerSettings,
        member: usize,
        replica: Vec<RankGeometry>,
        listener: Option<PeerListener>,
        scheduler: Arc<Mutex<Scheduler>>,
        reader: Arc<dyn PageReader>,
        pusher: Arc<dyn PagePusher>,
    ) -> Result<Self, PeerError> {
        let members = replica.len();
        let geometry = replica.get(member).cloned().ok_or(PeerError::Report)?;
        if listener.is_some() != (member == 0) {
            return Err(PeerError::Listener);
        }
        let (fence, fenced_plans) = mpsc::channel();
        let endpoint = listener
            .map(|listener| start_endpoint(&settings, replica, listener, scheduler))
            .transpose()?;
        let holder_fail_threshold = settings.holder_fail_threshold;
        let holder_cooldown_us = settings.holder_cooldown_us;
        Ok(Self {
            identity: settings.identity(),
            settings,
            member,
            members,
            geometry,
            endpoint,
            origin: Instant::now(),
            layouts: HashMap::new(),
            asked: HashMap::new(),
            landings: BTreeMap::new(),
            fences: VecDeque::new(),
            fence: PeerFence(fence),
            fenced_plans,
            fenced: HashSet::new(),
            reading: HashSet::new(),
            metrics: None,
            read_metrics: None,
            lookup_times: HashMap::new(),
            pending_lookups: BTreeMap::new(),
            decisions: Vec::new(),
            next_sample_us: 0,
            previous_totals: [0; 2],
            reads: Reads::start(reader, pusher, READ_CONCURRENCY)?,
            unplaced: Vec::new(),
            holder_health: HashMap::new(),
            holder_fail_threshold,
            holder_cooldown_us,
            cooldown_answers: Vec::new(),
        })
    }

    /// Override the reader-side holder failure circuit breaker parameters
    /// (`fail_threshold == 0` disables cooldown).
    pub fn set_holder_circuit_breaker(&mut self, fail_threshold: u32, cooldown: Duration) {
        self.holder_fail_threshold = fail_threshold;
        self.holder_cooldown_us = i64::try_from(cooldown.as_micros()).unwrap_or(i64::MAX);
    }

    /// Half-open probe guard duration in microseconds (`max(3 * answer_us, push_us + 6s)`).
    pub fn probe_guard_us(&self) -> i64 {
        self.settings
            .timeouts
            .answer_us
            .saturating_mul(3)
            .max(self.settings.timeouts.push_us.saturating_add(PROBE_PUSH_MARGIN_US))
    }

    /// Drain queued holder changes on the endpoint owner (empty on other members).
    pub fn take_holder_changes(&self) -> Vec<PeerEventView> {
        self.endpoint
            .as_ref()
            .map_or_else(Vec::new, |endpoint| endpoint.holder.lock().take_changes())
    }

    /// Encode holder changes for intra-replica broadcast.
    pub fn encode_changes(changes: &[PeerEventView]) -> Vec<u8> {
        if changes.is_empty() {
            return Vec::new();
        }
        let mut out = Writer::new();
        out.count(changes.len());
        for change in changes {
            write_change(&mut out, change);
        }
        out.finish()
    }

    /// Decode holder changes from intra-replica broadcast.
    pub fn decode_changes(bytes: &[u8]) -> Result<Vec<PeerEventView>, PeerError> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let mut reader = Reader::new(bytes);
        let count = reader.count().ok_or(PeerError::Report)?;
        let mut changes = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            changes.push(read_change(&mut reader).ok_or(PeerError::Report)?);
        }
        reader.finish(changes).ok_or(PeerError::Report)
    }

    /// Inspect `(consecutive_failures, cooldown_until_us)` for `holder`.
    pub fn holder_cooldown_state(&self, holder: &str) -> (u32, Option<i64>) {
        self.holder_health
            .get(holder)
            .map_or((0, None), |h| (h.consecutive_failures, h.cooldown_until_us))
    }

    /// Enable logical measurements on member zero and physical measurements on every member.
    pub fn set_metrics(
        &mut self,
        logical: Option<Arc<ts_metrics::PeerMetrics>>,
        physical: Option<Arc<ts_metrics::PeerMetrics>>,
    ) {
        self.metrics = logical;
        self.read_metrics = physical;
    }

    /// The native counters survive gaps between scrapes; gauges are sampled once per second.
    pub fn sample_scheduler(&mut self, scheduler: &mut Scheduler, now_us: i64) {
        let Some(metrics) = self.metrics.as_ref().filter(|_| self.member == 0) else {
            return;
        };
        for sample in scheduler.take_peer_pin_samples() {
            metrics.pin(sample.held_us, sample.remaining_us, sample.expired);
        }
        if now_us < self.next_sample_us {
            return;
        }
        self.next_sample_us = now_us.saturating_add(1_000_000);
        let stats = scheduler.peer_stats();
        for (name, n) in [
            ("leases", stats.leases),
            ("pinned_blocks", stats.pinned_blocks),
            ("looking_requests", stats.looking_requests),
            ("landing_requests", stats.landing_requests),
        ] {
            metrics.snapshot(name, n);
        }
        let totals = [stats.landed_tokens, stats.failed_landings];
        for (i, name) in ["landed_tokens", "failed_landings"].iter().enumerate() {
            metrics.increment(name, totals[i].saturating_sub(self.previous_totals[i]));
        }
        self.previous_totals = totals;
    }

    /// Drop an unanswered lookup when its owning request terminates.
    pub fn end_lookup(&mut self, rid: &str, outcome: &str) {
        if let Some(metrics) = &self.metrics {
            self.lookup_times.retain(|_, (owner, start, _)| {
                if owner != rid {
                    return true;
                }
                metrics.stage("lookup", outcome, start.elapsed().as_secs_f64());
                false
            });
        }
    }

    /// The bound port of the first member's endpoint.
    pub fn port(&self) -> Option<u16> {
        self.endpoint.as_ref().map(|endpoint| endpoint.net.port())
    }

    /// This member's report for the round.
    pub fn report(&mut self) -> Vec<u8> {
        let mut outcomes = std::mem::take(&mut self.unplaced);
        outcomes.extend(self.reads.poll());
        let mut report = Report {
            now_us: i64::try_from(self.origin.elapsed().as_micros()).unwrap_or(i64::MAX),
            outcomes,
            ..Report::default()
        };
        if let Some(endpoint) = &self.endpoint {
            report.changes = endpoint.holder.lock().take_changes();
            report.replies = endpoint.replies.try_iter().collect();
        }
        report.encode()
    }

    /// The round's agreement over every member's report, in member order;
    /// steps every landing, sends what the steps ask for and starts this
    /// member's reads.
    pub fn agree(&mut self, reports: &[Vec<u8>]) -> Result<PeerRound, PeerError> {
        let reports = reports
            .iter()
            .map(|bytes| Report::decode(bytes).ok_or(PeerError::Report))
            .collect::<Result<Vec<_>, _>>()?;
        let first = reports.first().ok_or(PeerError::Report)?;
        let now_us = first.now_us;
        self.asked.retain(|_, deadline_us| now_us < *deadline_us);
        let mut events = first.changes.clone();
        events.extend(self.cooldown_answers.drain(..));
        let mut actions = Vec::new();
        for reply in &first.replies {
            self.take_reply(reply, now_us, &mut events, &mut actions);
        }
        let mut timed_out_holders = Vec::new();
        self.pending_lookups
            .retain(|_, (holder, started_us, deadline_us)| {
                if now_us < *deadline_us {
                    true
                } else {
                    timed_out_holders.push((holder.clone(), *started_us));
                    false
                }
            });
        for (holder, started_us) in timed_out_holders {
            self.record_holder_read_failure(&holder, started_us, now_us);
        }
        if let Some(metrics) = &self.metrics {
            self.lookup_times.retain(|_, (rid, start, deadline)| {
                if now_us < *deadline {
                    return true;
                }
                self.decisions.push((rid.clone(), "lookup_timeout"));
                metrics.stage("lookup", "timeout", start.elapsed().as_secs_f64());
                false
            });
        }
        for (member, report) in reports.iter().enumerate() {
            for (request, outcome) in &report.outcomes {
                if let Some(landing) = self.landings.get_mut(request) {
                    if let Some(action) = landing.read(member, *outcome, now_us) {
                        actions.push((request.clone(), action));
                    }
                }
            }
        }
        for (request, landing) in &mut self.landings {
            landing.tick(now_us);
            if self.layouts.contains_key(&landing.fetch().holder) {
                if let Some(action) = landing.layout_known(now_us) {
                    actions.push((request.clone(), action));
                }
            }
        }
        // Drain one-shot `HealthOutcome`s at the exact transition into
        // `Phase::Quarantine` or `Phase::Done` (identically on every replica
        // member) rather than waiting 30s+ for `Quarantine` expiry!
        let mut health_updates = Vec::new();
        for landing in self.landings.values_mut() {
            if let Some(outcome) = landing.take_health_outcome() {
                health_updates.push((landing.fetch().holder.clone(), outcome));
            }
        }
        for (holder_name, outcome) in health_updates {
            match outcome {
                HealthOutcome::BenignMiss => {}
                HealthOutcome::Success { .. } => {
                    self.record_holder_read_success(&holder_name, now_us);
                }
                HealthOutcome::Fault { started_us } => {
                    self.record_holder_read_failure(&holder_name, started_us, now_us);
                }
            }
        }
        self.send(actions, now_us);
        let done: Vec<(String, i32)> = self
            .landings
            .iter()
            .filter_map(|(request, landing)| landing.done().map(|tokens| (request.clone(), tokens)))
            .collect();
        let mut landed = Vec::new();
        for (request, num_tokens) in done {
            tracing::debug!(request, num_tokens, "peer landing done");
            let landing = self.landings.remove(&request).expect("finished landing");
            if self.metrics.is_some()
                && self.member == 0
                && num_tokens == landing.fetch().end_tokens
            {
                landed.push((request.clone(), landing.fetch().begin_tokens, num_tokens));
            } else if let Some(stage) = landing.failure() {
                self.decisions.push((request.clone(), stage));
            }
            self.fenced.remove(&request);
            self.reading.remove(&request);
            events.push(PeerEventView {
                kind: PeerEventKind::FetchDone,
                request_id: request,
                num_tokens,
                ..empty_event()
            });
        }
        while self.fenced_plans.try_recv().is_ok() {
            if let Some(requests) = self.fences.pop_front() {
                self.fenced.extend(requests);
            }
        }
        self.start_reads(now_us);
        Ok(PeerRound {
            now_us,
            events,
            landed,
            decisions: std::mem::take(&mut self.decisions),
        })
    }

    /// A plan's landings and lookups: the landings start, and the first
    /// member sends the lookups. Every member calls this with the same plan.
    pub fn dispatch(
        &mut self,
        fetches: Vec<PeerFetchView>,
        lookups: Vec<PeerLookupView>,
        now_us: i64,
    ) {
        if !fetches.is_empty() {
            self.fences.push_back(
                fetches
                    .iter()
                    .map(|fetch| fetch.request_id.clone())
                    .collect(),
            );
        }
        let mut actions = Vec::new();
        for fetch in fetches {
            let known = self.layouts.contains_key(&fetch.holder);
            let request = fetch.request_id.clone();
            tracing::debug!(
                request,
                holder = fetch.holder,
                begin = fetch.begin_tokens,
                end = fetch.end_tokens,
                layout_known = known,
                "peer landing started"
            );
            let (mut landing, action) =
                Landing::new(fetch, self.members, known, now_us, self.settings.timeouts);
            if self.member == 0 {
                landing.observe_with(self.metrics.clone());
            }
            self.landings.insert(request.clone(), landing);
            actions.push((request, action));
        }
        self.send(actions, now_us);
        for lookup in lookups {
            if self.is_holder_in_cooldown(&lookup.holder, now_us) {
                self.cooldown_answers.push(PeerEventView {
                    kind: PeerEventKind::LookupAnswered,
                    lease: lookup.lease,
                    num_tokens: 0,
                    ..empty_event()
                });
                if self.member == 0 {
                    self.decisions.push((lookup.request_id, "holder_cooldown"));
                    if let Some(metrics) = &self.metrics {
                        metrics.stage("lookup", "holder_cooldown", 0.0);
                    }
                }
                continue;
            }
            self.pending_lookups.insert(
                lookup.lease.id,
                (lookup.holder.clone(), now_us, lookup.deadline_us),
            );
            if let Some(endpoint) = &self.endpoint {
                if self.metrics.is_some() {
                    self.lookup_times.insert(
                        lookup.lease.id,
                        (lookup.request_id, Instant::now(), lookup.deadline_us),
                    );
                }
                let request = Request::Lookup {
                    lease: lookup.lease,
                    prompt: Prompt {
                        prefix_hashes: lookup.prefix_hashes,
                        tokens: lookup.tokens,
                        max_tokens: lookup.max_tokens,
                    },
                };
                if let Err(error) = endpoint.net.request(&lookup.holder, &request) {
                    // The lookup's wait ends at its deadline.
                    tracing::warn!(holder = lookup.holder, %error, "peer lookup not sent");
                }
            }
        }
    }

    /// Where this member's data plane reports its plans' device work; the
    /// reads it allows start with the next round's agreement.
    pub fn fence(&self) -> PeerFence {
        self.fence.clone()
    }

    /// The replica committed new weights under `weight_version`: it reads
    /// and serves only that namespace from now on. The flush that goes with
    /// the update left no landing or pin behind; the holders' layouts are
    /// fetched again, so none of the old namespace is read.
    pub fn set_weight_version(&mut self, weight_version: &str) {
        self.settings.weight_version = weight_version.to_owned();
        self.identity = self.settings.identity();
        self.layouts.clear();
        self.asked.clear();
        self.pending_lookups.clear();
        self.holder_health.clear();
        self.cooldown_answers.clear();
        if let Some(endpoint) = &self.endpoint {
            endpoint.holder.lock().set_identity(self.identity.clone());
        }
    }

    /// Landings in flight.
    pub fn num_landings(&self) -> usize {
        self.landings.len()
    }

    fn take_reply(
        &mut self,
        reply: &Reply,
        now_us: i64,
        events: &mut Vec<PeerEventView>,
        actions: &mut Vec<(String, Action)>,
    ) {
        match reply {
            Reply::Layout { stamp, ranks } => {
                self.asked.remove(&stamp.name);
                if stamp.identity == self.identity && ranks.len() == self.members {
                    self.layouts.insert(
                        stamp.name.clone(),
                        Layout {
                            incarnation: stamp.incarnation,
                            ranks: ranks.clone(),
                        },
                    );
                } else {
                    tracing::warn!(
                        holder = stamp.name,
                        "peer holder of another namespace or shape"
                    );
                }
            }
            Reply::Answer { stamp, answer } => {
                let trusted = self.trust(stamp);
                match answer.kind {
                    PeerAnswerKind::Lookup => {
                        self.pending_lookups.remove(&answer.lease.id);
                        if let (Some(metrics), Some((rid, start, _))) =
                            (&self.metrics, self.lookup_times.remove(&answer.lease.id))
                        {
                            let outcome = if stamp.identity != self.identity {
                                "incompatible"
                            } else if answer.num_tokens == 0 {
                                "miss"
                            } else {
                                "ok"
                            };
                            let decision = match outcome {
                                "ok" => "admission",
                                "miss" => "lookup_miss",
                                _ => "incompatible",
                            };
                            self.decisions.push((rid, decision));
                            metrics.stage("lookup", outcome, start.elapsed().as_secs_f64());
                        }
                        events.push(PeerEventView {
                            kind: PeerEventKind::LookupAnswered,
                            lease: answer.lease.clone(),
                            // A holder of another namespace holds nothing of ours.
                            num_tokens: if stamp.identity == self.identity {
                                answer.num_tokens
                            } else {
                                0
                            },
                            ..empty_event()
                        });
                    }
                    PeerAnswerKind::Pin => {
                        if let Some((request, landing)) = self
                            .landings
                            .iter_mut()
                            .find(|(_, landing)| *landing.lease() == answer.lease)
                        {
                            tracing::debug!(
                                request,
                                held = answer.held,
                                trusted,
                                num_tokens = answer.num_tokens,
                                "peer pin answered"
                            );
                            if let Some(action) = landing.pinned(answer.clone(), trusted, now_us) {
                                actions.push((request.clone(), action));
                            }
                        }
                    }
                    _ => {
                        if let Some(landing) = self
                            .landings
                            .values_mut()
                            .find(|landing| *landing.lease() == answer.lease)
                        {
                            landing.released(answer, trusted, now_us);
                        }
                    }
                }
            }
        }
    }

    /// Whether an answer comes from the holder whose layout is known, in the
    /// same incarnation and namespace; a stale layout is forgotten.
    fn trust(&mut self, stamp: &Stamp) -> bool {
        if stamp.identity != self.identity {
            return false;
        }
        match self.layouts.get(&stamp.name) {
            Some(layout) if layout.incarnation == stamp.incarnation => true,
            Some(_) => {
                self.layouts.remove(&stamp.name);
                false
            }
            None => false,
        }
    }

    /// The first member sends what the steps asked for.
    fn send(&mut self, actions: Vec<(String, Action)>, now_us: i64) {
        let Some(endpoint) = &self.endpoint else {
            return;
        };
        for (request, action) in actions {
            let Some(landing) = self.landings.get(&request) else {
                continue;
            };
            let holder = &landing.fetch().holder;
            let lease = landing.lease().clone();
            let message = match action {
                Action::SendLayout => {
                    if self.asked.contains_key(holder) {
                        continue;
                    }
                    self.asked.insert(
                        holder.clone(),
                        now_us.saturating_add(self.settings.timeouts.answer_us),
                    );
                    Request::Layout {
                        reply_to: self.settings.name.clone(),
                    }
                }
                Action::SendPin => Request::Pin { lease },
                Action::SendRelease { demote } => Request::Release { lease, demote },
            };
            if let Err(error) = endpoint.net.request(holder, &message) {
                if matches!(message, Request::Layout { .. }) {
                    self.asked.remove(holder);
                }
                // The landing's wait ends at its deadline.
                tracing::warn!(holder, %error, "peer request not sent");
            }
        }
    }

    /// Start this member's reads of the landings that are pinned and whose
    /// plan's device work has run here.
    fn start_reads(&mut self, now_us: i64) {
        for (request, landing) in &self.landings {
            if self.reading.contains(request) || !self.fenced.contains(request) {
                continue;
            }
            let Some(pinned) = landing.pages_to_read() else {
                continue;
            };
            let fetch = landing.fetch();
            if let Some(metrics) = &self.read_metrics {
                metrics.fence(
                    &fetch.holder,
                    self.member,
                    (now_us - landing.phase_started_us()) as f64 / 1e6,
                );
            }
            let holder = self
                .layouts
                .get(&fetch.holder)
                .and_then(|layout| layout.ranks.get(self.member));
            let pages: Vec<_> = fetch
                .destinations
                .iter()
                .filter_map(|destination| {
                    pinned
                        .pages
                        .iter()
                        .find(|page| {
                            page.group_id == destination.group_id && page.slot == destination.slot
                        })
                        .map(|page| (*page, *destination))
                })
                .collect();
            // The holder's Disk pages are pushed, the rest read.
            let (disk_pages, read_pages): (Vec<_>, Vec<_>) = pages
                .into_iter()
                .partition(|(source, _)| source.tier == PeerTier::Disk);
            let budget =
                Duration::from_micros(u64::try_from(self.settings.timeouts.push_us).unwrap_or(0));
            let job = holder
                .ok_or_else(|| "no layout of the holder".to_owned())
                .and_then(|holder| {
                    let (total_read_bytes, batches) =
                        read_batches(holder, &self.geometry, &read_pages)
                            .map_err(|error| error.to_string())?;
                    let push = if disk_pages.is_empty() {
                        None
                    } else {
                        Some(PushJob {
                            endpoint: holder.push.clone().ok_or_else(|| {
                                "the holder named Disk pages without a push endpoint".to_owned()
                            })?,
                            session: self.geometry.session.clone(),
                            budget,
                            items: push_items(holder, &self.geometry, &disk_pages)
                                .map_err(|error| error.to_string())?,
                        })
                    };
                    Ok((holder.session.clone(), total_read_bytes, batches, push))
                });
            self.reading.insert(request.clone());
            match job {
                Ok((session, total_read_bytes, batches, push)) => {
                    let observation = self.read_metrics.as_ref().map(|metrics| {
                        metrics.queue_read(
                            fetch.holder.clone(),
                            self.member,
                            total_read_bytes + push.as_ref().map_or(0, PushJob::bytes),
                        )
                    });
                    self.reads.submit_observed(
                        ReadJob {
                            request_id: request.clone(),
                            session,
                            batches,
                            push,
                        },
                        observation,
                    );
                }
                Err(error) => {
                    // Nothing was posted: a failed read with no late writes,
                    // reported with the next round.
                    tracing::warn!(request, %error, "peer read not placed");
                    self.unplaced.push((request.clone(), ReadOutcome::Failed));
                }
            }
        }
    }

    fn record_holder_read_success(&mut self, holder: &str, now_us: i64) {
        if let Some(health) = self.holder_health.get_mut(holder) {
            if health.consecutive_failures > 0 || health.cooldown_until_us.is_some() {
                if self.member == 0 {
                    tracing::info!(
                        holder,
                        previous_failures = health.consecutive_failures,
                        "peer holder read succeeded; resetting failure circuit breaker"
                    );
                }
                if health.cooldown_until_us.is_some() {
                    health.last_recovered_us = Some(now_us);
                }
                health.consecutive_failures = 0;
                health.cooldown_until_us = None;
                health.cooldown_started_us = None;
                health.suppressed_since_warn = 0;
            }
        }
    }

    fn record_holder_read_failure(&mut self, holder: &str, started_us: i64, now_us: i64) {
        if self.holder_fail_threshold == 0 {
            return;
        }
        let health = self.holder_health.entry(holder.to_owned()).or_default();
        // Ignore stale in-flight faults initiated before the holder recovered
        // or before the active cooldown epoch began.
        if health
            .last_recovered_us
            .is_some_and(|recovered_us| started_us < recovered_us)
            || health
                .cooldown_started_us
                .is_some_and(|cd_started_us| started_us < cd_started_us)
        {
            return;
        }
        health.consecutive_failures = health.consecutive_failures.saturating_add(1);
        if health.consecutive_failures >= self.holder_fail_threshold {
            let until = now_us.saturating_add(self.holder_cooldown_us);
            health.cooldown_until_us = Some(until);
            health.cooldown_started_us = Some(now_us);
            health.last_warn_us = Some(now_us);
            health.suppressed_since_warn = 0;
            if self.member == 0 {
                tracing::warn!(
                    holder,
                    consecutive_failures = health.consecutive_failures,
                    threshold = self.holder_fail_threshold,
                    cooldown_s = self.holder_cooldown_us as f64 / 1e6,
                    "peer holder hit read failure threshold; entering lookup cooldown"
                );
            }
        }
    }

    fn is_holder_in_cooldown(&mut self, holder: &str, now_us: i64) -> bool {
        if self.holder_fail_threshold == 0 {
            return false;
        }
        let probe_guard_us = self.probe_guard_us();
        let Some(health) = self.holder_health.get_mut(holder) else {
            return false;
        };
        let Some(until_us) = health.cooldown_until_us else {
            return false;
        };
        if now_us < until_us {
            health.suppressed_since_warn = health.suppressed_since_warn.saturating_add(1);
            let should_warn = health
                .last_warn_us
                .is_none_or(|last| now_us.saturating_sub(last) >= COOLDOWN_WARN_INTERVAL_US);
            if should_warn {
                if self.member == 0 {
                    tracing::warn!(
                        holder,
                        suppressed = health.suppressed_since_warn,
                        remaining_ms = (until_us - now_us) / 1000,
                        "holder_cooldown: skipping peer lookup for unhealthy holder"
                    );
                }
                health.last_warn_us = Some(now_us);
                health.suppressed_since_warn = 0;
            }
            true
        } else {
            // Cooldown expired: allow a single half-open probe and re-arm a
            // probe guard window (`max(3 * answer_us, push_us + 6s)`) so
            // concurrent lookups do not flood the holder during a 16s Disk push.
            health.cooldown_until_us = Some(now_us.saturating_add(probe_guard_us));
            false
        }
    }
}

fn start_endpoint(
    settings: &PeerSettings,
    replica: Vec<RankGeometry>,
    listener: PeerListener,
    scheduler: Arc<Mutex<Scheduler>>,
) -> Result<Endpoint, PeerError> {
    let PeerListener { net, received } = listener;
    let holder = Arc::new(Mutex::new(Holder::new(
        Stamp {
            name: settings.name.clone(),
            incarnation: settings.incarnation,
            identity: settings.identity(),
        },
        replica,
        settings.record_lifetime,
    )));
    let (replied, replies) = mpsc::channel();
    let servicer = {
        let (net, holder) = (Arc::downgrade(&net), holder.clone());
        thread::Builder::new()
            .name("ts-peer-servicer".to_owned())
            .spawn(move || {
                for frames in received {
                    match Inbound::decode(&frames) {
                        Ok(Inbound::Request(Request::Layout { reply_to })) => {
                            let reply = holder.lock().layout_reply();
                            let Some(net) = net.upgrade() else { return };
                            if let Err(error) = net.reply(&reply_to, &reply) {
                                tracing::warn!(to = reply_to, %error, "peer layout reply not sent");
                            }
                        }
                        Ok(Inbound::Request(request)) => {
                            // Waits while the control thread runs a round's
                            // scheduler work; answers between rounds.
                            let (to, reply) = {
                                let scheduler = scheduler.lock();
                                holder.lock().handle(&*scheduler, request, Instant::now())
                            };
                            let Some(net) = net.upgrade() else { return };
                            if let Err(error) = net.reply(&to, &reply) {
                                tracing::warn!(to, %error, "peer reply not sent");
                            }
                        }
                        Ok(Inbound::Reply(reply)) => {
                            if replied.send(reply).is_err() {
                                return;
                            }
                        }
                        Err(_) => tracing::warn!("a malformed peer message dropped"),
                    }
                }
            })?
    };
    Ok(Endpoint {
        net,
        holder,
        replies,
        _servicer: servicer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holder_circuit_breaker_env_parser_defaults_overrides_and_disables() {
        assert_eq!(
            parse_holder_circuit_breaker_values(None, None),
            (3, 30_000_000)
        );
        assert_eq!(
            parse_holder_circuit_breaker_values(Some("0"), Some("12.5")),
            (0, 12_500_000)
        );
        assert_eq!(
            parse_holder_circuit_breaker_values(Some(" 5 "), Some(" 45 ")),
            (5, 45_000_000)
        );
        assert_eq!(
            parse_holder_circuit_breaker_values(Some("bad"), Some("-1.0")),
            (3, 30_000_000)
        );
        assert_eq!(
            parse_holder_circuit_breaker_values(Some("2"), Some("NaN")),
            (2, 30_000_000)
        );
    }

    #[test]
    fn replica_settings_encode_decode_roundtrip() {
        let rep = ReplicaSettings {
            quarantine_us: 66_000_000,
            holder_fail_threshold: 4,
            holder_cooldown_us: 15_000_000,
        };
        assert_eq!(ReplicaSettings::decode(&rep.encode()), Some(rep));
    }
}
