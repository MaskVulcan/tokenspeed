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

//! Two replicas sharing a prefix through their peer endpoints on this host:
//! real schedulers, real sockets, and memory both replicas address through
//! one arena (`docs/rfcs/0004-peer-prefix-sharing.md`).

use std::{
    ffi::CString,
    sync::{mpsc, Arc, Once},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_control::peer::{
    geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
    landing::Timeouts,
    net::{Inbound, PeerNet},
    push::{PagePusher, PushItem},
    read::Arena,
    wire::{Reply, Request},
    Peer, PeerListener, PeerSettings,
};
use ts_scheduler::{
    ForwardEvent, ForwardEventKind, PeerFetchView, PeerTier, RequestSpec, Scheduler,
};

const PAGE: i32 = 16;
const PAGE_BYTES: u64 = 64;
const PAGES: u64 = 64;
const HOLDER_BASE: u64 = 0;
const READER_BASE: u64 = 100_000;
/// A Host-landing reader's Host pool: one page per block.
const READER_HOST_BASE: u64 = 150_000;

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

/// Where a replica lands peer prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Landing {
    Device,
    /// Into a Host pool of `PAGES` one-page blocks, before admission.
    Host,
}

/// A replica's scheduler in `role` (`Fused` or `P`), landing as `landing`.
fn scheduler(name: &str, incarnation: u64, role: &str, landing: Landing) -> Scheduler {
    interpreter();
    let host = match landing {
        Landing::Device => "cfg.disable_l2_cache = True".to_owned(),
        Landing::Host => format!(
            "cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = {PAGES}\n\
             cfg.host_group_packing = [1]\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack"
        ),
    };
    let tier = match landing {
        Landing::Device => {
            "landing_tier=ts.PeerLandingTier.Device, max_host_landing_blocks=0, host_ready_timeout_us=0"
                .to_owned()
        }
        Landing::Host => format!(
            "landing_tier=ts.PeerLandingTier.Host, max_host_landing_blocks={}, \
             host_ready_timeout_us=1_000_000",
            PAGES - 1
        ),
    };
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.{role}
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 512
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = {PAGES}
cfg.num_snapshot_pages = 1
{host}
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={PAGE},
                                        total_pages={PAGES}, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation={incarnation}, max_leases=8,
                                soft_pin_us=1_000_000, hard_pin_us=1_000_000,
                                lookup_timeout_us=1_000_000, min_fetch_tokens={PAGE},
                                {tier})
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn geometry(base: u64) -> RankGeometry {
    RankGeometry {
        session: "local".into(),
        device: vec![DeviceField {
            group_id: 0,
            field_id: "layer.0.kv".into(),
            address: base,
            page_stride: PAGE_BYTES,
            bytes: PAGE_BYTES,
        }],
        device_regions: Vec::new(),
        host: None,
        push: None,
    }
}

/// A Host-landing reader's geometry: its Device fields and a Host pool of
/// one-page blocks at `host_base`.
fn geometry_with_host(base: u64, host_base: u64) -> RankGeometry {
    RankGeometry {
        host: Some(HostPool {
            address: host_base,
            groups: vec![HostGroup {
                parent_bytes: PAGE_BYTES,
                packing: 1,
                block_bytes: PAGE_BYTES,
                field_offsets: vec![("layer.0.kv".into(), 0)],
            }],
            region_lengths: Vec::new(),
        }),
        ..geometry(base)
    }
}

/// A replica whose holders never push: none of them has a Disk tier.
struct NoPushes;

impl PagePusher for NoPushes {
    fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
        Err("no holder pushes".into())
    }
}

fn settings(name: &str, weight_version: &str) -> PeerSettings {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_INCARNATION: AtomicU64 = AtomicU64::new(1);
    PeerSettings {
        name: name.to_owned(),
        incarnation: NEXT_INCARNATION.fetch_add(1, Ordering::Relaxed),
        deployment_id: "deployment".into(),
        weight_version: weight_version.to_owned(),
        timeouts: Timeouts {
            answer_us: 1_000_000,
            quarantine_us: 10_000,
            push_us: 1_000_000,
        },
        record_lifetime: Duration::from_secs(5),
        holder_fail_threshold: 3,
        holder_cooldown_us: 30_000_000,
    }
}

/// One single-member replica.
struct Replica {
    name: String,
    scheduler: Arc<Mutex<Scheduler>>,
    peer: Peer,
    decisions: Vec<(String, &'static str)>,
    landed: Vec<(String, i32, i32)>,
}

impl Replica {
    fn start(weight_version: &str, base: u64, arena: &Arena) -> Self {
        Self::start_as(weight_version, base, arena, "Fused")
    }

    fn start_as(weight_version: &str, base: u64, arena: &Arena, role: &str) -> Self {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        Self::with_listener(settings(&name, weight_version), listener, base, arena, role)
    }

    fn with_listener(
        settings: PeerSettings,
        listener: PeerListener,
        base: u64,
        arena: &Arena,
        role: &str,
    ) -> Self {
        Self::with_landing(
            settings,
            listener,
            geometry(base),
            arena,
            role,
            Landing::Device,
        )
    }

    /// A Fused replica landing into Host, its pool at `host_base`.
    fn start_host_landing(weight_version: &str, base: u64, host_base: u64, arena: &Arena) -> Self {
        let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        Self::with_landing(
            settings(&name, weight_version),
            listener,
            geometry_with_host(base, host_base),
            arena,
            "Fused",
            Landing::Host,
        )
    }

    fn with_landing(
        settings: PeerSettings,
        listener: PeerListener,
        geometry: RankGeometry,
        arena: &Arena,
        role: &str,
        landing: Landing,
    ) -> Self {
        let scheduler = Arc::new(Mutex::new(scheduler(
            &settings.name,
            settings.incarnation,
            role,
            landing,
        )));
        let name = settings.name.clone();
        let peer = Peer::start(
            settings,
            0,
            vec![geometry],
            Some(listener),
            scheduler.clone(),
            Arc::new(arena.clone()),
            Arc::new(NoPushes),
        )
        .unwrap();
        Self {
            name,
            scheduler,
            peer,
            decisions: Vec::new(),
            landed: Vec::new(),
        }
    }

    fn enable_logical_metrics(&mut self) {
        let exporter = ts_metrics::Exporter::new(&Default::default()).unwrap();
        let metrics = ts_metrics::PeerMetrics::new(
            &exporter,
            &ts_metrics::EngineLabels {
                model_name: "test".into(),
                app_key: String::new(),
                dp_rank: 0,
            },
        );
        self.peer.set_metrics(Some(Arc::new(metrics)), None);
    }

    /// One round as the control loop runs it: the agreement at the head,
    /// then the plan, its landings and lookups dispatched, and its device
    /// work reported done at once (its reads start with the next
    /// agreement). Returns the plan's prefill rows, as (request, prefix
    /// extended), and its landings.
    fn round(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>) {
        let (rows, fetches, _) = self.round_with_device();
        (rows, fetches)
    }

    /// `round`, and whether its plan went to the data plane.
    fn round_with_device(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>, bool) {
        let report = self.peer.report();
        let round = self.peer.agree(&[report]).unwrap();
        self.decisions.extend(round.decisions);
        self.landed.extend(round.landed);
        let mut scheduler = self.scheduler.lock();
        scheduler.advance_peer(round.now_us, &round.events).unwrap();
        let plan = scheduler.next_execution_plan(true, false);
        let fetches = plan.peer_fetches();
        let lookups = scheduler.take_peer_lookups();
        drop(scheduler);
        self.peer.dispatch(fetches.clone(), lookups, round.now_us);
        // The data plane reports the fence of every plan it runs, as the
        // control loop hands it the plans that need the device.
        let needs_device = plan.needs_device();
        if needs_device && !fetches.is_empty() {
            self.peer.fence().ready();
        }
        let rows = plan
            .forward_batches()
            .into_iter()
            .flat_map(|batch| {
                let extends = batch.num_extends;
                batch
                    .request_ids
                    .into_iter()
                    .zip(batch.extend_prefix_lens)
                    .take(extends)
                    .collect::<Vec<_>>()
            })
            .collect();
        (rows, fetches, needs_device)
    }

    fn advance(&self, events: &[ForwardEvent]) {
        self.scheduler.lock().advance(events).unwrap();
    }
}

fn event(kind: ForwardEventKind, request_id: &str, tokens: Vec<i32>) -> ForwardEvent {
    ForwardEvent {
        kind,
        request_id: request_id.to_owned(),
        tokens,
        spec_candidate_ids: Vec::new(),
        reserve_num_tokens: 0,
        bootstrap_token: -1,
    }
}

fn spec(request_id: &str, tokens: Vec<i32>, holder: &str) -> RequestSpec {
    RequestSpec {
        admission_id: 0,
        request_id: request_id.to_owned(),
        session_id: String::new(),
        cache_group_id: String::new(),
        shared_prefix_tokens: Vec::new(),
        peer_holder: holder.to_owned(),
        tokens,
        max_new_tokens: 4,
        max_cached_prefix_tokens: i32::MAX,
    }
}

/// The holder computes `tokens` and finishes, publishing them; each of its
/// pages holds its block id as a byte pattern.
fn holder_computes(holder: &mut Replica, tokens: &[i32], arena: &Arena) {
    for block in 1..PAGES {
        arena.write(
            HOLDER_BASE + block * PAGE_BYTES,
            &[block as u8; PAGE_BYTES as usize],
        );
    }
    holder
        .scheduler
        .lock()
        .submit_requests(&[spec("h", tokens.to_vec(), "")])
        .unwrap();
    let (rows, _) = holder.round();
    assert_eq!(rows, vec![("h".to_owned(), 0)]);
    holder.advance(&[
        event(ForwardEventKind::ExtendResult, "h", vec![7]),
        event(ForwardEventKind::Finish, "h", Vec::new()),
    ]);
    holder.round();
}

/// Runs both replicas' rounds until the reader prefills `request`; returns
/// the prefix it extends and the landings the reader's plans carried.
fn rounds_until_prefill(
    reader: &mut Replica,
    holder: Option<&mut Replica>,
    request: &str,
) -> (i32, Vec<PeerFetchView>) {
    let mut holder = holder;
    let mut landings = Vec::new();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        if let Some(holder) = holder.as_deref_mut() {
            holder.round();
        }
        let (rows, fetches) = reader.round();
        landings.extend(fetches);
        if let Some((_, prefix)) = rows.into_iter().find(|(rid, _)| rid == request) {
            return (prefix, landings);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("the reader never prefilled {request}");
}

#[test]
fn a_listener_keeps_its_port_reserved_through_peer_startup() {
    let arena = Arena::new(200_000);
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let port = listener.port();
    let name = format!("127.0.0.1:{port}");
    assert_eq!(
        std::net::TcpListener::bind(&name).unwrap_err().kind(),
        std::io::ErrorKind::AddrInUse
    );
    let replica = Replica::with_listener(
        settings(&name, "v1"),
        listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    assert_eq!(replica.peer.port(), Some(port));
    assert_eq!(
        std::net::TcpListener::bind(&name).unwrap_err().kind(),
        std::io::ErrorKind::AddrInUse
    );
}

#[test]
fn two_dp_replicas_read_through_one_public_port() {
    for public_reads in [true, false] {
        read_through_node_port(public_reads);
    }
}

fn node_listeners() -> [PeerListener; 2] {
    use ts_collective::{Listener, Mesh};

    let mesh_listeners = [(); 2].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addresses = mesh_listeners
        .each_ref()
        .map(|socket| format!("127.0.0.1:{}", socket.port().unwrap()));
    std::thread::scope(|scope| {
        let addresses = &addresses;
        let spawn = |rank, socket| {
            scope.spawn(move || {
                let mut mesh =
                    Mesh::connect(rank, socket, addresses, Duration::from_secs(5)).unwrap();
                PeerListener::bind_node(&mut mesh, 4, &[(0, 16), (1, 31)], "127.0.0.1", 0).unwrap()
            })
        };
        let [a, b] = mesh_listeners;
        [spawn(0, a), spawn(1, b)].map(|worker| worker.join().unwrap())
    })
}

#[test]
fn sibling_receives_requests_while_public_scheduler_is_locked() {
    let arena = Arena::new(200_000);
    let [public, private] = node_listeners();
    let address = format!("127.0.0.1:{}", public.port());
    let replica = |rank, listener, base| {
        Replica::with_listener(
            settings(&format!("{address}/{rank}"), "v1"),
            listener,
            base,
            &arena,
            "Fused",
        )
    };
    let owner = replica(16, public, READER_BASE);
    let sibling = replica(31, private, HOLDER_BASE);
    let (incoming, replies) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let client_addr = format!("127.0.0.1:{}", client.port());
    let layout = Request::Layout {
        reply_to: client_addr.clone(),
    };
    let receive_layout = || {
        let frames = replies.recv_timeout(Duration::from_secs(5)).unwrap();
        let Inbound::Reply(Reply::Layout { stamp, .. }) = Inbound::decode(&frames).unwrap() else {
            panic!("expected a layout reply");
        };
        stamp.name
    };
    let receive_pin = || {
        let frames = replies.recv_timeout(Duration::from_secs(5)).unwrap();
        let Inbound::Reply(Reply::Answer { stamp, .. }) = Inbound::decode(&frames).unwrap() else {
            panic!("expected a pin answer reply");
        };
        stamp.name
    };
    // 1. `Request::Layout` on `owner` answers immediately WITHOUT locking `owner.scheduler`!
    let lock = owner.scheduler.lock();
    client.request(&owner.name, &layout).unwrap();
    assert_eq!(receive_layout(), owner.name);

    // 2. Both `Request::Pin` messages enter the same public socket in FIFO order;
    //    `owner`'s servicer waits on `owner.scheduler.lock()`, while `ts-peer-router`
    //    forwards `sibling`'s request without head-of-line blocking!
    let pin = Request::Pin {
        lease: ts_scheduler::PeerLeaseView {
            reader: client_addr,
            id: 77,
        },
    };
    client.request(&owner.name, &pin).unwrap();
    client.request(&sibling.name, &pin).unwrap();
    assert_eq!(receive_pin(), sibling.name);
    drop(lock);
    assert_eq!(receive_pin(), owner.name);
}

fn read_through_node_port(public_reads: bool) {
    let arena = Arena::new(200_000);
    let [public, private] = node_listeners();
    let address = format!("127.0.0.1:{}", public.port());
    let (reader_listener, reader_rank, holder_listener, holder_rank) = if public_reads {
        (public, 16, private, 31)
    } else {
        (private, 31, public, 16)
    };
    let mut reader = Replica::with_listener(
        settings(&format!("{address}/{reader_rank}"), "v1"),
        reader_listener,
        READER_BASE,
        &arena,
        "Fused",
    );
    let mut holder = Replica::with_listener(
        settings(&format!("{address}/{holder_rank}"), "v1"),
        holder_listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    let mut longer = tokens;
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!(landings.len(), 1);
    assert!(prefix >= 5 * PAGE);
    assert_eq!(landings[0].end_tokens, prefix);
    for destination in &landings[0].destinations {
        let page = arena.read_bytes(
            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(page[0] != 0 && page.iter().all(|&byte| byte == page[0]));
    }
}

#[test]
fn a_reader_lands_the_holders_prefix_byte_for_byte_and_computes_the_rest() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");

    assert_eq!(landings.len(), 1, "one landing");
    let landing = &landings[0];
    assert_eq!(landing.begin_tokens, 0);
    assert_eq!(
        prefix, landing.end_tokens,
        "it computes on from the landed boundary"
    );
    assert!(landing.end_tokens >= 5 * PAGE);
    // Every destination holds one of the holder's pages, byte for byte.
    for destination in &landing.destinations {
        let page = arena.read_bytes(
            READER_BASE + destination.block_id as u64 * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(
            page[0] != 0 && page.iter().all(|&byte| byte == page[0]),
            "slot {}",
            destination.slot
        );
    }
    // The holder applied the release at its next head.
    holder.round();
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));
    assert_eq!(
        reader.scheduler.lock().peer_stats().landed_tokens,
        landing.end_tokens as u64
    );
    assert_eq!(reader.landed, vec![("r".into(), 0, prefix)]);
    assert!(reader.decisions.contains(&("r".into(), "admission")));
}

#[test]
fn a_reader_landing_into_host_admits_the_prefix_as_a_host_hit() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start_host_landing("v1", READER_BASE, READER_HOST_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");

    assert_eq!(landings.len(), 1, "one landing");
    let landing = &landings[0];
    assert!(landing.end_tokens >= 5 * PAGE);
    assert_eq!(
        prefix, landing.end_tokens,
        "admitted with the landed prefix"
    );
    // Every destination is a Host block holding one of the holder's pages.
    for destination in &landing.destinations {
        assert_eq!(destination.tier, PeerTier::Host);
        let page = arena.read_bytes(
            READER_HOST_BASE + (destination.block_id as u64 - 1) * PAGE_BYTES,
            PAGE_BYTES as usize,
        );
        assert!(
            page[0] != 0 && page.iter().all(|&byte| byte == page[0]),
            "slot {}",
            destination.slot
        );
    }
    holder.round();
    let stats = holder.scheduler.lock().peer_stats();
    assert_eq!((stats.leases, stats.pinned_blocks), (0, 0));
    assert_eq!(
        reader.scheduler.lock().peer_stats().landed_tokens,
        landing.end_tokens as u64
    );
}

/// A prefill replica reads a prefix too, and its landing's plan goes to the
/// data plane although it prepays no decode room and so zeroes, copies and
/// computes nothing else: the reads start from the fence the data plane
/// reports for that plan.
#[test]
fn a_prefill_replica_lands_a_prefix_from_a_plan_with_no_other_device_work() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start_as("v1", READER_BASE, &arena, "P");
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    reader.advance(&[event(ForwardEventKind::PdBootstrapped, "r", Vec::new())]);
    let start = Instant::now();
    let mut landing_plans = Vec::new();
    while start.elapsed() < Duration::from_secs(10) {
        holder.round();
        let (rows, fetches, needs_device) = reader.round_with_device();
        if !fetches.is_empty() {
            landing_plans.push(needs_device);
        }
        if let Some((_, prefix)) = rows.into_iter().find(|(rid, _)| rid == "r") {
            assert_eq!(
                landing_plans,
                [true],
                "one landing, from a plan the data plane runs"
            );
            assert!(
                prefix >= 5 * PAGE,
                "it computes on from the landed boundary"
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("the prefill replica never prefilled r; landing plans {landing_plans:?}");
}

#[test]
fn a_holder_that_never_answers_leaves_the_reader_to_compute() {
    let arena = Arena::new(200_000);
    // Hold a real endpoint without a servicer, so no other test can answer it.
    let silent = PeerListener::bind("127.0.0.1", 0).unwrap();
    let silent_name = format!("127.0.0.1:{}", silent.port());
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens: Vec<i32> = (1..=4 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", tokens, &silent_name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, None, "r");
    assert_eq!((prefix, landings.len()), (0, 0));
    assert!(reader.decisions.contains(&("r".into(), "lookup_timeout")));
}

#[test]
fn a_lost_layout_query_does_not_disable_later_landings() {
    let arena = Arena::new(200_000);
    let (incoming, received) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let holder_name = format!("127.0.0.1:{}", proxy.port());
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_endpoint = format!("127.0.0.1:{}", listener.port());
    let mut holder = Replica::with_listener(
        settings(&holder_name, "v1"),
        listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let prefixes: Vec<Vec<i32>> = [1, 1001]
        .into_iter()
        .map(|first| (first..first + 6 * PAGE).collect())
        .collect();
    for tokens in &prefixes {
        holder_computes(&mut holder, tokens, &arena);
    }
    let mut dropped_layout = false;
    for (index, tokens) in prefixes.into_iter().enumerate() {
        let request = format!("r{index}");
        let mut longer = tokens;
        longer.extend(2001..2001 + PAGE);
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&request, longer, &holder_name)])
            .unwrap();
        let start = Instant::now();
        let (prefix, landings) = loop {
            assert!(start.elapsed() < Duration::from_secs(10));
            for frames in received.try_iter() {
                let Inbound::Request(message) = Inbound::decode(&frames).unwrap() else {
                    panic!("the proxy only receives requests");
                };
                // Drop one real message, then restore the network path.
                if matches!(message, Request::Layout { .. }) && !dropped_layout {
                    dropped_layout = true;
                } else {
                    proxy.request(&holder_endpoint, &message).unwrap();
                }
            }
            holder.round();
            let (rows, _) = reader.round();
            if let Some((_, prefix)) = rows.into_iter().find(|(id, _)| id == &request) {
                break (prefix, reader.scheduler.lock().peer_stats().landed_tokens);
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        if index == 0 {
            assert!(dropped_layout);
            assert_eq!((prefix, landings), (0, 0));
        } else {
            assert!(prefix >= 5 * PAGE, "the recovered holder must be reusable");
            assert_eq!(landings, prefix as u64);
        }
        reader.advance(&[
            event(ForwardEventKind::ExtendResult, &request, vec![7]),
            event(ForwardEventKind::Finish, &request, Vec::new()),
        ]);
        reader.round();
    }
}

#[test]
fn an_empty_holder_returns_a_lookup_miss() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens = (1..=4 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", tokens, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!((prefix, landings.len()), (0, 0));
    assert_eq!(reader.decisions, vec![("r".into(), "lookup_miss")]);
}

#[test]
fn a_holder_of_another_namespace_is_never_read() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v2", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    reader.enable_logical_metrics();
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!((prefix, landings.len()), (0, 0));
    assert!(reader.decisions.contains(&("r".into(), "incompatible")));
}

#[test]
fn a_reader_that_moved_to_the_holders_weights_reads_them() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v2", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    reader.peer.set_weight_version("v2");

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!(landings.len(), 1, "one landing");
    assert_eq!(prefix, landings[0].end_tokens);
}

#[test]
fn a_holder_that_moved_to_new_weights_is_no_longer_read() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let mut reader = Replica::start("v1", READER_BASE, &arena);
    let tokens: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &tokens, &arena);
    holder.peer.set_weight_version("v2");

    let mut longer = tokens.clone();
    longer.extend(1000..1000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("r", longer, &holder.name)])
        .unwrap();
    let (prefix, landings) = rounds_until_prefill(&mut reader, Some(&mut holder), "r");
    assert_eq!((prefix, landings.len()), (0, 0));
}

#[test]
fn holder_read_failure_circuit_breaker_enters_cooldown_suppresses_lookups_and_recovers() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use ts_control::peer::{geometry::Spans, read::PageReader};

    struct ToggleReader {
        arena: Arena,
        fail: Arc<AtomicBool>,
    }

    impl PageReader for ToggleReader {
        fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
            if self.fail.load(Ordering::Acquire) {
                return Err("injected HIXL physical read failure".to_owned());
            }
            self.arena.read(session, spans)
        }
    }

    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);

    // Build reader with a toggleable physical PageReader and short cooldown (150 ms).
    let fail = Arc::new(AtomicBool::new(true));
    let toggle_reader = Arc::new(ToggleReader {
        arena: arena.clone(),
        fail: fail.clone(),
    });
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", listener.port());
    let reader_sched = Arc::new(Mutex::new(scheduler(
        &reader_name,
        1,
        "Fused",
        Landing::Device,
    )));
    let mut reader_peer = Peer::start(
        settings(&reader_name, "v1"),
        0,
        vec![geometry(READER_BASE)],
        Some(listener),
        reader_sched.clone(),
        toggle_reader,
        Arc::new(NoPushes),
    )
    .unwrap();
    reader_peer.set_holder_circuit_breaker(3, Duration::from_millis(150));
    let mut reader = Replica {
        name: reader_name,
        scheduler: reader_sched,
        peer: reader_peer,
        decisions: Vec::new(),
        landed: Vec::new(),
    };
    reader.enable_logical_metrics();

    // Seed 5 distinct 6-page prefixes on the holder so reader has no local cache for each.
    let prefixes: Vec<Vec<i32>> = (0..5)
        .map(|k| ((k + 1) * 1000..(k + 1) * 1000 + 6 * PAGE).collect())
        .collect();
    for p in &prefixes {
        holder_computes(&mut holder, p, &arena);
    }

    // Helper to submit, drive to prefill, and finish a request on reader.
    let run_reader_req =
        |reader: &mut Replica, holder: &mut Replica, rid: &str, prefix: &[i32]| -> i32 {
            let mut prompt = prefix.to_vec();
            prompt.extend(90_000..90_000 + PAGE);
            reader
                .scheduler
                .lock()
                .submit_requests(&[spec(rid, prompt, &holder.name)])
                .unwrap();
            let (hit, _) = rounds_until_prefill(reader, Some(holder), rid);
            reader.advance(&[
                event(ForwardEventKind::ExtendResult, rid, vec![7]),
                event(ForwardEventKind::Finish, rid, Vec::new()),
            ]);
            reader.round();
            hit
        };

    // 1st, 2nd, 3rd consecutive physical read failures trip the circuit breaker (threshold = 3).
    assert_eq!(run_reader_req(&mut reader, &mut holder, "f1", &prefixes[0]), 0);
    assert_eq!(reader.peer.holder_cooldown_state(&holder.name).0, 1);
    assert!(reader.peer.holder_cooldown_state(&holder.name).1.is_none());

    assert_eq!(run_reader_req(&mut reader, &mut holder, "f2", &prefixes[1]), 0);
    assert_eq!(reader.peer.holder_cooldown_state(&holder.name).0, 2);
    assert!(reader.peer.holder_cooldown_state(&holder.name).1.is_none());

    assert_eq!(run_reader_req(&mut reader, &mut holder, "f3", &prefixes[2]), 0);
    let (fails, cooldown_until) = reader.peer.holder_cooldown_state(&holder.name);
    assert_eq!(fails, 3);
    assert!(cooldown_until.is_some(), "expected holder to enter cooldown after 3 failures");

    // 4th request while in cooldown: lookup is suppressed immediately with "holder_cooldown"
    // without touching the holder (even if we don't step holder at all!).
    reader.decisions.clear();
    let mut prompt4 = prefixes[3].clone();
    prompt4.extend(90_000..90_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("f4-suppressed", prompt4, &holder.name)])
        .unwrap();
    let (hit4, landings4) = rounds_until_prefill(&mut reader, None, "f4-suppressed");
    assert_eq!((hit4, landings4.len()), (0, 0));
    assert!(
        reader
            .decisions
            .contains(&("f4-suppressed".into(), "holder_cooldown")),
        "expected holder_cooldown decision, got {:?}",
        reader.decisions
    );
    reader.advance(&[
        event(ForwardEventKind::ExtendResult, "f4-suppressed", vec![7]),
        event(ForwardEventKind::Finish, "f4-suppressed", Vec::new()),
    ]);
    reader.round();

    // Wait for cooldown (150 ms) to expire, heal physical reader, and send a
    // 4-request concurrent burst to prove half-open single-probe guard:
    // ONLY the first lookup (`f5-probe`) passes while `f6..f8` are suppressed
    // with `holder_cooldown` (zero stampede!), and once `f5-probe` lands and
    // completes `Release`, `holder_health` resets to `(0, None)`.
    fail.store(false, Ordering::Release);
    std::thread::sleep(Duration::from_millis(200));
    reader.decisions.clear();
    let burst_specs: Vec<RequestSpec> = (0..4)
        .map(|k| {
            let mut p = prefixes[4].clone();
            p.extend((91_000 + k * 100)..(91_000 + k * 100 + PAGE));
            spec(&format!("f5-burst-{k}"), p, &holder.name)
        })
        .collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&burst_specs)
        .unwrap();
    // First round dispatches all 4 lookups (`f5-burst-0` is the single probe;
    // `f5-burst-1..3` are immediately suppressed by the re-armed probe guard,
    // and their `holder_cooldown` decisions are drained on the next `agree()`).
    let (probe_hit, _) = rounds_until_prefill(&mut reader, Some(&mut holder), "f5-burst-0");
    let suppressed_in_half_open = reader
        .decisions
        .iter()
        .filter(|(_, d)| *d == "holder_cooldown")
        .count();
    assert_eq!(
        suppressed_in_half_open, 3,
        "half-open guard must allow exactly 1 probe and suppress the 3 sibling burst lookups, got {:?}",
        reader.decisions
    );
    assert_eq!(
        probe_hit,
        6 * PAGE,
        "post-cooldown single probe must succeed and land the holder's prefix"
    );
    assert_eq!(
        reader.peer.holder_cooldown_state(&holder.name),
        (0, None),
        "successful probe must reset failure counter and clear cooldown"
    );
}

#[test]
fn dead_holder_lookup_timeouts_trip_cooldown_even_without_metrics() {
    let arena = Arena::new(200_000);
    // Silent bound socket: never answers `Lookup`, `Layout`, or `Pin`.
    let silent = PeerListener::bind("127.0.0.1", 0).unwrap();
    let silent_name = format!("127.0.0.1:{}", silent.port());

    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", listener.port());
    // Fast lookup timeout (40 ms) so 2 timeouts run in ~100 ms; metrics intentionally OFF!
    interpreter();
    let reader_sched = Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 512
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = {PAGES}
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = True
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={PAGE},
                                        total_pages={PAGES}, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{reader_name}", incarnation=1, max_leases=8,
                                soft_pin_us=500_000, hard_pin_us=500_000,
                                lookup_timeout_us=40_000, min_fetch_tokens={PAGE},
                                landing_tier=ts.PeerLandingTier.Device,
                                max_host_landing_blocks=0, host_ready_timeout_us=0)
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Arc::new(Mutex::new(
            Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap(),
        ))
    });
    let mut peer_settings = settings(&reader_name, "v1");
    peer_settings.timeouts.answer_us = 40_000;
    let mut reader_peer = Peer::start(
        peer_settings,
        0,
        vec![geometry(READER_BASE)],
        Some(listener),
        reader_sched.clone(),
        Arc::new(arena),
        Arc::new(NoPushes),
    )
    .unwrap();
    reader_peer.set_holder_circuit_breaker(2, Duration::from_secs(30));
    let mut reader = Replica {
        name: reader_name,
        scheduler: reader_sched,
        peer: reader_peer,
        decisions: Vec::new(),
        landed: Vec::new(),
    };

    for idx in 0..2 {
        let tokens: Vec<i32> = ((idx + 1) * 1000..(idx + 1) * 1000 + 4 * PAGE).collect();
        let rid = format!("dead-{idx}");
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&rid, tokens, &silent_name)])
            .unwrap();
        let (hit, landings) = rounds_until_prefill(&mut reader, None, &rid);
        assert_eq!((hit, landings.len()), (0, 0));
        reader.advance(&[
            event(ForwardEventKind::ExtendResult, &rid, vec![7]),
            event(ForwardEventKind::Finish, &rid, Vec::new()),
        ]);
        reader.round();
    }

    let (failures, cooldown_until) = reader.peer.holder_cooldown_state(&silent_name);
    assert_eq!(failures, 2, "lookup timeouts must increment holder_health even without metrics");
    assert!(cooldown_until.is_some(), "dead holder must enter cooldown via lookup timeouts");

    // 3rd request must prefill immediately (< 20 ms, without waiting 40 ms lookup_timeout)!
    let t0 = Instant::now();
    let tokens3: Vec<i32> = (5000..5000 + 4 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("dead-fast-skip", tokens3, &silent_name)])
        .unwrap();
    let (hit3, _) = rounds_until_prefill(&mut reader, None, "dead-fast-skip");
    assert_eq!(hit3, 0);
    assert!(
        t0.elapsed() < Duration::from_millis(25),
        "cooldown must short-circuit dead holder lookup immediately, took {:?}",
        t0.elapsed()
    );
    assert!(reader
        .decisions
        .contains(&("dead-fast-skip".into(), "holder_cooldown")));
}

#[test]
fn unvouched_release_does_not_clear_holder_failures_and_trips_cooldown() {
    let arena = Arena::new(200_000);
    let (incoming, received) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let holder_name = format!("127.0.0.1:{}", proxy.port());
    let listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_endpoint = format!("127.0.0.1:{}", listener.port());
    let mut holder = Replica::with_listener(
        settings(&holder_name, "v1"),
        listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );

    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    reader_settings.timeouts.answer_us = 60_000; // 60 ms release timeout
    let mut reader = Replica::with_listener(
        reader_settings,
        reader_listener,
        READER_BASE,
        &arena,
        "Fused",
    );
    reader
        .peer
        .set_holder_circuit_breaker(2, Duration::from_secs(30));

    let prefixes: Vec<Vec<i32>> = (0..2)
        .map(|k| ((k + 1) * 2000..(k + 1) * 2000 + 6 * PAGE).collect())
        .collect();
    for p in &prefixes {
        holder_computes(&mut holder, p, &arena);
    }

    // Forward `Lookup`, `Layout`, and `Pin` so local `Arena::read` succeeds
    // (`Action::SendRelease { demote: true }`), then DROP `Request::Release`!
    // Under the old bug, `Phase::Read` prematurely called `record_holder_read_success`,
    // resetting `consecutive_failures = 0` even though `Release` timed out (`num_tokens == 0`)!
    for (idx, p) in prefixes.iter().enumerate() {
        let rid = format!("rel-fail-{idx}");
        let mut prompt = p.clone();
        prompt.extend(80_000..80_000 + PAGE);
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&rid, prompt, &holder_name)])
            .unwrap();
        let start = Instant::now();
        let hit = loop {
            assert!(start.elapsed() < Duration::from_secs(5));
            for frames in received.try_iter() {
                let Inbound::Request(message) = Inbound::decode(&frames).unwrap() else {
                    panic!("proxy only receives requests");
                };
                if !matches!(message, Request::Release { .. }) {
                    proxy.request(&holder_endpoint, &message).unwrap();
                }
            }
            holder.round();
            let (rows, _) = reader.round();
            if let Some((_, prefix)) = rows.into_iter().find(|(id, _)| id == &rid) {
                break prefix;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(hit, 0, "unanswered Release must discard the read");
        assert_eq!(
            reader.peer.holder_cooldown_state(&holder_name).0,
            (idx + 1) as u32,
            "failed Release after local read Ok(()) must increment failures, NOT reset them"
        );
        reader.advance(&[
            event(ForwardEventKind::ExtendResult, &rid, vec![7]),
            event(ForwardEventKind::Finish, &rid, Vec::new()),
        ]);
        reader.round();
    }
    assert!(
        reader.peer.holder_cooldown_state(&holder_name).1.is_some(),
        "2 consecutive Release failures must trip holder cooldown"
    );
}

#[test]
fn normal_pin_lru_evictions_under_30s_quarantine_never_trip_holder_cooldown() {
    let arena = Arena::new(200_000);
    let (incoming, received) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let holder_name = format!("127.0.0.1:{}", proxy.port());
    let holder_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_endpoint = format!("127.0.0.1:{}", holder_listener.port());
    let mut holder = Replica::with_listener(
        settings(&holder_name, "v1"),
        holder_listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );

    // Realistic production quarantine_us = 30s (30_000_000 us) on reader!
    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    reader_settings.timeouts.quarantine_us = 30_000_000;
    reader_settings.holder_fail_threshold = 3;
    reader_settings.holder_cooldown_us = 30_000_000;
    let mut reader = Replica::with_listener(
        reader_settings,
        reader_listener,
        READER_BASE,
        &arena,
        "Fused",
    );

    // 3x normal soft-pin evictions before `Pin` arrives (`Pin { held: false, trusted: true }`
    // followed by `Release { held: false, trusted: true }`) must emit `BenignMiss`, NOT `Fault`!
    for idx in 0..3 {
        let prefix: Vec<i32> = ((idx + 1) * 3000..(idx + 1) * 3000 + 6 * PAGE).collect();
        holder_computes(&mut holder, &prefix, &arena);

        let rid = format!("lru-evict-{idx}");
        let mut prompt = prefix;
        prompt.extend(70_000..70_000 + PAGE);
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&rid, prompt, &holder_name)])
            .unwrap();

        let start = Instant::now();
        let hit = loop {
            assert!(start.elapsed() < Duration::from_secs(5));
            for frames in received.try_iter() {
                let Inbound::Request(message) = Inbound::decode(&frames).unwrap() else {
                    panic!("proxy only receives requests");
                };
                if matches!(message, Request::Pin { .. }) {
                    // Simulate normal LRU cache eviction of the soft-pinned prefix
                    // on a busy holder before `Request::Pin` is serviced.
                    holder.scheduler.lock().clear_cache();
                }
                proxy.request(&holder_endpoint, &message).unwrap();
            }
            holder.round();
            let (rows, _) = reader.round();
            if let Some((_, p)) = rows.into_iter().find(|(id, _)| id == &rid) {
                break p;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(hit, 0, "evicted soft-pin must fall back cleanly to local compute");
        assert_eq!(
            reader.peer.holder_cooldown_state(&holder_name),
            (0, None),
            "normal Pin(held=false, trusted=true) LRU eviction must NEVER increment holder failures"
        );
        reader.advance(&[
            event(ForwardEventKind::ExtendResult, &rid, vec![7]),
            event(ForwardEventKind::Finish, &rid, Vec::new()),
        ]);
        reader.round();
    }

    // 4th request without eviction still lands full prefix immediately.
    let prefix4: Vec<i32> = (20_000..20_000 + 6 * PAGE).collect();
    holder_computes(&mut holder, &prefix4, &arena);
    let mut prompt4 = prefix4;
    prompt4.extend(71_000..71_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("after-evicts", prompt4, &holder_name)])
        .unwrap();
    let start = Instant::now();
    let hit4 = loop {
        assert!(start.elapsed() < Duration::from_secs(5));
        for frames in received.try_iter() {
            let Inbound::Request(message) = Inbound::decode(&frames).unwrap() else {
                panic!("proxy only receives requests");
            };
            proxy.request(&holder_endpoint, &message).unwrap();
        }
        holder.round();
        let (rows, _) = reader.round();
        if let Some((_, p)) = rows.into_iter().find(|(id, _)| id == "after-evicts") {
            break p;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    assert!(hit4 >= 5 * PAGE);
}

#[test]
fn three_rdma_faults_trip_cooldown_immediately_without_waiting_for_30s_quarantine() {
    use ts_control::peer::{geometry::Spans, read::PageReader};

    struct AlwaysFailReader;
    impl PageReader for AlwaysFailReader {
        fn read(&self, _session: &str, _spans: &Spans) -> Result<(), String> {
            Err("simulated RDMA/ADXL transfer timeout".to_owned())
        }
    }

    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);

    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    // Realistic 30-second production quarantine!
    reader_settings.timeouts.quarantine_us = 30_000_000;
    reader_settings.holder_fail_threshold = 3;
    reader_settings.holder_cooldown_us = 30_000_000;

    let reader_sched = Arc::new(Mutex::new(scheduler(
        &reader_name,
        1,
        "Fused",
        Landing::Device,
    )));
    let reader_peer = Peer::start(
        reader_settings,
        0,
        vec![geometry(READER_BASE)],
        Some(reader_listener),
        reader_sched.clone(),
        Arc::new(AlwaysFailReader),
        Arc::new(NoPushes),
    )
    .unwrap();
    let mut reader = Replica {
        name: reader_name,
        scheduler: reader_sched,
        peer: reader_peer,
        decisions: Vec::new(),
        landed: Vec::new(),
    };

    let t0 = Instant::now();
    for idx in 0..3 {
        let prefix: Vec<i32> = ((idx + 1) * 4000..(idx + 1) * 4000 + 6 * PAGE).collect();
        holder_computes(&mut holder, &prefix, &arena);
        let rid = format!("imm-fault-{idx}");
        let mut prompt = prefix;
        prompt.extend(72_000..72_000 + PAGE);
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&rid, prompt, &holder.name)])
            .unwrap();
        // `imm-fault-{idx}` enters its 30s destination-buffer `Phase::Quarantine`
        // (keeping its HBM blocks quarantined against late DMA), AND `HealthOutcome::Fault`
        // MUST be drained immediately on that exact tick without waiting 30s!
        let step_start = Instant::now();
        while reader.peer.holder_cooldown_state(&holder.name).0 < (idx + 1) as u32 {
            assert!(
                step_start.elapsed() < Duration::from_secs(3),
                "fault {idx} did not increment holder_health immediately while entering 30s Quarantine"
            );
            holder.round();
            reader.round();
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(
            reader.peer.holder_cooldown_state(&holder.name).0,
            (idx + 1) as u32,
            "HealthOutcome::Fault must be recorded on the exact tick Landing enters Quarantine"
        );
    }

    // All 3 landings are STILL inside their 30s Quarantine (`elapsed << 30s` and
    // `landing_requests == 3`), yet holder cooldown is already active!
    assert!(
        t0.elapsed() < Duration::from_secs(3),
        "cooldown must trip immediately without waiting 30s for Quarantine expiry, took {:?}",
        t0.elapsed()
    );
    assert_eq!(reader.scheduler.lock().peer_stats().landing_requests, 3);
    assert!(reader.peer.holder_cooldown_state(&holder.name).1.is_some());

    // 4th request is immediately suppressed with `holder_cooldown` and prefills locally!
    reader.enable_logical_metrics();
    let prompt4: Vec<i32> = (40_000..40_000 + 6 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("imm-suppressed", prompt4, &holder.name)])
        .unwrap();
    let (hit4, _) = rounds_until_prefill(&mut reader, None, "imm-suppressed");
    assert_eq!(hit4, 0);
    assert!(reader
        .decisions
        .contains(&("imm-suppressed".into(), "holder_cooldown")));
}

#[test]
fn stale_quarantined_and_inflight_faults_from_before_recovery_never_ghost_retrip_cooldown() {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Condvar,
    };
    use ts_control::peer::{geometry::Spans, read::PageReader};

    // First 3 reads fail immediately; the 4th pre-recovery read blocks until
    // AFTER the holder recovers, then fails; the 5th+ (recovery probe) succeeds!
    struct StaleFaultReader {
        arena: Arena,
        calls: AtomicUsize,
        release_stale: (std::sync::Mutex<bool>, Condvar),
        stale_waiting: AtomicBool,
    }

    impl PageReader for StaleFaultReader {
        fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
            let idx = self.calls.fetch_add(1, Ordering::SeqCst);
            if idx < 3 {
                return Err(format!("initial fault {idx}"));
            }
            if idx == 3 {
                self.stale_waiting.store(true, Ordering::Release);
                let guard = self.release_stale.0.lock().unwrap();
                let _guard = self
                    .release_stale
                    .1
                    .wait_while(guard, |released| !*released)
                    .unwrap();
                return Err("late pre-recovery fault arriving after recovery".to_owned());
            }
            self.arena.read(session, spans)
        }
    }

    let arena = Arena::new(300_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    // Give holder plenty of pages and compute 6 distinct 4-page prefixes.
    let prefixes: Vec<Vec<i32>> = (0..6)
        .map(|k| ((k + 1) * 5000..(k + 1) * 5000 + 4 * PAGE).collect())
        .collect();
    for p in &prefixes {
        holder_computes(&mut holder, p, &arena);
    }

    let custom_reader = Arc::new(StaleFaultReader {
        arena: arena.clone(),
        calls: AtomicUsize::new(0),
        release_stale: (std::sync::Mutex::new(false), Condvar::new()),
        stale_waiting: AtomicBool::new(false),
    });
    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    // Quarantine (350 ms) outlives Cooldown (100 ms), modeling the exact
    // 30s quarantine vs 14s recovery race AND testing quarantine expiry!
    reader_settings.timeouts.quarantine_us = 350_000;
    reader_settings.holder_fail_threshold = 3;
    reader_settings.holder_cooldown_us = 100_000;

    let reader_sched = Arc::new(Mutex::new(scheduler(
        &reader_name,
        1,
        "Fused",
        Landing::Device,
    )));
    let reader_peer = Peer::start(
        reader_settings,
        0,
        vec![geometry(READER_BASE)],
        Some(reader_listener),
        reader_sched.clone(),
        custom_reader.clone(),
        Arc::new(NoPushes),
    )
    .unwrap();
    let mut reader = Replica {
        name: reader_name,
        scheduler: reader_sched,
        peer: reader_peer,
        decisions: Vec::new(),
        landed: Vec::new(),
    };

    // Submit 4 concurrent requests at `t = 0`: all 4 pass lookup before cooldown trips!
    // `g0..g2` fail immediately and trip cooldown while entering 350ms Quarantine;
    // `g3` blocks in `StaleFaultReader` (`started_us` is from `t = 0`, before cooldown & recovery!).
    let specs: Vec<RequestSpec> = (0..4)
        .map(|k| {
            let mut prompt = prefixes[k].clone();
            let base = 75_000 + k as i32 * 100;
            prompt.extend(base..base + PAGE);
            spec(&format!("g{k}"), prompt, &holder.name)
        })
        .collect();
    reader.scheduler.lock().submit_requests(&specs).unwrap();

    let start = Instant::now();
    while reader.peer.holder_cooldown_state(&holder.name).1.is_none()
        || !custom_reader.stale_waiting.load(Ordering::Acquire)
    {
        assert!(start.elapsed() < Duration::from_secs(5));
        holder.round();
        let (rows, _) = reader.round();
        for (rid, _) in rows {
            reader.advance(&[
                event(ForwardEventKind::ExtendResult, &rid, vec![7]),
                event(ForwardEventKind::Finish, &rid, Vec::new()),
            ]);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(reader.peer.holder_cooldown_state(&holder.name).0, 3);

    // Wait 130ms for cooldown (100ms) to expire while `g0..g2` are STILL in 350ms Quarantine!
    std::thread::sleep(Duration::from_millis(130));

    // Send recovery probe `g-probe`: it succeeds (`calls == 4`) and resets health to `(0, None)`!
    let mut probe_prompt = prefixes[4].clone();
    probe_prompt.extend(76_000..76_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("g-probe", probe_prompt, &holder.name)])
        .unwrap();
    let (probe_hit, _) = rounds_until_prefill(&mut reader, Some(&mut holder), "g-probe");
    assert!(probe_hit >= 3 * PAGE, "recovery probe must land prefix");
    assert_eq!(reader.peer.holder_cooldown_state(&holder.name), (0, None));
    reader.advance(&[
        event(ForwardEventKind::ExtendResult, "g-probe", vec![7]),
        event(ForwardEventKind::Finish, "g-probe", Vec::new()),
    ]);
    reader.round();

    // Now release `g3` (pre-recovery in-flight fault!) AND wait past 350ms so
    // ALL quarantined landings (`g0..g3`) expire and get drained from `Peer.landings`!
    *custom_reader.release_stale.0.lock().unwrap() = true;
    custom_reader.release_stale.1.notify_all();
    let (g3_hit, _) = rounds_until_prefill(&mut reader, Some(&mut holder), "g3");
    assert_eq!(g3_hit, 0);
    reader.advance(&[
        event(ForwardEventKind::ExtendResult, "g3", vec![7]),
        event(ForwardEventKind::Finish, "g3", Vec::new()),
    ]);

    std::thread::sleep(Duration::from_millis(380));
    for _ in 0..5 {
        holder.round();
        reader.round();
    }
    assert_eq!(
        reader.peer.holder_cooldown_state(&holder.name),
        (0, None),
        "neither Quarantine expiry of g0..g2 nor late pre-recovery fault g3 may increment failures or re-trip cooldown"
    );
}

#[test]
fn half_open_probe_guard_spans_disk_push_window() {
    use ts_control::peer::{geometry::Spans, read::PageReader};

    struct FailReader;
    impl PageReader for FailReader {
        fn read(&self, _session: &str, _spans: &Spans) -> Result<(), String> {
            Err("fault".to_owned())
        }
    }

    let arena = Arena::new(200_000);
    let (incoming, received) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let holder_name = format!("127.0.0.1:{}", proxy.port());
    let holder_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_endpoint = format!("127.0.0.1:{}", holder_listener.port());
    let mut holder = Replica::with_listener(
        settings(&holder_name, "v1"),
        holder_listener,
        HOLDER_BASE,
        &arena,
        "Fused",
    );
    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &prefix, &arena);

    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    // Production answer_us = 50ms, push_us = 10s -> probe_guard_us = 16s (>> 3 * 50ms = 150ms!).
    reader_settings.timeouts.answer_us = 50_000;
    reader_settings.timeouts.push_us = 10_000_000;
    reader_settings.timeouts.quarantine_us = 30_000_000;
    reader_settings.holder_fail_threshold = 2;
    reader_settings.holder_cooldown_us = 80_000;

    let reader_sched = Arc::new(Mutex::new(scheduler(
        &reader_name,
        1,
        "Fused",
        Landing::Device,
    )));
    let reader_peer = Peer::start(
        reader_settings,
        0,
        vec![geometry(READER_BASE)],
        Some(reader_listener),
        reader_sched.clone(),
        Arc::new(FailReader),
        Arc::new(NoPushes),
    )
    .unwrap();
    let mut reader = Replica {
        name: reader_name,
        scheduler: reader_sched,
        peer: reader_peer,
        decisions: Vec::new(),
        landed: Vec::new(),
    };

    reader.enable_logical_metrics();

    // Trip cooldown with 2 RDMA faults (while both stay parked in their 30s Quarantine!).
    for idx in 0..2i32 {
        let rid = format!("pg-fault-{idx}");
        let mut prompt = prefix.clone();
        prompt.extend((77_000 + idx * 100)..(77_000 + idx * 100 + PAGE));
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&rid, prompt, &holder_name)])
            .unwrap();
        let start = Instant::now();
        while reader.peer.holder_cooldown_state(&holder_name).0 < (idx + 1) as u32 {
            assert!(start.elapsed() < Duration::from_secs(5));
            for frames in received.try_iter() {
                let Inbound::Request(msg) = Inbound::decode(&frames).unwrap() else {
                    panic!("expected request");
                };
                proxy.request(&holder_endpoint, &msg).unwrap();
            }
            holder.round();
            reader.round();
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    assert!(reader.peer.holder_cooldown_state(&holder_name).1.is_some());

    // Wait 100ms for cooldown (80ms) to expire; launch 1st half-open probe and hold proxy
    // (simulating an in-flight Disk push taking > 150ms).
    std::thread::sleep(Duration::from_millis(100));
    reader.decisions.clear();
    let mut probe1 = prefix.clone();
    probe1.extend(78_000..78_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("pg-probe-1", probe1, &holder_name)])
        .unwrap();
    reader.round();

    // Sleep 220ms (> old `3 * answer_us = 150ms`, but < `push_us + 6s = 16s`!):
    // a 2nd lookup arriving during the Disk push window MUST still be suppressed!
    std::thread::sleep(Duration::from_millis(220));
    let mut probe2 = prefix;
    probe2.extend(79_000..79_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("pg-probe-2", probe2, &holder_name)])
        .unwrap();
    let (hit2, _) = rounds_until_prefill(&mut reader, None, "pg-probe-2");
    assert_eq!(hit2, 0);
    assert!(
        reader
            .decisions
            .contains(&("pg-probe-2".into(), "holder_cooldown")),
        "second probe at +220ms must still be suppressed by 16s Disk push probe_guard_us, got {:?}",
        reader.decisions
    );
}

#[test]
fn push_busy_backpressure_is_benign_miss() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use ts_control::peer::{push::PushFailure, wire::Stamp};
    use ts_scheduler::{PeerAnswerKind, PeerAnswerView, PeerPageView};

    struct BusyPusher {
        calls: Arc<AtomicUsize>,
    }
    impl PagePusher for BusyPusher {
        fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
            Err("push queue busy".into())
        }
        fn push_checked(
            &self,
            _: &str,
            _: &str,
            _: Duration,
            _: Vec<PushItem>,
        ) -> Result<(), PushFailure> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(PushFailure::Busy)
        }
    }

    let arena = Arena::new(200_000);
    let (incoming, received) = mpsc::channel();
    let proxy = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let holder_name = format!("127.0.0.1:{}", proxy.port());
    let holder_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_endpoint = format!("127.0.0.1:{}", holder_listener.port());
    let mut holder_geom = geometry_with_host(HOLDER_BASE, READER_HOST_BASE);
    holder_geom.push = Some("127.0.0.1:9".to_owned());
    let mut holder = Replica::with_landing(
        settings(&holder_name, "v1"),
        holder_listener,
        holder_geom,
        &arena,
        "Fused",
        Landing::Device,
    );

    // Query `holder`'s real `Stamp` once via `Request::Layout` so synthesized Disk-tier
    // `Pin`/`Release` answers carry the exact namespace identity `Peer` expects.
    let (probe_tx, probe_rx) = mpsc::channel();
    let probe_net = PeerNet::start("127.0.0.1", 0, probe_tx).unwrap();
    probe_net
        .request(
            &holder_endpoint,
            &Request::Layout {
                reply_to: format!("127.0.0.1:{}", probe_net.port()),
            },
        )
        .unwrap();
    let holder_stamp: Stamp = match Inbound::decode(
        &probe_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
    )
    .unwrap()
    {
        Inbound::Reply(Reply::Layout { stamp, .. }) => stamp,
        other => panic!("expected Layout reply, got {other:?}"),
    };

    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    reader_settings.timeouts.quarantine_us = 30_000_000;
    reader_settings.holder_fail_threshold = 2;
    let reader_sched = Arc::new(Mutex::new(scheduler(
        &reader_name,
        1,
        "Fused",
        Landing::Device,
    )));
    let busy_calls = Arc::new(AtomicUsize::new(0));
    let reader_peer = Peer::start(
        reader_settings,
        0,
        vec![geometry(READER_BASE)],
        Some(reader_listener),
        reader_sched.clone(),
        Arc::new(arena.clone()),
        Arc::new(BusyPusher {
            calls: busy_calls.clone(),
        }),
    )
    .unwrap();
    let mut reader = Replica {
        name: reader_name,
        scheduler: reader_sched,
        peer: reader_peer,
        decisions: Vec::new(),
        landed: Vec::new(),
    };

    // Proxy intercepts `Request::Pin` so the reply advertises Disk blocks matching
    // the reader's `PeerFetchView` (driving `BusyPusher::push_checked -> Err(PushFailure::Busy)`),
    // while forwarding `Layout` and `Lookup` normally.
    for idx in 0..3 {
        let prefix: Vec<i32> = ((idx + 1) * 6000..(idx + 1) * 6000 + 6 * PAGE).collect();
        holder_computes(&mut holder, &prefix, &arena);
        let rid = format!("busy-push-{idx}");
        let mut prompt = prefix;
        prompt.extend(85_000..85_000 + PAGE);
        reader
            .scheduler
            .lock()
            .submit_requests(&[spec(&rid, prompt, &holder_name)])
            .unwrap();
        let start = Instant::now();
        let mut expected_end_tokens = 6 * PAGE;
        let mut expected_slots = 6i32;
        let hit = loop {
            assert!(start.elapsed() < Duration::from_secs(5));
            for frames in received.try_iter() {
                let Inbound::Request(msg) = Inbound::decode(&frames).unwrap() else {
                    panic!("expected request");
                };
                if let Request::Pin { lease } = &msg {
                    let pages = (0..expected_slots)
                        .map(|slot| PeerPageView {
                            group_id: 0,
                            slot,
                            tier: PeerTier::Disk,
                            block_id: slot + 1,
                        })
                        .collect();
                    proxy
                        .reply(
                            &lease.reader,
                            &Reply::Answer {
                                stamp: holder_stamp.clone(),
                                answer: PeerAnswerView {
                                    lease: lease.clone(),
                                    kind: PeerAnswerKind::Pin,
                                    num_tokens: expected_end_tokens,
                                    pages,
                                    held: true,
                                },
                            },
                        )
                        .unwrap();
                } else if let Request::Release { lease, .. } = &msg {
                    proxy
                        .reply(
                            &lease.reader,
                            &Reply::Answer {
                                stamp: holder_stamp.clone(),
                                answer: PeerAnswerView {
                                    lease: lease.clone(),
                                    kind: PeerAnswerKind::Release,
                                    num_tokens: 0,
                                    pages: Vec::new(),
                                    held: true,
                                },
                            },
                        )
                        .unwrap();
                } else {
                    proxy.request(&holder_endpoint, &msg).unwrap();
                }
            }
            holder.round();
            let (rows, fetches) = reader.round();
            if let Some(f) = fetches.first() {
                expected_end_tokens = f.end_tokens;
                expected_slots = f.destinations.len() as i32;
            }
            if let Some((_, p)) = rows.into_iter().find(|(id, _)| id == &rid) {
                break p;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(hit, 0, "busy push backpressure must fall back to local compute");
        assert_eq!(
            busy_calls.load(Ordering::SeqCst),
            idx as usize + 1,
            "BusyPusher::push_checked must actually be invoked"
        );
        assert_eq!(
            reader.peer.holder_cooldown_state(&holder_name),
            (0, None),
            "MAX_QUEUED_PUSHES busy backpressure must map to BenignMiss and never trip cooldown"
        );
        reader.advance(&[
            event(ForwardEventKind::ExtendResult, &rid, vec![7]),
            event(ForwardEventKind::Finish, &rid, Vec::new()),
        ]);
        reader.round();
    }
}

#[test]
fn arrived_layout_reply_advances_to_pin_even_past_answer_timeout_without_fault() {
    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &prefix, &arena);

    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let mut reader_settings = settings(&reader_name, "v1");
    // Short 40 ms answer timeout + threshold = 1 so even ONE false Layout timeout
    // fault immediately trips cooldown!
    reader_settings.timeouts.answer_us = 40_000;
    reader_settings.holder_fail_threshold = 1;
    reader_settings.holder_cooldown_us = 30_000_000;
    let mut reader = Replica::with_listener(
        reader_settings,
        reader_listener,
        READER_BASE,
        &arena,
        "Fused",
    );

    let mut prompt = prefix;
    prompt.extend(95_000..95_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("slow-step-layout", prompt, &holder.name)])
        .unwrap();

    // Drive until `reader` dispatches its `Landing` in `Phase::Layout` (sending `Request::Layout`).
    let wait_start = Instant::now();
    loop {
        assert!(wait_start.elapsed() < Duration::from_secs(5));
        holder.round();
        let (_, fetches) = reader.round();
        if !fetches.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    // `holder`'s lock-free `Reply::Layout` arrives in < 1 ms, while `reader` simulates
    // an eager forward step lasting 90 ms (> 2x `answer_us = 40 ms`) before its next `agree()`.
    std::thread::sleep(Duration::from_millis(90));

    let (hit, _) = rounds_until_prefill(&mut reader, Some(&mut holder), "slow-step-layout");
    assert!(
        hit >= 5 * PAGE,
        "arrived Reply::Layout must advance Landing to Phase::Pin instead of timing out, got hit={hit}"
    );
    assert_eq!(
        reader.peer.holder_cooldown_state(&holder.name),
        (0, None),
        "arrived Reply::Layout after a long forward step must emit ZERO false HealthOutcome::Fault"
    );
}

#[test]
fn servicer_demux_delivers_replies_and_layout_while_scheduler_locked_and_batch_drains_lookups() {
    use ts_control::peer::wire::Prompt;
    use ts_scheduler::{PeerAnswerKind, PeerLeaseView};

    let arena = Arena::new(200_000);
    let mut holder_b = Replica::start("v1", HOLDER_BASE, &arena);
    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder_b, &prefix, &arena);

    let mut rank_a = Replica::start("v1", READER_BASE, &arena);
    // Seed a prefix on `rank_a` too so external peers can send `Lookup`s to `rank_a`.
    let seed_a: Vec<i32> = (500..500 + 4 * PAGE).collect();
    holder_computes(&mut rank_a, &seed_a, &arena);

    // Start a peer read on `rank_a` from `holder_b` so `rank_a` has an active `Landing`
    // waiting on `Inbound::Reply` frames from `holder_b`.
    let mut prompt_from_b = prefix;
    prompt_from_b.extend(96_000..96_000 + PAGE);
    rank_a
        .scheduler
        .lock()
        .submit_requests(&[spec("hol-check", prompt_from_b, &holder_b.name)])
        .unwrap();

    // Round 1 on `rank_a`: dispatches `Lookup` to `holder_b` (`holder_b` has NOT answered yet,
    // or answers while we lock `rank_a.scheduler`!).
    // Wait until `rank_a` sends `Lookup` and receives `Lookup` answer so `Landing` starts in
    // `Phase::Layout` on `rank_a`, OR test while `rank_a.scheduler` is locked right before
    // `Inbound::Reply` arrives!
    // Even stronger: hold `rank_a.scheduler.lock()` while 4 `Request::Lookup`s are queued on `rank_a`
    // AHEAD of `Request::Layout` AND an incoming `Inbound::Reply(Lookup)` from `holder_b`!
    let (ext_tx, ext_rx) = mpsc::channel();
    let ext_client = PeerNet::start("127.0.0.1", 0, ext_tx).unwrap();
    let ext_addr = format!("127.0.0.1:{}", ext_client.port());

    // Build a valid `Prompt` for `seed_a` using a scratch scheduler.
    let lookup_prompt = {
        let mut scratch = scheduler("scratch", 99, "Fused", Landing::Device);
        let mut s_tokens = seed_a.clone();
        s_tokens.extend(88_000..88_000 + PAGE);
        scratch
            .submit_requests(&[spec("s", s_tokens, "target")])
            .unwrap();
        let _ = scratch.next_execution_plan(true, false);
        let l = scratch.take_peer_lookups().pop().unwrap();
        Prompt {
            prefix_hashes: l.prefix_hashes,
            tokens: l.tokens,
            max_tokens: l.max_tokens,
        }
    };

    // Step `rank_a` once while `holder_b`'s scheduler is temporarily locked so `rank_a`'s
    // `Lookup` to `holder_b` is in flight and its `Inbound::Reply` has NOT arrived yet!
    let b_guard = holder_b.scheduler.lock();
    let _ = rank_a.round();

    // Now LOCK `rank_a.scheduler` (simulating a long forward or lazy HCCL init) and queue
    // 4 `Request::Lookup` RPCs on `rank_a`'s endpoint!
    let a_guard = rank_a.scheduler.lock();
    for i in 0..4u64 {
        ext_client
            .request(
                &rank_a.name,
                &Request::Lookup {
                    lease: PeerLeaseView {
                        reader: ext_addr.clone(),
                        id: 500 + i,
                    },
                    prompt: lookup_prompt.clone(),
                },
            )
            .unwrap();
    }
    std::thread::sleep(Duration::from_millis(20));

    // 1. Send `Request::Layout` BEHIND the 4 blocked `Lookup`s while `a_guard` is held:
    //    `ts-peer-demux` MUST answer `Layout` immediately without HOL blocking!
    ext_client
        .request(
            &rank_a.name,
            &Request::Layout {
                reply_to: ext_addr.clone(),
            },
        )
        .unwrap();
    let layout_frame = ext_rx
        .recv_timeout(Duration::from_millis(300))
        .expect("Request::Layout was HOL-blocked behind scheduler.lock()-waiting Lookups!");
    assert!(matches!(
        Inbound::decode(&layout_frame).unwrap(),
        Inbound::Reply(Reply::Layout { .. })
    ));

    // 2. Unlock `holder_b.scheduler` so `holder_b` sends `Inbound::Reply(Lookup)` back to
    //    `rank_a` BEHIND the 4 blocked `Lookup`s while `a_guard` is STILL held on `rank_a`!
    drop(b_guard);
    let wait_reply = Instant::now();
    let lookup_events = loop {
        assert!(
            wait_reply.elapsed() < Duration::from_millis(500),
            "Inbound::Reply on rank_a was HOL-blocked behind rank_a.scheduler.lock()!"
        );
        let rep = rank_a.peer.report();
        let agreed = rank_a.peer.agree(&[rep]).unwrap();
        if !agreed.events.is_empty() {
            break agreed.events;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(lookup_events.len(), 1);
    assert!(lookup_events[0].num_tokens >= 5 * PAGE);

    // 3. Drop `a_guard` so `ts-peer-servicer` batch-drains all 4 queued `Lookup`s!
    drop(a_guard);
    for _ in 0..4 {
        let frame = ext_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("queued Lookup was not drained after unlocking scheduler");
        match Inbound::decode(&frame).unwrap() {
            Inbound::Reply(Reply::Answer { answer, .. })
                if answer.kind == PeerAnswerKind::Lookup =>
            {
                assert_eq!(answer.num_tokens, 4 * PAGE);
            }
            other => panic!("expected Lookup answer, got {other:?}"),
        }
    }
}

#[test]
fn trailing_slash_zero_normalized_and_self_incarnation_reply_rejected() {
    use ts_control::peer::wire::Stamp;
    use ts_scheduler::{PeerAnswerKind, PeerAnswerView};

    let arena = Arena::new(200_000);
    let mut holder = Replica::start("v1", HOLDER_BASE, &arena);
    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder_computes(&mut holder, &prefix, &arena);

    // 1. Self-hint on DP0 with non-canonical `"/0"` suffix (`"<host>:<port>/0"`) is skipped immediately.
    let mut self_prompt = prefix.clone();
    self_prompt.extend(97_000..97_000 + PAGE);
    holder
        .scheduler
        .lock()
        .submit_requests(&[spec("self-slash-zero", self_prompt, &format!("{}/0", holder.name))])
        .unwrap();
    let (rows, fetches) = holder.round();
    assert!(fetches.is_empty(), "self '/0' hint must never start a peer landing");
    assert_eq!(rows.len(), 1);
    assert!(rows[0].1 >= 5 * PAGE, "self '/0' hint must admit directly from local cache");
    holder.advance(&[
        event(ForwardEventKind::ExtendResult, "self-slash-zero", vec![7]),
        event(ForwardEventKind::Finish, "self-slash-zero", Vec::new()),
    ]);
    holder.round();

    // 2. Cross-replica hint with non-canonical `"/0"` suffix normalizes to `holder.name`
    //    so `Reply::Layout` (`stamp.name == holder.name`) matches `landing.fetch().holder`!
    let reader_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let reader_name = format!("127.0.0.1:{}", reader_listener.port());
    let reader_settings = settings(&reader_name, "v1");
    let reader_incarnation = reader_settings.incarnation;
    let mut reader = Replica::with_listener(
        reader_settings,
        reader_listener,
        READER_BASE,
        &arena,
        "Fused",
    );
    reader.enable_logical_metrics();

    let mut cross_prompt = prefix.clone();
    cross_prompt.extend(98_000..98_000 + PAGE);
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("cross-slash-zero", cross_prompt, &format!("{}/0", holder.name))])
        .unwrap();
    let (cross_hit, cross_landings) =
        rounds_until_prefill(&mut reader, Some(&mut holder), "cross-slash-zero");
    assert_eq!(cross_landings.len(), 1);
    assert!(cross_hit >= 5 * PAGE);
    assert_eq!(reader.peer.holder_cooldown_state(&holder.name), (0, None));
    reader.advance(&[
        event(ForwardEventKind::ExtendResult, "cross-slash-zero", vec![7]),
        event(ForwardEventKind::Finish, "cross-slash-zero", Vec::new()),
    ]);
    reader.round();

    // 3. Uncanonicalized alias returning a reply with `stamp.incarnation == reader.incarnation`
    //    is rejected as a self-reply (`num_tokens == 0`, `"incompatible"`), never stalling!
    let (alias_tx, alias_rx) = mpsc::channel();
    let alias_net = PeerNet::start("127.0.0.1", 0, alias_tx).unwrap();
    let alias_name = format!("127.0.0.1:{}", alias_net.port());
    reader.decisions.clear();
    let self_inc_prompt: Vec<i32> = (55_000..55_000 + 4 * PAGE).collect();
    reader
        .scheduler
        .lock()
        .submit_requests(&[spec("self-inc-alias", self_inc_prompt, &alias_name)])
        .unwrap();
    let start = Instant::now();
    let hit = loop {
        assert!(start.elapsed() < Duration::from_secs(5));
        for frames in alias_rx.try_iter() {
            if let Ok(Inbound::Request(Request::Lookup { lease, .. })) = Inbound::decode(&frames) {
                let to = lease.reader.clone();
                alias_net
                    .reply(
                        &to,
                        &Reply::Answer {
                            stamp: Stamp {
                                name: alias_name.clone(),
                                incarnation: reader_incarnation,
                                identity: "deployment/v1".into(),
                            },
                            answer: PeerAnswerView {
                                lease,
                                kind: PeerAnswerKind::Lookup,
                                num_tokens: 4 * PAGE,
                                pages: Vec::new(),
                                held: true,
                            },
                        },
                    )
                    .unwrap();
            }
        }
        let (rows, fetches) = reader.round();
        assert!(fetches.is_empty(), "self-incarnation lookup reply must never start a landing");
        if let Some((_, p)) = rows.into_iter().find(|(id, _)| id == "self-inc-alias") {
            break p;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(hit, 0);
    assert!(reader
        .decisions
        .contains(&("self-inc-alias".into(), "incompatible")));
    assert_eq!(reader.peer.holder_cooldown_state(&alias_name), (0, None));
}
