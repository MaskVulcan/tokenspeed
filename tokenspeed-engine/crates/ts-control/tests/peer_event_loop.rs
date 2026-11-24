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

//! CPU `EventLoop` integration tests for peer prefix sharing scheduler lock
//! invariants (`TS-R2-P0-1` Vectors A/B/C and `R2-TEST-P1-2`):
//! - Unlocked `gate()` `all_reduce` and `DataPlane::idle_forward` answer
//!   `Request::Layout`, `Request::Lookup`, and `Request::Pin` immediately.
//! - `plan_and_dispatch`'s `dp_round(allow_unlock = false)` keeps
//!   `self.scheduler` locked after `next_execution_plan` (Vector A).
//! - `commit_down_to` keeps `self.scheduler` locked while overlapping batches
//!   remain in flight (Vector B).
//! - `sync_peer_holder_changes` syncs `Pin`/`Release` across TP ranks before
//!   `clear_l1_cache()` in `finish_drain` and `clear_cache()` in `warm_up`
//!   (Vector C).

use std::{
    collections::{HashMap, VecDeque},
    ffi::CString,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex as StdMutex, Once,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_collective::{Group, Listener, Mesh};
use ts_control::{
    batch_log::BatchLogConfig,
    control::ControlConfig,
    eplb::SlotMoves,
    event_loop::{
        DataPlane, DataPlaneError, Dispatch, DpRound, EventLoop, Frontend, Load, LoopConfig, Ranks,
        Subsystems,
    },
    ingress::{Admission, IngressLimits, LogprobServability},
    output::{FinishLog, ForwardResult, OutputProcessor, Role, StreamedOutput},
    peer::{
        geometry::{DeviceField, RankGeometry},
        landing::Timeouts,
        net::{Inbound, PeerNet},
        push::{PagePusher, PushItem},
        read::Arena,
        wire::{Prompt, Reply, Request},
        Peer, PeerListener, PeerSettings,
    },
};
use ts_pd::LandedRequest;
use ts_scheduler::{
    CacheEvent, ExecutionPlan, PeerAnswerKind, PeerLeaseView, RequestSpec, Scheduler,
};
use ts_wire::{
    control::{
        PauseSchedulerReqInput, ProfileReq, ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput, ResumeSchedulerReqInput,
    },
    ControlOutput, ControlRequest, EngineRequest, SamplingParams, TokenizedGenerateReqInput,
};

const PAGE: i32 = 16;
const PAGE_BYTES: u64 = 64;
const PAGES: usize = 65;

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

fn peer_scheduler(name: &str) -> Scheduler {
    interpreter();
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 256
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = {PAGES}
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = True
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={PAGE},
                                        total_pages={PAGES}, attn_kind=ts.AttnKind.Full,
                                        transfer_policy=ts.CacheTransferPolicy.FullSuffix)]
cfg.peer_config = ts.PeerConfig(name="{name}", incarnation=1, max_leases=16,
                                soft_pin_us=5_000_000, hard_pin_us=5_000_000,
                                lookup_timeout_us=2_000_000, min_fetch_tokens={PAGE},
                                landing_tier=ts.PeerLandingTier.Device,
                                max_host_landing_blocks=0, host_ready_timeout_us=0)
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

struct NoPushes;
impl PagePusher for NoPushes {
    fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
        Err("no holder pushes".into())
    }
}

fn rank_geometry(base: u64) -> RankGeometry {
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

fn peer_settings(name: &str) -> PeerSettings {
    PeerSettings {
        name: name.to_owned(),
        incarnation: 1,
        deployment_id: "deployment".into(),
        weight_version: "v1".into(),
        timeouts: Timeouts {
            answer_us: 1_000_000,
            quarantine_us: 30_000_000,
            push_us: 10_000_000,
        },
        record_lifetime: Duration::from_secs(10),
        holder_fail_threshold: 3,
        holder_cooldown_us: 30_000_000,
    }
}

fn loop_config(in_flight_depth: usize) -> LoopConfig {
    let kv_total_pages = (PAGES - 1) as i64;
    LoopConfig {
        in_flight_depth,
        limits: IngressLimits {
            vocab_size: 100_000,
            eos_token_ids: Vec::new(),
            top_k_exclusive_max: Some(ts_control::sampling::TOP_K_FUSED_MAX),
            max_model_len: 4096,
            max_req_input_len: 4000,
            enable_output_logprobs: false,
            requires_bootstrap: false,
            multimodal: false,
            grammar: false,
            reasoning: None,
            reasoning_usage: None,
            logprobs: LogprobServability {
                supports_prompt_logprobs: true,
                logprob_order: "torch".to_owned(),
                pipeline_parallel: false,
                disaggregated: false,
                lm_head_tp_under_dp: false,
                computes_prompt_logprobs: true,
            },
        },
        kv_total_pages,
        kv_token_capacity: kv_total_pages * i64::from(PAGE),
        ngram_context_len: 0,
        request_token_history: false,
        request_token_history_lookback: None,
        controls: ControlConfig {
            memory_saver: true,
            memory_regions: vec!["kv_cache".to_owned(), "weights".to_owned()],
            shared_weight_version: None,
            allow_reload_forward: true,
        },
        batch_log: BatchLogConfig {
            enabled: false,
            decode_log_interval: 40,
            num_total_pages: kv_total_pages,
            spec_num_steps: 0,
            dp_rank: 0,
            disaggregated: false,
            cache_state_group_ids: Vec::new(),
            log_spec_accept_lengths: false,
        },
        kv_events: false,
        epd_receive_enabled: false,
        eager_grammar: false,
        prefill_delay: None,
    }
}

/// Reusable pause point: when armed (`armed == true`), the thread entering
/// `pause_if_armed()` signals `entered` and blocks until `release()` is called.
#[derive(Default)]
struct PauseHook {
    state: StdMutex<(bool, bool)>, // (armed, entered)
    cv: Condvar,
}

impl PauseHook {
    fn arm(&self) {
        let mut s = self.state.lock().unwrap();
        s.0 = true;
        s.1 = false;
    }

    fn wait_entered(&self, timeout: Duration) {
        let s = self.state.lock().unwrap();
        let (s, res) = self
            .cv
            .wait_timeout_while(s, timeout, |state| !state.1)
            .unwrap();
        assert!(!res.timed_out(), "timed out waiting for PauseHook entry; state={:?}", *s);
    }

    fn release(&self) {
        let mut s = self.state.lock().unwrap();
        s.0 = false;
        s.1 = false;
        self.cv.notify_all();
    }

    fn pause_if_armed(&self) {
        let mut s = self.state.lock().unwrap();
        if !s.0 {
            return;
        }
        s.1 = true;
        self.cv.notify_all();
        drop(self.cv.wait_while(s, |state| state.0).unwrap());
    }
}

#[derive(Default)]
struct ChannelFrontendState {
    inbox: StdMutex<VecDeque<EngineRequest>>,
    finished: StdMutex<Vec<String>>,
    replies: StdMutex<Vec<ControlOutput>>,
    receive_hook: PauseHook,
}

impl ChannelFrontendState {
    fn push(&self, req: EngineRequest) {
        self.inbox.lock().unwrap().push_back(req);
    }

    fn wait_finished(&self, rid: &str, timeout: Duration) {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.finished.lock().unwrap().iter().any(|r| r == rid) {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for request {rid} to finish");
    }

    fn wait_reply_count(&self, count: usize, timeout: Duration) -> Vec<ControlOutput> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            let replies = self.replies.lock().unwrap();
            if replies.len() >= count {
                return replies.clone();
            }
            drop(replies);
            thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for {count} control replies");
    }
}

struct ChannelFrontend {
    state: Arc<ChannelFrontendState>,
}

impl Frontend for ChannelFrontend {
    fn receive(&mut self, wait: Option<Duration>) -> Vec<EngineRequest> {
        self.state.receive_hook.pause_if_armed();
        let mut items: Vec<EngineRequest> = self.state.inbox.lock().unwrap().drain(..).collect();
        if items.is_empty() {
            if let Some(w) = wait {
                thread::sleep(w.min(Duration::from_millis(2)));
                items.extend(self.state.inbox.lock().unwrap().drain(..));
            }
        }
        items
    }

    fn closed(&self) -> bool {
        false
    }

    fn send(&mut self, outputs: Vec<StreamedOutput>, _load: Load) {
        let mut fin = self.state.finished.lock().unwrap();
        for out in outputs {
            if out.finish_reason.is_some() {
                fin.push(out.rid);
            }
        }
    }

    fn reply(&mut self, output: ControlOutput) {
        self.state.replies.lock().unwrap().push(output);
    }
}

#[derive(Default)]
struct DeviceHooks {
    idle_forward_hook: PauseHook,
    complete_hook: PauseHook,
    execute_hook: PauseHook,
    device_ops: StdMutex<Vec<String>>,
}

struct HookedDevice {
    generated: HashMap<String, i32>,
    hooks: Arc<DeviceHooks>,
}

impl HookedDevice {
    fn new(hooks: Arc<DeviceHooks>) -> Self {
        Self {
            generated: HashMap::new(),
            hooks,
        }
    }
}

impl DataPlane for HookedDevice {
    type Pending = ForwardResult;

    fn admit(&mut self, requests: &[Admission]) -> Result<(), DataPlaneError> {
        for req in requests {
            self.generated.insert(req.rid.clone(), 0);
        }
        Ok(())
    }

    fn execute(
        &mut self,
        _plan: ExecutionPlan,
        dispatch: &Dispatch<'_>,
    ) -> Result<Option<ForwardResult>, DataPlaneError> {
        self.hooks.execute_hook.pause_if_armed();
        let Some(batch) = dispatch.forward else {
            return Ok(None);
        };
        let mut result = ForwardResult::default();
        for rid in &batch.request_ids {
            let gen = self.generated.entry(rid.clone()).or_default();
            *gen += 1;
            let token = 1000 + *gen - 1;
            result.output_tokens.push(token);
            result.output_lengths.push(1);
        }
        result.output_logprobs = Some(vec![-1.0; result.output_tokens.len()]);
        Ok(Some(result))
    }

    fn complete(&mut self, pending: ForwardResult) -> Result<ForwardResult, DataPlaneError> {
        self.hooks.complete_hook.pause_if_armed();
        Ok(pending)
    }

    fn poll_cache_events(&mut self) -> Result<Vec<CacheEvent>, DataPlaneError> {
        Ok(Vec::new())
    }

    fn release(&mut self, rids: &[String]) -> Result<(), DataPlaneError> {
        for rid in rids {
            self.generated.remove(rid);
        }
        Ok(())
    }

    fn land(
        &mut self,
        _spec_candidates: Option<(i32, Vec<i32>)>,
        _cache_slot: Option<i32>,
    ) -> Result<(), DataPlaneError> {
        Ok(())
    }

    fn idle_forward(&mut self, _dp: &DpRound) -> Result<(), DataPlaneError> {
        self.hooks.idle_forward_hook.pause_if_armed();
        Ok(())
    }

    fn check_weight_update(
        &mut self,
        _op: &ControlRequest,
    ) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }

    fn update_weights(&mut self, _op: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }

    fn commit_weight_update(&mut self, _success: bool) -> Result<(), DataPlaneError> {
        Ok(())
    }

    fn profile(&mut self, _request: &ProfileReq) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }

    fn release_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        self.hooks
            .device_ops
            .lock()
            .unwrap()
            .push(format!("release:{}", tags.join(",")));
        Ok(())
    }

    fn reload_forward(&mut self, _op: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }

    fn resume_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        self.hooks
            .device_ops
            .lock()
            .unwrap()
            .push(format!("resume:{}", tags.join(",")));
        Ok(())
    }

    fn finish_memory_resume(&mut self) -> Result<(), DataPlaneError> {
        self.hooks
            .device_ops
            .lock()
            .unwrap()
            .push("finish_memory_resume".to_owned());
        Ok(())
    }

    fn attach_grammars(&mut self, _grammars: &[(String, u64, Vec<i32>)]) -> Result<(), DataPlaneError> {
        Ok(())
    }

    fn epd_publish(&mut self, _landed: &[LandedRequest]) -> Result<(), DataPlaneError> {
        Ok(())
    }

    fn epd_released(&mut self) -> Result<Vec<usize>, DataPlaneError> {
        Ok(Vec::new())
    }

    fn snapshot_expert_load(&mut self) -> Result<(Vec<Vec<i64>>, Vec<Vec<i32>>), DataPlaneError> {
        Ok((Vec::new(), Vec::new()))
    }

    fn apply_expert_placement(
        &mut self,
        _layer_ids: &[usize],
        _rows: &[Vec<i32>],
        _moves: &[SlotMoves],
    ) -> Result<(), DataPlaneError> {
        Ok(())
    }
}

fn add_req(rid: &str, tokens: Vec<i32>, max_new_tokens: i64) -> EngineRequest {
    EngineRequest::Add(Box::new(TokenizedGenerateReqInput {
        input_ids: Some(tokens),
        stream: true,
        ..TokenizedGenerateReqInput::new(
            rid,
            SamplingParams {
                max_new_tokens: Some(max_new_tokens),
                ..Default::default()
            },
        )
    }))
}

/// Builds a wire `Prompt` with real scheduler-computed `prefix_hashes` for `prefix_tokens`.
fn make_lookup_prompt(prefix_tokens: &[i32]) -> Prompt {
    let mut helper = peer_scheduler("prompt-builder");
    let mut full = prefix_tokens.to_vec();
    full.extend(99_000..99_000 + PAGE);
    helper
        .submit_requests(&[RequestSpec {
            admission_id: 0,
            request_id: "p".into(),
            session_id: String::new(),
            cache_group_id: String::new(),
            shared_prefix_tokens: Vec::new(),
            peer_holder: "holder".into(),
            tokens: full,
            max_new_tokens: 1,
            max_cached_prefix_tokens: i32::MAX,
        }])
        .unwrap();
    let _ = helper.next_execution_plan(true, false);
    let mut lookups = helper.take_peer_lookups();
    let lookup = lookups.pop().expect("expected one peer lookup");
    Prompt {
        prefix_hashes: lookup.prefix_hashes,
        tokens: lookup.tokens,
        max_tokens: lookup.max_tokens,
    }
}

#[test]
fn unlocked_gate_and_idle_forward_answer_peer_while_plan_dispatch_and_overlap_stay_locked() {
    interpreter();
    let arena = Arena::new(200_000);
    let listeners = [(); 2].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addresses: Vec<String> = listeners
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.port().unwrap()))
        .collect();

    let peer_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_name = format!("127.0.0.1:{}", peer_listener.port());
    let sched0 = Arc::new(Mutex::new(peer_scheduler(&holder_name)));
    let sched1 = Arc::new(Mutex::new(peer_scheduler("dp1-unused")));

    let peer0 = Peer::start(
        peer_settings(&holder_name),
        0,
        vec![rank_geometry(0)],
        Some(peer_listener),
        sched0.clone(),
        Arc::new(arena),
        Arc::new(NoPushes),
    )
    .unwrap();

    let fe0 = Arc::new(ChannelFrontendState::default());
    let fe1 = Arc::new(ChannelFrontendState::default());
    let hooks0 = Arc::new(DeviceHooks::default());
    let hooks1 = Arc::new(DeviceHooks::default());
    let stop = Arc::new(AtomicBool::new(false));

    let [l0, l1] = listeners;
    let spawn_rank = |rank: usize,
                      listener: Listener,
                      sched: Arc<Mutex<Scheduler>>,
                      fe: Arc<ChannelFrontendState>,
                      hooks: Arc<DeviceHooks>,
                      peer: Option<Peer>| {
        let addresses = addresses.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            let mesh = Mesh::connect(rank, listener, &addresses, Duration::from_secs(10)).unwrap();
            let replica = Group::new(1, vec![rank], rank).unwrap();
            let dp = Some(Group::new(0, vec![0, 1], rank).unwrap());
            let outputs = OutputProcessor::new(
                None,
                1,
                None,
                Role::Plain,
                FinishLog {
                    enabled: false,
                    request_stats: false,
                    ttft_percentage: 0,
                },
                false,
                None,
            );
            let mut ev = EventLoop::new(
                sched,
                HookedDevice::new(hooks),
                Some(ChannelFrontend { state: fe }),
                outputs,
                loop_config(2), // overlap schedule enabled (`in_flight_depth = 2`)
                Ranks {
                    mesh,
                    replica,
                    pp: None,
                    dp,
                    ep: None,
                },
                Subsystems {
                    transfer: None,
                    embeddings: None,
                    metrics: None,
                    kv_events: None,
                    eplb: None,
                    grammars: None,
                    l3_metrics: None,
                    l3: None,
                    disk: None,
                    disk_metrics: None,
                    peer,
                },
            );
            ev.run(&stop).unwrap();
        })
    };

    let t0 = spawn_rank(0, l0, sched0.clone(), fe0.clone(), hooks0.clone(), Some(peer0));
    let t1 = spawn_rank(1, l1, sched1.clone(), fe1.clone(), hooks1.clone(), None);

    // 1. Seed a 3-page (48-token) prefix on Rank 0.
    let seed_tokens: Vec<i32> = (1..=3 * PAGE).collect();
    fe0.push(add_req("seed", seed_tokens.clone(), 1));
    fe0.wait_finished("seed", Duration::from_secs(5));

    let (incoming, replies) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let client_addr = format!("127.0.0.1:{}", client.port());

    let lookup_prompt = make_lookup_prompt(&seed_tokens);
    let lookup_and_pin = |lease_id: u64, timeout: Duration| -> Option<(i32, bool)> {
        client
            .request(
                &holder_name,
                &Request::Layout {
                    reply_to: client_addr.clone(),
                },
            )
            .unwrap();
        let f_layout = replies.recv_timeout(timeout).ok()?;
        assert!(matches!(
            Inbound::decode(&f_layout).unwrap(),
            Inbound::Reply(Reply::Layout { .. })
        ));
        let lease = PeerLeaseView {
            reader: client_addr.clone(),
            id: lease_id,
        };
        client
            .request(
                &holder_name,
                &Request::Lookup {
                    lease: lease.clone(),
                    prompt: lookup_prompt.clone(),
                },
            )
            .unwrap();
        let f_lookup = replies.recv_timeout(timeout).ok()?;
        let matched = match Inbound::decode(&f_lookup).unwrap() {
            Inbound::Reply(Reply::Answer { answer, .. })
                if answer.kind == PeerAnswerKind::Lookup =>
            {
                answer.num_tokens
            }
            other => panic!("expected Lookup Answer reply, got {other:?}"),
        };
        client
            .request(&holder_name, &Request::Pin { lease: lease.clone() })
            .unwrap();
        let f_pin = replies.recv_timeout(timeout).ok()?;
        let held = match Inbound::decode(&f_pin).unwrap() {
            Inbound::Reply(Reply::Answer { answer, .. }) if answer.kind == PeerAnswerKind::Pin => {
                answer.held
            }
            other => panic!("expected Pin Answer reply, got {other:?}"),
        };
        client
            .request(
                &holder_name,
                &Request::Release {
                    lease,
                    demote: false,
                },
            )
            .unwrap();
        let _ = replies.recv_timeout(timeout).ok()?;
        Some((matched, held))
    };

    // 2. UNLOCKED `gate()` `all_reduce`:
    //    Pause Rank 1 in `Frontend::receive` BEFORE `gate()`, so Rank 0 enters
    //    `gate()`'s `all_reduce` and blocks waiting on Rank 1 with `self.scheduler` unlocked!
    fe1.receive_hook.arm();
    fe1.receive_hook.wait_entered(Duration::from_secs(5));
    thread::sleep(Duration::from_millis(15));
    assert_eq!(
        lookup_and_pin(101, Duration::from_millis(500)),
        Some((3 * PAGE, true)),
        "ts-peer-servicer must answer Layout, Lookup, and Pin immediately while Rank 0 waits in gate() all_reduce"
    );
    fe1.receive_hook.release();

    // 2b. UNLOCKED `gate()` WITH UN-OVERLAPPED IN-FLIGHT BATCH (`TS-R4-P0-1` regression):
    //     With `in_flight_depth > 0`, dispatch 1 batch (`B_1`) on Rank 0; `B_1`'s pages are
    //     NOT yet overlap-published until Round 2's `next_execution_plan`, so `B_1` must be
    //     pushed with `hold_lock = false` and Rank 0 MUST unlock `self.scheduler` during
    //     Round 2's `bump()` and `gate()` `all_reduce` instead of starving `ts-peer-servicer`!
    //     Pin Rank 1 inside Round 1's `execute` (idle_forward) alongside Rank 0's Round 1
    //     `execute` so `fe1.receive_hook` deterministically catches Round 2 (before Round 2's
    //     `next_execution_plan` sets `B_1.hold_lock = true` for `in_flight_depth = 2` Round 3).
    hooks0.execute_hook.arm();
    hooks1.execute_hook.arm();
    fe0.push(add_req("r0-inflight-gate", (400..400 + 2 * PAGE).collect(), 2));
    hooks0.execute_hook.wait_entered(Duration::from_secs(5));
    hooks1.execute_hook.wait_entered(Duration::from_secs(5));
    fe1.receive_hook.arm();
    hooks0.execute_hook.release();
    hooks1.execute_hook.release();
    fe1.receive_hook.wait_entered(Duration::from_secs(5));
    thread::sleep(Duration::from_millis(15));
    assert_eq!(
        lookup_and_pin(103, Duration::from_millis(500)),
        Some((3 * PAGE, true)),
        "TS-R4-P0-1: ts-peer-servicer must still answer Lookup/Pin in gate() with 1 un-overlapped batch in flight"
    );
    fe1.receive_hook.release();
    fe0.wait_finished("r0-inflight-gate", Duration::from_secs(5));

    // 3. UNLOCKED `DataPlane::idle_forward`:
    //    Pause Rank 0 (`!forward_allowed && memory_resident`) while Rank 1 runs `"r1-only"`,
    //    putting Rank 0 into `ArcMutexGuard::unlocked(..., || data_plane.idle_forward(&dp))`!
    fe0.push(EngineRequest::Control(Box::new(
        ControlRequest::PauseScheduler(PauseSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: "keep".to_owned(),
        }),
    )));
    fe0.wait_reply_count(1, Duration::from_secs(5));

    hooks0.idle_forward_hook.arm();
    fe1.push(add_req("r1-only", (500..500 + 2 * PAGE).collect(), 2));
    hooks0.idle_forward_hook.wait_entered(Duration::from_secs(5));
    assert_eq!(
        lookup_and_pin(102, Duration::from_millis(500)),
        Some((3 * PAGE, true)),
        "ts-peer-servicer must answer Lookup and Pin immediately during DataPlane::idle_forward"
    );
    hooks0.idle_forward_hook.release();
    fe1.wait_finished("r1-only", Duration::from_secs(5));

    fe0.push(EngineRequest::Control(Box::new(
        ControlRequest::ResumeScheduler(ResumeSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
        }),
    )));
    fe0.wait_reply_count(2, Duration::from_secs(5));

    // 4. VECTOR A (`plan_and_dispatch`'s `dp_round(..., allow_unlock = false)`):
    //    Pause Rank 1 in Round R's `execute` with 1 in-flight batch, queue `PauseScheduler(wait)`
    //    on Rank 1 and `"r0-vec-a"` on Rank 0, and arm `hooks1.complete_hook`.
    //    In Round R+1, both pass `gate()`; Rank 1 (`!forward_allowed`) pauses in `commit_down_to(0)`
    //    (`DataPlane::complete`) BEFORE `dp_round`, while Rank 0 calls `next_execution_plan`
    //    and blocks inside `plan_and_dispatch`'s `dp_round(..., false)`!
    hooks1.execute_hook.arm();
    fe1.push(add_req("r1-inflight", (600..600 + 2 * PAGE).collect(), 3));
    hooks1.execute_hook.wait_entered(Duration::from_secs(5));

    fe1.push(EngineRequest::Control(Box::new(
        ControlRequest::PauseScheduler(PauseSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: "keep".to_owned(),
        }),
    )));
    fe0.push(add_req("r0-vec-a", (700..700 + 2 * PAGE).collect(), 2));
    hooks1.complete_hook.arm();
    hooks1.execute_hook.release();

    hooks1.complete_hook.wait_entered(Duration::from_secs(5));
    thread::sleep(Duration::from_millis(25));

    // Rank 0 has now called `next_execution_plan` for `"r0-vec-a"` and is blocked in
    // `plan_and_dispatch`'s `dp_round(..., false)`: `sched0` MUST remain LOCKED!
    assert!(
        sched0.try_lock().is_none(),
        "Vector A: self.scheduler MUST stay locked inside plan_and_dispatch's dp_round"
    );
    hooks1.complete_hook.release();
    fe1.wait_reply_count(1, Duration::from_secs(5));
    fe1.push(EngineRequest::Control(Box::new(
        ControlRequest::ResumeScheduler(ResumeSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
        }),
    )));
    fe1.wait_reply_count(2, Duration::from_secs(5));
    fe0.wait_finished("r0-vec-a", Duration::from_secs(5));
    fe1.wait_finished("r1-inflight", Duration::from_secs(5));

    // 5. VECTOR B (`commit_down_to` with `in_flight_depth > 0` overlap):
    //    Submit `"r0-overlap"` (`max_new_tokens = 4`) on Rank 0 with `in_flight_depth = 2`
    //    and arm `hooks0.complete_hook`: when `complete()` fires, 1 newer overlapping
    //    batch is STILL in `self.in_flight`! `sched0` MUST stay LOCKED!
    hooks0.complete_hook.arm();
    fe0.push(add_req("r0-overlap", (800..800 + 2 * PAGE).collect(), 4));
    hooks0.complete_hook.wait_entered(Duration::from_secs(5));
    assert!(
        sched0.try_lock().is_none(),
        "Vector B: self.scheduler MUST stay locked in commit_down_to while an overlapping batch is in flight"
    );
    hooks0.complete_hook.release();
    fe0.wait_finished("r0-overlap", Duration::from_secs(5));

    stop.store(true, Ordering::Relaxed);
    t0.join().unwrap();
    t1.join().unwrap();
}

#[test]
fn sync_peer_holder_changes_preserves_hard_pins_across_finish_drain_on_tp_replica() {
    interpreter();
    // 4-rank mesh (`2 DP replicas x 2 TP ranks`):
    // - Replica 0 = {rank 0 (TP0, runs PeerListener/servicer), rank 1 (TP1 mirror)}
    // - Replica 1 = {rank 2 (TP0), rank 3 (TP1)}
    let arena = Arena::new(200_000);
    let listeners = [(); 4].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addresses: Vec<String> = listeners
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.port().unwrap()))
        .collect();

    let peer_listener = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_name = format!("127.0.0.1:{}", peer_listener.port());

    let scheds: Vec<Arc<Mutex<Scheduler>>> = (0..4)
        .map(|r| {
            let name = if r < 2 {
                holder_name.as_str()
            } else {
                "dp1-unused"
            };
            Arc::new(Mutex::new(peer_scheduler(name)))
        })
        .collect();

    let peer_r0 = Peer::start(
        peer_settings(&holder_name),
        0,
        vec![rank_geometry(0), rank_geometry(50_000)],
        Some(peer_listener),
        scheds[0].clone(),
        Arc::new(arena.clone()),
        Arc::new(NoPushes),
    )
    .unwrap();
    let peer_r1 = Peer::start(
        peer_settings(&holder_name),
        1,
        vec![rank_geometry(0), rank_geometry(50_000)],
        None,
        scheds[1].clone(),
        Arc::new(arena),
        Arc::new(NoPushes),
    )
    .unwrap();

    let fe_dp0 = Arc::new(ChannelFrontendState::default());
    let fe_dp1 = Arc::new(ChannelFrontendState::default());
    let hooks: Vec<Arc<DeviceHooks>> = (0..4).map(|_| Arc::new(DeviceHooks::default())).collect();
    let stop = Arc::new(AtomicBool::new(false));

    let mut peers: [Option<Peer>; 4] = [Some(peer_r0), Some(peer_r1), None, None];
    let mut threads = Vec::new();
    for (rank, listener) in listeners.into_iter().enumerate() {
        let addresses = addresses.clone();
        let sched = scheds[rank].clone();
        let hook = hooks[rank].clone();
        let peer = peers[rank].take();
        let fe = match rank {
            0 => Some(ChannelFrontend {
                state: fe_dp0.clone(),
            }),
            2 => Some(ChannelFrontend {
                state: fe_dp1.clone(),
            }),
            _ => None,
        };
        let stop = stop.clone();
        threads.push(thread::spawn(move || {
            let mesh = Mesh::connect(rank, listener, &addresses, Duration::from_secs(10)).unwrap();
            let tp_first = (rank / 2) * 2;
            let replica = Group::new(1, vec![tp_first, tp_first + 1], rank).unwrap();
            let dp = Some(Group::new(0, vec![0, 1, 2, 3], rank).unwrap());
            let outputs = OutputProcessor::new(
                None,
                1,
                None,
                Role::Plain,
                FinishLog {
                    enabled: false,
                    request_stats: false,
                    ttft_percentage: 0,
                },
                false,
                None,
            );
            let mut ev = EventLoop::new(
                sched,
                HookedDevice::new(hook),
                fe,
                outputs,
                loop_config(1),
                Ranks {
                    mesh,
                    replica,
                    pp: None,
                    dp,
                    ep: None,
                },
                Subsystems {
                    transfer: None,
                    embeddings: None,
                    metrics: None,
                    kv_events: None,
                    eplb: None,
                    grammars: None,
                    l3_metrics: None,
                    l3: None,
                    disk: None,
                    disk_metrics: None,
                    peer,
                },
            );
            ev.run(&stop).unwrap();
        }));
    }

    // 1. Seed a 3-page prefix on Replica 0 (`rank 0` and `rank 1`).
    let seed_tokens: Vec<i32> = (1..=3 * PAGE).collect();
    fe_dp0.push(add_req("seed-tp", seed_tokens.clone(), 1));
    fe_dp0.wait_finished("seed-tp", Duration::from_secs(5));

    // 2. Peer client performs `Layout` + `Lookup` before drain so Replica 0 holds a soft-pin lease.
    let (incoming, replies) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, incoming).unwrap();
    let client_addr = format!("127.0.0.1:{}", client.port());
    let lease = PeerLeaseView {
        reader: client_addr.clone(),
        id: 777,
    };
    client
        .request(&holder_name, &Request::Layout { reply_to: client_addr })
        .unwrap();
    let _ = replies.recv_timeout(Duration::from_secs(5)).unwrap();
    client
        .request(
            &holder_name,
            &Request::Lookup {
                lease: lease.clone(),
                prompt: make_lookup_prompt(&seed_tokens),
            },
        )
        .unwrap();
    let hit_frame = replies.recv_timeout(Duration::from_secs(5)).unwrap();
    match Inbound::decode(&hit_frame).unwrap() {
        Inbound::Reply(Reply::Answer { answer, .. }) if answer.kind == PeerAnswerKind::Lookup => {
            assert_eq!(answer.num_tokens, 3 * PAGE);
        }
        other => panic!("expected Lookup Answer, got {other:?}"),
    }

    // 3. Vector C race on a 4-rank (`2 DP x 2 TP`) mesh:
    //    Hold Replica 1 in `Frontend::receive` BEFORE `gate()`, and send
    //    `ReleaseMemoryOccupation(["kv_cache"])` directly to Replica 0 so Replica 0
    //    enters `gate()`'s `all_reduce` with `Drain::Release(["kv_cache"])` active and
    //    `self.scheduler` unlocked!
    fe_dp1.receive_hook.arm();
    fe_dp1.receive_hook.wait_entered(Duration::from_secs(5));

    fe_dp0.push(EngineRequest::Control(Box::new(
        ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: Some(vec!["kv_cache".to_owned()]),
        }),
    )));
    thread::sleep(Duration::from_millis(20));

    // Acquire HARD PIN (`Request::Pin`) on Rank 0's servicer while Replica 0 is waiting
    // on Replica 1 in `gate()`'s `all_reduce`, then submit `ReleaseMemoryOccupation` to
    // Replica 1 and release `fe_dp1.receive_hook`!
    client
        .request(&holder_name, &Request::Pin { lease: lease.clone() })
        .unwrap();
    let pin_frame = replies.recv_timeout(Duration::from_secs(5)).unwrap();
    match Inbound::decode(&pin_frame).unwrap() {
        Inbound::Reply(Reply::Answer { answer, .. }) => assert!(answer.held),
        other => panic!("expected Answer(held=true), got {other:?}"),
    }
    fe_dp1.push(EngineRequest::Control(Box::new(
        ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: Some(vec!["kv_cache".to_owned()]),
        }),
    )));
    fe_dp1.receive_hook.release();

    // Replica 1 has no pins so its `ReleaseMemoryOccupation` completes immediately,
    // while `sync_peer_holder_changes()` syncs the hard pin onto BOTH Rank 0 (TP0) AND
    // Rank 1 (TP1) BEFORE `clear_l1_cache()`, withholding `release:kv_cache` on BOTH TP ranks!
    let dp1_replies = fe_dp1.wait_reply_count(1, Duration::from_secs(5));
    assert!(matches!(
        &dp1_replies[0],
        ControlOutput::ReleaseMemoryOccupation(out) if out.success
    ));
    thread::sleep(Duration::from_millis(60));

    assert!(
        fe_dp0.replies.lock().unwrap().is_empty(),
        "Replica 0 ReleaseMemoryOccupation reply must stay pending while hard pin is held; got {:?}",
        *fe_dp0.replies.lock().unwrap()
    );
    assert!(
        hooks[0].device_ops.lock().unwrap().is_empty(),
        "Rank 0 must withhold release_memory(kv_cache) while peer hard pin is active"
    );
    assert!(
        hooks[1].device_ops.lock().unwrap().is_empty(),
        "TP1 mirror Rank 1 must ALSO withhold release_memory(kv_cache) via sync_peer_holder_changes"
    );
    assert!(scheds[0].lock().peer_stats().pinned_blocks > 0);
    assert!(scheds[1].lock().peer_stats().pinned_blocks > 0);

    // 4. Replica 0 is now in `!forward_allowed` (`!memory_resident` because Replica 1
    //    released memory), unlocking `self.scheduler` during `PAUSED_IDLE_SLEEP` right
    //    before `finish_drain()`. Send `Request::Release`: `finish_drain()`'s
    //    `sync_peer_holder_changes()` syncs the release across TP0 + TP1, `clear_l1_cache()`
    //    succeeds, and BOTH TP ranks execute `release:kv_cache`!
    client
        .request(
            &holder_name,
            &Request::Release {
                lease,
                demote: false,
            },
        )
        .unwrap();
    let _ = replies.recv_timeout(Duration::from_secs(5)).unwrap();

    let replies_dp0 = fe_dp0.wait_reply_count(1, Duration::from_secs(5));
    assert!(matches!(
        &replies_dp0[0],
        ControlOutput::ReleaseMemoryOccupation(out) if out.success
    ));
    assert_eq!(
        *hooks[0].device_ops.lock().unwrap(),
        vec!["release:kv_cache".to_owned()]
    );
    assert_eq!(
        *hooks[1].device_ops.lock().unwrap(),
        vec!["release:kv_cache".to_owned()]
    );

    // Resume memory cleanly and shut down all 4 ranks without any TP scheduler drift!
    fe_dp0.push(EngineRequest::Control(Box::new(
        ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: Some(vec!["kv_cache".to_owned()]),
        }),
    )));
    fe_dp1.push(EngineRequest::Control(Box::new(
        ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: Some(vec!["kv_cache".to_owned()]),
        }),
    )));
    fe_dp0.wait_reply_count(2, Duration::from_secs(5));
    fe_dp1.wait_reply_count(2, Duration::from_secs(5));

    stop.store(true, Ordering::Relaxed);
    for t in threads {
        t.join().unwrap();
    }
}
