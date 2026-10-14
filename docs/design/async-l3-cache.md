# Asynchronous global L3 cache

L3 extends the existing Device → Host cache hierarchy with shared Mooncake
objects. The scheduler prepares Host data before ordinary Device admission.
All requests still use the same matcher, admission planner, H2D operations and
forward path. Selecting `--kvstore-storage-backend mooncake` selects this
asynchronous implementation; there is no second synchronous scheduling mode.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| C++ scheduler and `StorageTransferManager` | Request preparation, prefix boundaries, pressure admission, soft deadlines, operation IDs, credits, Host references, publication, cancellation and quarantine |
| `CacheCoordinator` | Group matching, allocation and pinning; ordinary probes see published Device/Host data only |
| Python `AsyncL3Executor` | Host job snapshots, namespace and backend lifetime, bounded query/read/write queues, completion/pressure facts, and private backend health monitoring |
| `L3CacheHooks` | Replica agreement on time, permissions and completion facts; feedback to the existing event-loop head |
| `HostCacheStorage` | Shared pinned Host allocation and CPU-only block geometry; holds no device buffers or transfer layout |
| `DeviceHandle` / `HostCacheExecutor` | Ordered D2H/H2D and device completion fences |

Python never selects a reusable prefix or allocates scheduler blocks. Workers
never call the scheduler, GPU APIs or collectives. Host references remain in
C++ until every cache-owning rank proves retirement. L2 and L3 independently
retain one Host allocation. L3 owns its backend and accepts only Host storage
and immutable namespace facts; it has no reference to L2 or DeviceHandle.

The construction boundary returns separate device and L3 capabilities. The
loop submits `plan.storage` to L3 and device work to DeviceHandle from the same
plan, including frozen-round cleanup. L3 hooks poll the L3 executor directly
and return agreed facts to the existing head feedback point. RequestHandler
coordinates namespace/version changes through that same L3 capability.
No additional HiCacheEngine is needed: C++ owns policy, the existing loop owns
round sequencing, and the shared Host resource owns its allocation.

## Read transaction

1. The scheduler probes local Device and Host caches. A local hit covering
   the maximum reusable boundary, or an extension below the configured
   minimum, bypasses L3. Otherwise it considers a bounded extension of that
   prefix, respecting replay/logprob limits and group geometry. It expands
   whole prefix boundaries once until the query key budget is exhausted.
2. `plan.storage` carries a query with a monotonically increasing operation
   ID and immutable keys. Querying reserves no Device pages or batch slots.
3. After all owners report existence, C++ runs the existing group matcher
   with these temporary hints. It pins all warm dependencies before allocating
   Host destinations. Under shortage it shortens the boundary and re-runs the
   matcher, including sliding-window and snapshot-state requirements.
4. A GET job receives fixed backend keys, Host offsets and lengths. No worker
   reads a mutable namespace or a later scheduler plan.
5. `StorageDone` carries per-key success, confirmed object-failure facts,
   and an operation-wide retirement fact. Only all-owner, full-length, retired successes publish Host entries.
   Partial success may populate a shorter legal prefix; ordinary matching
   determines what is actually reusable.
6. A short Ready hold protects the result until ordinary admission transfers
   ownership to the existing H2D ticket. Ready expiry releases the hold. H2D
   and the forward use the same execution path as every Host hit.

Retraction restores exact images through `HostCacheExecutor`; it never starts
a fresh L3 preparation. State groups use the current fixed two-slot checkpoint
contract. Runtime compatibility epoch 2 isolates objects from the prior layout.

The request has one soft deadline shared by query and GET. A soft timeout
ends its wait and permits local admission; it does not retract the request,
raise admission headroom, mark the backend unhealthy or release native I/O
buffers. Detached completions cannot publish data. Another request's local
publication may also satisfy the wait before the remote operation completes.

Client cancellation enters at the existing loop head. A fused request still
in Submitted can abort immediately when C++ confirms it has no
forward resources. PD requests detach L3 but retain their existing transport
completion fence and output lifecycle. This removes the extra L3 delay; it
does not make PD cancellation immediate or bound it to one forward chunk.

## Write transaction

A D2H completion publishes Host L2 and releases its Device source immediately
after replica agreement. It never waits for Mooncake. C++ queues bounded,
nonowning backup candidates. At PUT dispatch it rechecks Host residency and
acquires source references; evicted candidates are skipped. Queued candidates
therefore cannot pin Host memory indefinitely.

PUT is create-only. A retired failure leaves the local Host entry usable.
An in-flight PUT retains its source references independently of the request,
and source ownership ends only after all owners report retirement. Local Host
publication does not imply that a shared object exists. Different compatible
instances use the same namespace and per-TP-shard key format; DP replica IDs
are deliberately absent from object keys. The Mooncake master owns metadata,
not the cached bytes. Independently mounted data segments can outlive inference
workers: setting `global_segment_size=0` explicitly makes an inference worker
a client without mounting storage of its own.

## Controller queues, monitor and bounds

Before allocating the shared Host arena, the device construction owner gathers
each rank's future Host bytes over the existing world CPU group. Accounting
includes L2, the retraction snapshot pool, the slot-state arena, a locally
mounted Mooncake segment, and native staging, each once. It sums ranks on each physical node and in each visible common cgroup
(including limited ancestors), using the smallest sampled availability and
reserving headroom once per domain. Resident model/state memory is already
charged in the sampled usage. A sampling failure is agreed before any rank
allocates. L2 receives only an allocation-check callback; it has no knowledge
of the storage backend. This collective runs during construction, never in the
serving loop.

This is a startup check, not a node-wide reservation service or a cap on process
RSS. Independent P/D worlds and separately launched replicas need a combined
deployment preflight before concurrent startup. That preflight must also count
independent store owners, SSD restore buffers and other planned allocations on
the same node. It cannot discover hidden cgroup ancestors or prevent a competing
job from allocating after the sample. Queue/Host credits continue to bound L3
work at runtime; process and store memory must also be monitored under load.

Query, GET and PUT have separate bounded queues and workers. The total
outstanding-operation limit includes queued jobs, native work and undelivered
results. C++ separately limits key count, Host references, lookahead tokens,
negative metadata and backup candidates. Each held CacheBlock is charged
conservatively as one potentially pinned LCM parent, including warm
requirements and quarantine. This can underuse packed Host capacity but cannot
oversubscribe it.

Backups may occupy at most half of the operation credits and half of the Host
credits, rounded down. Queries and reads may use the entire shared budget.
These are admission limits in the C++ cache owner, not separate allocators or
worker-side decisions. They keep slow PUTs from exhausting the resources a
new prefetch needs; they do not preempt native I/O or guarantee a large read
fits. A one-credit budget admits reads but no backups. Backup candidates stay
bounded and nonowning until admission, so write pressure cannot pin queued
data indefinitely.

Temporary credit exhaustion leaves candidates queued until credits return;
it does not discard a still-resident prefix. Dispatch rechecks Host residency.
A permanently zero write budget rejects candidates immediately. The bounded
queue drops its oldest candidate on overflow; pressure closure and frozen-round
cleanup still discard queued work. Scheduler counters distinguish pressure,
overflow, unavailable Host data, disabled writes and cleanup, exported through
`tokenspeed:l3_dropped_backup_keys` with a fixed `reason` label. These count
candidate occurrences, not distinct objects. Coalescing a candidate already
covered by a pending PUT is not a loss. The existing pressure-only counter
retains its meaning. None of these observations change retirement or read reserves.

The executor's completion poll also returns a small pressure snapshot under the
same lock: the number and payload bytes of queued QUERY/GET/PUT jobs, and the
age of the oldest queued or running job. Query payload bytes are zero. Retired
work is excluded; detached native work remains included until it retires.
Sampling scans at most the configured operation limit and never calls storage
or the GPU. Metrics snapshots remain a separate, lower-frequency observation.

Hooks MAX-reduce these durations and counts across cache owners through the
existing fixed CPU header. C++ closes a shared read/write admission gate when
any high watermark is reached and reopens it only when all observations reach
their low watermarks. Local wall-clock timestamps never cross this boundary.
The gate is independent of pause/freeze, backend health and quarantine. It does
not pause forward execution, completion handling or soft deadlines.

Closed admission bypasses a new request's L3 preparation. A request still in
QUERY detaches and uses ordinary local admission without waiting for its soft
deadline; no new GET may acquire destinations. GETs already submitted keep
their original lifetime and deadline, and Ready results remain usable. Host
credit exhaustion also rejects a new QUERY before backend work. Closing the
gate drops nonowning backup candidates and rejects new ones before acquiring
Host references. Already submitted PUTs continue; L2 publication is unaffected.
Recovery does not replay dropped candidates. Native retirement is never inferred
from pressure recovery, and an empty runtime queue cannot clear quarantine.

The L2 executor binds immutable transfer geometry and mapped-buffer address
tables for every write lane and load workspace during construction. Their
initial uploads must not occur on the first serving H2D/D2H submission.
Per-operation block tables still use pinned, asynchronous uploads and their
existing lifetime fences.

Queued cancellation is acknowledged without waiting for an occupied worker.
A cancelled queue item becomes inert. Started I/O has a separate lifetime:
request cancellation is not native retirement. Operation IDs never reuse a
request slot or an earlier operation's ID.

The executor's private health monitor rejects work during cooldown and permits
one recovery probe. It has no separate thread or public control interface.
Health epochs prevent a stale successful call from undoing a newer outage.
Existence misses and safely retired GET misses do not indicate an outage;
exceptions, PUT failures and unknown native failures do. A soft timeout is
only a scheduling decision. The first version uses fixed deadlines and
credits and explicit pressure watermarks; it does not estimate per-request
transfer benefit or auto-tune limits. Pressure does not report a backend fault,
and a successful backend call does not override C++ pressure admission.

## Completion agreement and ordering

Every attention-TP owner sends its newly completed records once over the CPU
group. Hooks retain partial-owner records until the other owners finish.
Each round first exchanges a fixed CPU header containing the clock, permissions
presence of completion/error facts, and the three pressure observations. Empty
and pressure-only rounds need no object serialization;
nonempty rounds exchange the bounded facts after every owner sees that header.
The agreed tick is still delivered every round, including idle and frozen rounds.
Results are agreed per operation, so a slow earlier operation cannot block a
later complete one. Per-key validity and retirement are AND-reduced;
confirmed missing/invalid object facts are OR-reduced. A single missing shard
invalidates the shared object, but a peer that did not read contributes no
object-failure evidence. The
window is bounded by operation and key credits; it is not truncated to a
prefix of pending operations. A local polling error joins the collective
before every owner raises the protocol error.
Local completion records are validated and captured inside that same error
boundary, before header exchange. Success, object-failure and retirement
fields must be actual booleans; per-key masks have equal lengths and success
cannot also report an object failure. Query/PUT cannot report read failures.
A truthy value is not proof of native retirement. Malformed backend
results follow the backend's explicit exception-retirement contract.

The TP leader supplies the common monotonic clock; permissions are intersected.
The head advance applies that tick and cancellation before completion records,
so an already expired/detached read cannot publish a late result. The scheduler
emits storage commands beside cache and forward operations. Storage dispatch
bypasses the GPU FIFO; ordinary D2H → zeroing → H2D → forward ordering is
unchanged. Frozen rounds still service the same planner with forward/storage
permissions disabled, allowing completions, expiration and cancellation.

## Retirement and maintenance

For the Mooncake adapter, exact byte counts prove GET success. Nonnegative
short reads are retired failures. `OBJECT_NOT_FOUND` (`-704`) is a metadata
miss before destination transfer submission. PUT `OBJECT_ALREADY_EXISTS`
(`-705`) is a create-only rejection before source transfer and is idempotent
success. Other negative transfer errors, exceptions after native submission,
and malformed transfer results do not prove retirement.

Only confirmed missing or invalid GET objects enter C++'s bounded negative
metadata. Health/queue/shutdown rejection, queued cancellation and backend
exceptions carry no object evidence. They still complete the request's attempt
and preserve the retirement contract, but the same keys can be queried and read
by a new request after recovery. Genuine negative metadata retains its existing
bounded FIFO lifetime; expiry/reprobe after external object repair is separate
work. Python reports facts and does not choose negative-cache policy.

An unretired result quarantines every reference of that job. It is never
published, freed or repaired by an existence recheck. The first version
stops new L3 work for the replica after quarantine and requires process
restart to reclaim that storage safely. Ordinary computation and valid L2
hits can continue within the remaining capacity. A deployment SDK must honor
the adapter's return-code contract; real transport fault tests remain part of
hardware validation.

A full cache clear, namespace rotation or weight update must reject outstanding
storage operations, including Ready holds and quarantine; it must not
synchronously drain QUERY/GET/PUT on the scheduling thread. After replica-agreed
preflight, remote namespace deletion retains the existing synchronous
administrative call on the control thread. This purge can delay request and
control processing until the backend returns; preparation soft timeouts do not
bound or cancel it. A deletion failure prevents local cache clearing.
Asynchronous administrative purge is outside the first-version scope.
L1-only clear may proceed while Host-only storage I/O remains active.
Shutdown drains device copies through DeviceHandle before closing L3 from the
control plane. L3 joins its workers with a deadline and then closes its backend;
if either device drain fails or native retirement remains unknown it refuses backend close/Host deregistration
and retains the store and Host allocation even if the runtime owner is collected.

## Initial scope and validation

### Observability

With Prometheus metrics enabled, L3 exports rank-local query/GET/PUT counters,
queue/backend cumulative durations, queued/running/completed job counts and
their byte ranges. Completed jobs remain charged until their facts are polled.
Separate scheduler gauges retain visibility of Ready holds and quarantined
references after runtime completion. Waiting soft timeouts and Ready expiration
are different counters; client detachment increments neither.
Pressure bypasses and dropped backup candidate keys have separate counters;
the admission gate is a mirrored scheduler gauge. These do not count unique
remote objects or saved prefill tokens.

Health state/epoch and unproven native retirement are separate gauges. Reading
health never consumes a recovery probe. Snapshots carry no request IDs or object
keys. Labels identify model, engine role, DP rank and TP rank: I/O is shard-local,
while scheduler resource/deadline values are mirrored and must not be summed
across TP owners. Successful adapter byte counters include idempotent PUT hits
and detached GET results; they measure neither network traffic nor published
cache hits. Metrics do not participate in scheduler decisions or agreement.
The `backend_calls` event counts adapter invocations; a create-only PUT can
finish after its existence check without calling the native PUT API.

### Supported scope

The first version supports PP=1 and the existing Host-KVStore model/cache-group
capabilities. DCP remains excluded by Host KVStore. Fused and PD-prefill roles
can prepare L3 hits. Ordinary PD-decode remote landing keeps its Device-only
prefix contract; local recovery uses the existing matcher. There is no new PD
handshake, direct L3-to-Device DMA, shared read deduplication or layer-wise GET
publication in this version.

Scheduler tests exercise mixed block granularities, sliding windows, sparse
state, partial results, deadline expiry, cancellation, ID reuse and ownership.
Executor tests cover bounded queues, health epochs, queued cancellation,
shared-store byte round trips and quarantine. Two-process Gloo tests cover
out-of-order owner agreement and collective error handling. GPU H2D/D2H and
real Mooncake/RDMA validation are distinct from these CPU protocol tests.

`test/runtime/test_l3_flat_kv_e2e.py` checks packed history/state groups through
D2H, asynchronous PUT/GET and H2D, including untouched GPU bytes. Its native
case requires `MOONCAKE_GPU_TEST_CONFIG`, a JSON backend configuration with
`global_segment_size=0`, and an already running master plus an independent
storage owner. It exercises both `direct` and `kernel` I/O settings. Each writer
process exits before a new reader allocates Host/GPU buffers; the reader restores
the objects into different block slots and checks exact bytes, then deletes
them and verifies safely retired misses. Without that environment variable,
the native case skips and only the memory-backend case runs. This transfer test
complements serving tests that verify scheduler admission and model output.

Validation counters must distinguish successful adapter PUT results from new
native writes: create-only PUT can succeed because an object already exists.
Native PUT success can also mean that the SDK skipped an existing object;
successful submitted payload is not proof of new writes or network traffic.
Use SDK transport counters when reporting wire bytes. Model
reuse evidence should include fresh inference processes, successful GETs on
every owning rank, and answers requiring information in the restored prefix.

The primary latency measure for L3 prefix reuse is end-to-end time to first
token (TTFT), compared with recomputation when the tested prefix is absent
from both Device and Host caches. Report the prefix actually used by admission,
remaining prefill work and fallback frequency alongside TTFT. Separate remote
DRAM hits, SSD hits with warm or cold file caches, and misses. Concurrent decode
time per output token (TPOT) is an interference check; reducing idle L3 hook
overhead does not establish a prefill benefit. A storage call overlapping GPU
work also does not establish how much latency was hidden on the requesting
request's critical path. PD measurements must include transfer and decode-side
admission through the first delivered token.

The opt-in helpers in `test/benchmarks/l3/` measure complete SSE responses,
native SDK bytes and scheduler ownership. A benchmark must explicitly install
the observer in an isolated deployment; production code never imports it.
Idle-only local resets distinguish Host hits from remote restores without
purging the shared namespace. Test cleanup removes only successful PUT keys
inside the explicitly isolated benchmark namespace, and waits for store-side
offload to settle; it must not be installed on a namespace owned by other jobs.
These resets and cleanup calls stay outside timed request cohorts. Formal
comparisons use equal Device capacity, warm execution paths, the default L3
wait budget, and cache-off/L2-only/L3 runs; partial hits, misses and fallback
remain in the results. Profiling and artificial I/O delays are separate runs.
Check the engine's reported single-request limit, including both PD roles,
against rendered input and generation tokens. Nominal pool token capacity
does not imply that a hybrid request of that length fits every cache group;
context-budget truncation invalidates an answer-comparison workload.
SSE chunk intervals are not token ITL, and small-cohort empirical p99 values
do not establish a production tail guarantee.

See [server configuration](../configuration/server.md#host-l2-and-mooncake-store-l3)
for limits and backend settings, [cache concepts](cache-concepts.md) for
geometry and key compatibility, and [event loop](event-loop.md) for the two
centralized feedback points.
