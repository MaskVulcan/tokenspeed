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

//! Peer prefix sharing on the host (`docs/rfcs/0004-peer-prefix-sharing.md`):
//! the endpoint and reads of ts-control's `peer` module, started from the
//! bootstrap's `PeerSharingSettings` (`cache/peer.py`), and the outbox the
//! data plane reports each plan's landing fence through.

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use parking_lot::Mutex;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use ts_collective::{Group, Mesh};
use ts_control::{
    disk::DiskTier,
    peer::{
        geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
        landing::Timeouts,
        parse_holder_circuit_breaker_env,
        push::{EngineWriter, PushEndpoint, PushSource},
        read::EngineReader,
        Peer, PeerFence, PeerListener, PeerSettings, ReplicaSettings,
    },
};
use ts_scheduler::Scheduler;

/// Where the data plane tells the host that a plan's landings may be read
/// (`DeviceHandle.execute`).
#[pyclass(frozen, name = "PeerOutbox")]
pub struct Outbox {
    fence: OnceLock<PeerFence>,
}

impl Outbox {
    /// An outbox whose peer starts later.
    pub fn new() -> Self {
        Self {
            fence: OnceLock::new(),
        }
    }
}

#[pymethods]
impl Outbox {
    /// The oldest plan with landings not yet reported ran its device
    /// prerequisites. The forward thread runs this, inside the plan's
    /// execution.
    fn reads_ready(&self) -> PyResult<()> {
        self.fence
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("the host's peer has not started"))?
            .ready();
        Ok(())
    }
}

/// What the bootstrap resolved for this rank.
pub struct Settings {
    peer: PeerSettings,
    node_leaders: Vec<(usize, u32)>,
    bind_host: String,
    port: u16,
    /// The host other replicas reach this rank's push endpoint at.
    push_host: String,
    library: PathBuf,
    local_ip: String,
    protocol: String,
    ib_device: String,
    regions: Vec<(usize, usize)>,
    device: Vec<DeviceField>,
    device_regions: Vec<(u64, u64)>,
    host: HostPool,
}

fn micros(seconds: f64) -> i64 {
    (seconds.max(0.0) * 1e6) as i64
}

impl Settings {
    /// The weight version the cache namespace starts at.
    pub fn weight_version(&self) -> &str {
        &self.peer.weight_version
    }

    /// The settings of `PeerSharingSettings`, or `None` for None.
    pub fn from_python(settings: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if settings.is_none() {
            return Ok(None);
        }
        let geometry = settings.getattr("geometry")?;
        let device = geometry
            .getattr("device")?
            .try_iter()?
            .map(|field| {
                let field = field?;
                Ok(DeviceField {
                    group_id: field.getattr("group")?.extract()?,
                    field_id: field.getattr("field_id")?.extract()?,
                    address: field.getattr("address")?.extract()?,
                    page_stride: field.getattr("page_stride")?.extract()?,
                    bytes: field.getattr("bytes")?.extract()?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let host = geometry.getattr("host")?;
        let host_address: u64 = host.getattr("address")?.extract()?;
        let regions: Vec<(usize, usize)> = geometry.getattr("regions")?.extract()?;
        let mut region_lengths: Vec<u64> = host
            .getattr("region_lengths")
            .and_then(|attr| attr.extract())
            .unwrap_or_default();
        if region_lengths.is_empty() {
            // Backward-compatible fallback: reconstruct contiguous registered
            // Host MR segments starting at `host_address` from `regions`.
            let mut cursor = host_address;
            for &(addr, len) in &regions {
                if addr as u64 == cursor && len > 0 {
                    region_lengths.push(len as u64);
                    cursor = cursor.saturating_add(len as u64);
                }
            }
        }
        let host_total_bytes: u64 = region_lengths.iter().copied().sum();
        let mut device_regions: Vec<(u64, u64)> = geometry
            .getattr("device_regions")
            .and_then(|attr| attr.extract())
            .unwrap_or_default();
        if device_regions.is_empty() {
            let host_end = host_address.saturating_add(host_total_bytes);
            for &(addr, len) in &regions {
                let addr_u64 = addr as u64;
                if len > 0 && (addr_u64 < host_address || addr_u64 >= host_end) {
                    device_regions.push((addr_u64, len as u64));
                }
            }
        }
        let host = HostPool {
            address: host_address,
            groups: host
                .getattr("groups")?
                .try_iter()?
                .map(|group| {
                    let group = group?;
                    Ok(HostGroup {
                        parent_bytes: group.getattr("parent_bytes")?.extract()?,
                        packing: group.getattr("packing")?.extract()?,
                        block_bytes: group.getattr("block_bytes")?.extract()?,
                        field_offsets: group.getattr("field_offsets")?.extract()?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
            region_lengths,
        };
        let protocol: String = settings.getattr("protocol")?.extract()?;
        let (env_threshold, env_cooldown_us) = parse_holder_circuit_breaker_env();
        let holder_fail_threshold: u32 = settings
            .getattr("holder_fail_threshold")
            .and_then(|attr| attr.extract())
            .unwrap_or(env_threshold);
        let holder_cooldown_us: i64 = settings
            .getattr("holder_cooldown_s")
            .and_then(|attr| attr.extract::<f64>())
            .map(micros)
            .unwrap_or(env_cooldown_us);
        Ok(Some(Self {
            peer: PeerSettings {
                name: settings.getattr("name")?.extract()?,
                incarnation: settings.getattr("incarnation")?.extract()?,
                deployment_id: settings.getattr("deployment_id")?.extract()?,
                weight_version: settings.getattr("weight_version")?.extract()?,
                timeouts: Timeouts {
                    answer_us: micros(settings.getattr("answer_timeout_s")?.extract()?),
                    quarantine_us: micros(settings.getattr("quarantine_s")?.extract()?),
                    push_us: micros(settings.getattr("push_timeout_s")?.extract()?),
                },
                record_lifetime: Duration::from_secs_f64(
                    settings
                        .getattr("record_lifetime_s")?
                        .extract::<f64>()?
                        .max(0.0),
                ),
                holder_fail_threshold,
                holder_cooldown_us,
            },
            node_leaders: settings.getattr("node_leaders")?.extract()?,
            bind_host: settings.getattr("bind_host")?.extract()?,
            port: settings.getattr("port")?.extract()?,
            push_host: settings.getattr("push_host")?.extract()?,
            library: PathBuf::from(settings.getattr("library")?.extract::<String>()?),
            local_ip: settings.getattr("local_ip")?.extract()?,
            protocol,
            ib_device: settings.getattr("ib_device")?.extract()?,
            regions,
            device,
            device_regions,
            host,
        }))
    }
}

/// Start this rank's peer sharing over `scheduler`: its transfer engine with
/// the cache registered, its push endpoint, serving `disk`'s pages when the
/// rank has a Disk tier, the replica's geometries gathered over `mesh` (every
/// member of `replica` calls this), and `outbox` connected to the reads.
pub fn start(
    mut settings: Settings,
    mesh: &mut Mesh,
    replica: &Group,
    scheduler: Arc<Mutex<Scheduler>>,
    outbox: &Outbox,
    disk: Option<&DiskTier>,
) -> Result<Peer, String> {
    let library =
        ts_mooncake::Library::load(&settings.library).map_err(|error| error.to_string())?;
    if !library.reads() {
        return Err(format!(
            "{} has no READ entry points: peer prefix sharing needs a \
             tokenspeed-mooncake build with ts_mooncake_read",
            settings.library.display()
        ));
    }
    let engine = ts_mooncake::Engine::start(
        library,
        &settings.local_ip,
        &settings.protocol,
        &settings.ib_device,
    )
    .map_err(|error| error.to_string())?;
    for &(address, length) in &settings.regions {
        engine
            .register(address, length)
            .map_err(|error| error.to_string())?;
    }
    let engine = Arc::new(engine);
    // Every rank answers its own pushes here; one with a Disk tier also
    // pushes its Disk pages from the staging ring, registered for the writes.
    let push = PushEndpoint::bind(
        &settings.bind_host,
        0,
        &settings.push_host,
        settings.peer.incarnation,
    )
    .map_err(|error| error.to_string())?;
    if let Some((executor, disk_layout, ring, slot_bytes)) = disk.and_then(DiskTier::push_source) {
        engine
            .register(ring.address(), ring.size())
            .map_err(|error| error.to_string())?;
        push.serve(PushSource {
            executor,
            disk_layout,
            ring,
            slot_bytes,
            writer: Arc::new(EngineWriter(engine.clone())),
        })
        .map_err(|error| error.to_string())?;
    }
    let local = RankGeometry {
        session: engine.session_id().to_owned(),
        device: settings.device,
        device_regions: settings.device_regions,
        host: Some(settings.host),
        push: Some(push.name().to_owned()),
    };
    let gathered = mesh
        .all_gather(replica, &local.encode())
        .map_err(|error| error.to_string())?;
    let geometries = gathered
        .iter()
        .map(|bytes| RankGeometry::decode(bytes))
        .collect::<Option<Vec<_>>>()
        .ok_or("a replica member's peer geometry does not decode")?;
    if replica.size() > 1 {
        let local_replica_settings = ReplicaSettings {
            quarantine_us: settings.peer.timeouts.quarantine_us,
            holder_fail_threshold: settings.peer.holder_fail_threshold,
            holder_cooldown_us: settings.peer.holder_cooldown_us,
        };
        let payload = if replica.index() == 0 {
            local_replica_settings.encode()
        } else {
            Vec::new()
        };
        let shared = mesh
            .broadcast(replica, 0, payload)
            .map_err(|error| error.to_string())?;
        let agreed = ReplicaSettings::decode(&shared)
            .ok_or("replica peer settings broadcast did not decode")?;
        settings.peer.timeouts.quarantine_us = agreed.quarantine_us;
        settings.peer.holder_fail_threshold = agreed.holder_fail_threshold;
        settings.peer.holder_cooldown_us = agreed.holder_cooldown_us;
    }
    tracing::debug!(
        regions = ?settings
            .regions
            .iter()
            .map(|(address, length)| format!("{address:#x}+{length:#x}"))
            .collect::<Vec<_>>(),
        fields = ?local
            .device
            .iter()
            .map(|field| format!("{}@{:#x}/{:#x}", field.field_id, field.address, field.page_stride))
            .collect::<Vec<_>>(),
        "peer geometry"
    );
    // Replicas read only holders of the same namespace; a mismatch answers
    // as a miss, so the namespace is worth seeing in the log.
    tracing::info!(
        name = settings.peer.name,
        session = engine.session_id(),
        incarnation = settings.peer.incarnation,
        deployment_id = settings.peer.deployment_id,
        weight_version = settings.peer.weight_version,
        "peer prefix sharing up"
    );
    let listener = if replica.index() == 0 {
        Some(
            PeerListener::bind_node(
                mesh,
                super::PEER_NODE_GROUP,
                &settings.node_leaders,
                &settings.bind_host,
                settings.port,
            )
            .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let peer = Peer::start(
        settings.peer,
        replica.index(),
        geometries,
        listener,
        scheduler,
        Arc::new(EngineReader(engine)),
        Arc::new(push),
    )
    .map_err(|error| error.to_string())?;
    // A second start would be a second peer; there is one per process.
    if outbox.fence.set(peer.fence()).is_err() {
        return Err("the host's peer started twice".to_owned());
    }
    Ok(peer)
}
