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

#include "scheduler/scheduler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "cache/prefix/prefix_hasher.h"

namespace tokenspeed {

StorageStats Scheduler::GetStorageStats() const {
    StorageStats stats;
    if (!storage_transfers_) return stats;
    stats.operations = storage_transfers_->NumOperations();
    stats.held_blocks = storage_transfers_->NumHeldBlocks();
    stats.quarantined = storage_transfers_->NumQuarantined();
    stats.negative_keys = storage_transfers_->NumNegativeKeys();
    stats.backup_candidates = storage_transfers_->NumBackupCandidates();
    stats.wait_timeouts = storage_wait_timeouts_;
    stats.ready_expirations = storage_ready_expirations_;
    stats.pressure_gate_closed = storage_transfers_->PressureGateClosed();
    stats.pressure_bypasses = storage_pressure_bypasses_;
    stats.pressure_dropped_backup_keys = storage_transfers_->NumPressureDroppedBackups();
    stats.overflow_dropped_backup_keys = storage_transfers_->NumOverflowDroppedBackups();
    stats.unavailable_dropped_backup_keys = storage_transfers_->NumUnavailableDroppedBackups();
    stats.disabled_dropped_backup_keys = storage_transfers_->NumDisabledDroppedBackups();
    stats.cleanup_dropped_backup_keys = storage_transfers_->NumCleanupDroppedBackups();
    for (const auto& request : requests_) {
        const auto phase = request->StoragePreparationRef().phase;
        stats.waiting_requests += phase == Request::StoragePreparation::Phase::kWaiting;
        stats.ready_requests += phase == Request::StoragePreparation::Phase::kReady;
    }
    return stats;
}

void Scheduler::expireStorage(Request& request) {
    const auto phase = request.StoragePreparationRef().phase;
    if (phase == Request::StoragePreparation::Phase::kWaiting) ++storage_wait_timeouts_;
    if (phase == Request::StoragePreparation::Phase::kReady) ++storage_ready_expirations_;
    releaseStorage(request);
}

std::int32_t Scheduler::cacheablePrefixTokens(const Request& request) const {
    const auto replay = std::max(config_.prefix_replay_tokens, 1);
    return std::max(std::min(request.PrefillSize() - replay, request.MaxCachedPrefixTokens()), 0);
}

void Scheduler::handleEvent(const cache::StorageTick& event) {
    if (event.now_us < storage_now_us_) throw std::invalid_argument("L3 control clock moved backwards");
    if (storage_transfers_) storage_transfers_->UpdatePressure(event.pressure);
    storage_now_us_ = event.now_us;
    storage_allowed_ = event.allow_storage;
    forward_allowed_ = event.allow_forward;
    for (const auto& request : requests_) {
        auto& state = request->StoragePreparationRef();
        if ((state.phase == Request::StoragePreparation::Phase::kWaiting ||
             state.phase == Request::StoragePreparation::Phase::kReady) &&
            storage_now_us_ >= state.deadline_us) {
            expireStorage(*request);
        }
    }
}

void Scheduler::handleEvent(const cache::StorageDone& event) {
    if (!storage_transfers_) throw std::invalid_argument("storage completion without configured L3");
    storage_transfers_->Complete(event);
}

void Scheduler::handleEvent(const cache::StorageDetach& event) {
    if (auto* request = findRequest(event.request_id)) releaseStorage(*request);
}

void Scheduler::releaseStorage(Request& request) {
    auto& state = request.StoragePreparationRef();
    if (state.operation_id && storage_transfers_) storage_transfers_->Detach(state.operation_id);
    state.operation_id = 0;
    state.hashes.clear();
    state.phase = Request::StoragePreparation::Phase::kBypass;
}

void Scheduler::prepareStorage(Request& request) {
    using Phase = Request::StoragePreparation::Phase;
    if (!storage_transfers_) return;
    auto& state = request.StoragePreparationRef();
    // Both pre-admission states hold no Device resources. Storage preparation
    // may overlap bootstrap; the normal admission still waits for its ACK.
    if (!request.IsAnyOf<fsm::Bootstrapping, fsm::Submitted>()) {
        releaseStorage(request);
        return;
    }
    if (state.phase == Phase::kBypass) return;
    if (state.phase == Phase::kReady) {
        if (storage_now_us_ >= state.deadline_us) expireStorage(request);
        return;
    }
    const auto& limits = storage_transfers_->Config();
    const auto grain = config_.prefix_granularity;
    const auto localBound = [this](const std::vector<std::string>& hashes) {
        const auto probe = coordinator_.ProbePrefix(hashes);
        return std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    };
    if (state.phase == Phase::kWaiting) {
        if (storage_transfers_->Completed(state.operation_id) && !storage_transfers_->IsQuery(state.operation_id)) {
            state.phase = Phase::kReady;
            state.deadline_us = storage_now_us_ + std::min(limits.ready_timeout_us,
                                                           std::numeric_limits<std::int64_t>::max() - storage_now_us_);
            return;
        }
        // This is a local, bounded check over already hashed keys. Another
        // request's D2H/GET may have fulfilled this dependency in the meantime.
        if (storage_now_us_ >= state.deadline_us) {
            expireStorage(request);
            return;
        }
        if (localBound(state.hashes) >= static_cast<std::int32_t>(state.hashes.size()) * grain) {
            releaseStorage(request);
            return;
        }
        // Do not wait for congestion to clear before starting a GET. Existing
        // GETs keep their original deadline and Host references until retirement.
        if (storage_transfers_->IsQuery(state.operation_id) && storage_transfers_->PressureGateClosed()) {
            ++storage_pressure_bypasses_;
            releaseStorage(request);
            return;
        }
        if (!storage_transfers_->Completed(state.operation_id)) return;
        if (storage_transfers_->IsQuery(state.operation_id)) {
            if (!storage_allowed_) return;
            const auto hits = storage_transfers_->QueryHits(state.operation_id);
            storage_transfers_->Detach(state.operation_id);
            state.operation_id = storage_transfers_->StartRead(state.hashes, hits);
            if (!state.operation_id) releaseStorage(request);
        } else {
            state.phase = Phase::kReady;
            state.deadline_us = storage_now_us_ + std::min(limits.ready_timeout_us,
                                                           std::numeric_limits<std::int64_t>::max() - storage_now_us_);
        }
        return;
    }
    if (!storage_allowed_) return;
    // Decode's ordinary remote landing reuses Device history only; local
    // recovery retains the normal group matcher. No new PD handshake here.
    if (config_.role == Role::kD || config_.disable_prefix_cache || !storage_transfers_->CanStart()) {
        if (storage_transfers_->PressureGateClosed()) ++storage_pressure_bypasses_;
        state.phase = Phase::kBypass;
        return;
    }
    auto pages = request.FullPrefixPages(false);
    const auto max_tokens = cacheablePrefixTokens(request) / grain * grain;
    pages.resize(std::min(pages.size(), static_cast<std::size_t>(max_tokens / grain)));
    state.hashes = ComputePrefixHashes(pages, "");
    const auto local = localBound(state.hashes);
    if (max_tokens - local < limits.min_prefetch_tokens) {
        releaseStorage(request);
        return;
    }
    const auto end =
        std::min<std::int64_t>(max_tokens, static_cast<std::int64_t>(local) + limits.max_prefetch_tokens) / grain;
    state.hashes.resize(static_cast<std::size_t>(end));
    auto query = coordinator_.MissingStorageKeys(state.hashes, limits.max_keys_per_operation);
    state.hashes.resize(query.num_prefix_pages);
    auto keys = std::move(query.keys);
    if (static_cast<std::int32_t>(state.hashes.size()) * grain - local < limits.min_prefetch_tokens || keys.empty()) {
        releaseStorage(request);
        return;
    }
    // A known failed immutable object remains unavailable for this attempt.
    std::erase_if(keys, [this](const auto& key) { return storage_transfers_->IsNegative(key); });
    state.operation_id = storage_transfers_->StartQuery(std::move(keys));
    if (!state.operation_id) {
        releaseStorage(request);
        return;
    }
    state.phase = Phase::kWaiting;
    state.deadline_us =
        storage_now_us_ + std::min(limits.wait_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
}
}  // namespace tokenspeed
