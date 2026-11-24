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

//! A holder's answers (`docs/rfcs/0004-peer-prefix-sharing.md`): each request
//! is answered at once with reads of the cache as it stands, between the
//! control thread's rounds, and what it changes is queued for the next head,
//! where every member of the replica applies it before any other event.

use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use ts_scheduler::{
    PeerAnswerKind, PeerAnswerView, PeerEventKind, PeerEventView, PeerLeaseView, PeerPageView,
    Scheduler,
};

use crate::peer::{
    geometry::RankGeometry,
    wire::{Reply, Request, Stamp},
};

/// The reads a holder answers with (`Scheduler`'s peer reads).
pub trait HolderCache {
    /// A lookup's answer, bounded by `pending_new_leases` unsynced new lookup hits.
    fn answer_lookup(&self, lookup: &PeerEventView, pending_new_leases: usize) -> PeerAnswerView;
    /// Whether the lookup's prompt, up to `num_tokens`, still resolves to
    /// exactly `pages`, and every page is ready for an unordered remote read.
    fn pages_match(&self, lookup: &PeerEventView, num_tokens: i32, pages: &[PeerPageView]) -> bool;
    /// Whether `lease` still holds an active soft pin in `PeerExports`.
    fn lease_can_pin(&self, lease: &PeerLeaseView) -> bool;
    /// Whether the lease's pin holds its pages now.
    fn lease_held(&self, lease: &PeerLeaseView) -> bool;
    /// Plans built (and cache clears run) so far.
    fn plan_calls(&self) -> i64;
}

impl HolderCache for Scheduler {
    fn answer_lookup(&self, lookup: &PeerEventView, pending_new_leases: usize) -> PeerAnswerView {
        self.answer_peer_lookup_bounded(lookup, pending_new_leases)
    }

    fn pages_match(&self, lookup: &PeerEventView, num_tokens: i32, pages: &[PeerPageView]) -> bool {
        self.peer_pages_match(lookup, num_tokens, pages)
    }

    fn lease_can_pin(&self, lease: &PeerLeaseView) -> bool {
        self.peer_lease_can_pin(lease)
    }

    fn lease_held(&self, lease: &PeerLeaseView) -> bool {
        self.peer_lease_held(lease)
    }

    fn plan_calls(&self) -> i64 {
        self.plan_calls()
    }
}

/// What a lease's lookup asked and was answered, and when its pin was.
struct Record {
    lookup: PeerEventView,
    answer: PeerAnswerView,
    /// Whether the lookup event has been drained for synchronization into `PeerExports`.
    lookup_synced: bool,
    /// The holder's plan count when the pin was answered.
    pinned_at: Option<i64>,
    expires: Instant,
}

/// The holder's half: answers, the leases' records and the queued changes.
pub struct Holder {
    stamp: Stamp,
    ranks: Vec<RankGeometry>,
    records: HashMap<PeerLeaseView, Record>,
    /// Unsynced lookup hits that will allocate a new soft lease in `PeerExports`.
    unsynced_new_leases: HashSet<PeerLeaseView>,
    /// How long a record outlives its last request: longer than any pin the
    /// scheduler holds, so a release always finds its pin's record.
    record_lifetime: Duration,
    changes: Vec<PeerEventView>,
}

impl Holder {
    /// A holder answering as `stamp`, with its replica's geometries.
    pub fn new(stamp: Stamp, ranks: Vec<RankGeometry>, record_lifetime: Duration) -> Self {
        Self {
            stamp,
            ranks,
            records: HashMap::new(),
            unsynced_new_leases: HashSet::new(),
            record_lifetime,
            changes: Vec::new(),
        }
    }

    /// Answer as the namespace `identity` from now on.
    pub fn set_identity(&mut self, identity: String) {
        self.stamp.identity = identity;
    }

    /// Answer a `Layout` query without touching the scheduler.
    pub fn layout_reply(&self) -> Reply {
        Reply::Layout {
            stamp: self.stamp.clone(),
            ranks: self.ranks.clone(),
        }
    }

    /// The answer to `request`, and the endpoint it goes to.
    pub fn handle(
        &mut self,
        cache: &impl HolderCache,
        request: Request,
        now: Instant,
    ) -> (String, Reply) {
        self.records.retain(|_, record| now < record.expires);
        match request {
            Request::Layout { reply_to } => (reply_to, self.layout_reply()),
            Request::Lookup { lease, prompt } => {
                let lookup = prompt.lookup_event(&lease);
                let pending_new_leases = self.unsynced_new_leases.len()
                    - usize::from(self.unsynced_new_leases.contains(&lease));
                let answer = cache.answer_lookup(&lookup, pending_new_leases);
                if answer.num_tokens > 0
                    && !cache.lease_can_pin(&lease)
                    && !cache.lease_held(&lease)
                {
                    self.unsynced_new_leases.insert(lease.clone());
                }
                if answer.num_tokens > 0 || cache.lease_can_pin(&lease) {
                    self.changes.push(lookup.clone());
                }
                self.records.insert(
                    lease.clone(),
                    Record {
                        lookup,
                        answer: answer.clone(),
                        lookup_synced: false,
                        pinned_at: None,
                        expires: now + self.record_lifetime,
                    },
                );
                self.reply(lease, answer)
            }
            Request::Pin { lease } => {
                let mut answer = self.answer(&lease, PeerAnswerKind::Pin);
                if let Some(record) = self.records.get_mut(&lease) {
                    let looked_up = &record.answer;
                    answer.num_tokens = looked_up.num_tokens;
                    answer.pages = looked_up.pages.clone();
                    answer.held = looked_up.num_tokens > 0
                        && (!record.lookup_synced
                            || cache.lease_can_pin(&lease)
                            || cache.lease_held(&lease))
                        && cache.pages_match(
                            &record.lookup,
                            looked_up.num_tokens,
                            &looked_up.pages,
                        );
                    if answer.held {
                        // The head after this applies the pin; until then no
                        // plan runs, and none may evict the pages. Keep the
                        // original `pinned_at` across idempotent `Pin` retries.
                        let first_pin = record.pinned_at.is_none();
                        record.pinned_at.get_or_insert_with(|| cache.plan_calls());
                        record.expires = now + self.record_lifetime;
                        if first_pin {
                            self.changes.push(PeerEventView {
                                kind: PeerEventKind::Pin,
                                lease: lease.clone(),
                                num_tokens: looked_up.num_tokens,
                                pages: looked_up.pages.clone(),
                                ..empty_event()
                            });
                        }
                    }
                }
                self.reply(lease, answer)
            }
            Request::Release { lease, demote } => {
                let mut answer = self.answer(&lease, PeerAnswerKind::Release);
                // Untouched from the pin: no plan ran since its answer, or its
                // pin holds the pages now.
                answer.held = self
                    .records
                    .remove(&lease)
                    .and_then(|record| record.pinned_at)
                    .is_some_and(|pinned_at| {
                        cache.plan_calls() == pinned_at || cache.lease_held(&lease)
                    });
                self.changes.push(PeerEventView {
                    kind: PeerEventKind::Release,
                    lease: lease.clone(),
                    demote: demote && answer.held,
                    ..empty_event()
                });
                self.reply(lease, answer)
            }
        }
    }

    /// The changes answered since the last take, in order, for the next head.
    pub fn take_changes(&mut self) -> Vec<PeerEventView> {
        for record in self.records.values_mut() {
            record.lookup_synced = true;
        }
        self.unsynced_new_leases.clear();
        std::mem::take(&mut self.changes)
    }

    /// Lease records held.
    pub fn num_records(&self) -> usize {
        self.records.len()
    }

    fn answer(&self, lease: &PeerLeaseView, kind: PeerAnswerKind) -> PeerAnswerView {
        PeerAnswerView {
            lease: lease.clone(),
            kind,
            num_tokens: 0,
            pages: Vec::new(),
            held: false,
        }
    }

    fn reply(&self, lease: PeerLeaseView, answer: PeerAnswerView) -> (String, Reply) {
        (
            lease.reader,
            Reply::Answer {
                stamp: self.stamp.clone(),
                answer,
            },
        )
    }
}

/// An event with nothing set, to fill in.
pub fn empty_event() -> PeerEventView {
    PeerEventView {
        kind: PeerEventKind::Lookup,
        lease: PeerLeaseView {
            reader: String::new(),
            id: 0,
        },
        prefix_hashes: Vec::new(),
        tokens: Vec::new(),
        max_tokens: 0,
        pages: Vec::new(),
        demote: false,
        num_tokens: 0,
        request_id: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use ts_scheduler::PeerTier;

    use super::*;
    use crate::peer::wire::Prompt;

    struct FakeCache {
        max_leases: usize,
        soft_leases: HashSet<PeerLeaseView>,
        hard_leases: HashSet<PeerLeaseView>,
        matching: bool,
        plans: i64,
    }

    impl HolderCache for FakeCache {
        fn answer_lookup(
            &self,
            lookup: &PeerEventView,
            pending_new_leases: usize,
        ) -> PeerAnswerView {
            let known =
                self.soft_leases.contains(&lookup.lease) || self.hard_leases.contains(&lookup.lease);
            let occupied = self.soft_leases.len() + self.hard_leases.len();
            let accepted = known
                || (occupied < self.max_leases
                    && pending_new_leases < self.max_leases - occupied);
            PeerAnswerView {
                lease: lookup.lease.clone(),
                kind: PeerAnswerKind::Lookup,
                num_tokens: if accepted { lookup.max_tokens } else { 0 },
                pages: if accepted {
                    vec![PeerPageView {
                        group_id: 0,
                        slot: 0,
                        tier: PeerTier::Device,
                        block_id: 1,
                    }]
                } else {
                    Vec::new()
                },
                held: false,
            }
        }

        fn pages_match(
            &self,
            _lookup: &PeerEventView,
            _num_tokens: i32,
            _pages: &[PeerPageView],
        ) -> bool {
            self.matching
        }

        fn lease_can_pin(&self, lease: &PeerLeaseView) -> bool {
            self.soft_leases.contains(lease)
        }

        fn lease_held(&self, lease: &PeerLeaseView) -> bool {
            self.hard_leases.contains(lease)
        }

        fn plan_calls(&self) -> i64 {
            self.plans
        }
    }

    fn lease(id: u64) -> PeerLeaseView {
        PeerLeaseView {
            reader: "10.0.0.2:7100".into(),
            id,
        }
    }

    fn prompt() -> Prompt {
        Prompt {
            prefix_hashes: vec!["h0".into()],
            tokens: vec![1, 2, 3, 4],
            max_tokens: 4,
        }
    }

    #[test]
    fn expired_soft_pin_and_max_leases_overflow_never_phantom_pin_or_vouch() {
        let mut cache = FakeCache {
            max_leases: 2,
            soft_leases: HashSet::new(),
            hard_leases: HashSet::new(),
            matching: true,
            plans: 10,
        };
        let mut holder = Holder::new(
            Stamp {
                name: "10.0.0.1:7100".into(),
                incarnation: 1,
                identity: "ns/v1".into(),
                quarantine_us: 54_000_000,
            },
            Vec::new(),
            Duration::from_secs(60),
        );
        let now = Instant::now();

        // 1. Two unsynced lookups fill `max_leases = 2`; the 3rd unsynced lookup gets 0 tokens!
        let (_, Reply::Answer { answer: a1, .. }) = holder.handle(
            &cache,
            Request::Lookup {
                lease: lease(1),
                prompt: prompt(),
            },
            now,
        ) else {
            panic!();
        };
        let (_, Reply::Answer { answer: a2, .. }) = holder.handle(
            &cache,
            Request::Lookup {
                lease: lease(2),
                prompt: prompt(),
            },
            now,
        ) else {
            panic!();
        };
        let (_, Reply::Answer { answer: a3, .. }) = holder.handle(
            &cache,
            Request::Lookup {
                lease: lease(3),
                prompt: prompt(),
            },
            now,
        ) else {
            panic!();
        };
        assert_eq!(a1.num_tokens, 4);
        assert_eq!(a2.num_tokens, 4);
        assert_eq!(a3.num_tokens, 0);

        // Unsynced `lease(1)` CAN pin before `take_changes()` syncs it to `PeerExports`.
        let (_, Reply::Answer { answer: p1, .. }) =
            holder.handle(&cache, Request::Pin { lease: lease(1) }, now)
        else {
            panic!();
        };
        assert!(p1.held);

        // Sync changes (`take_changes`): `lease(2)` enters `soft_leases`, then expires before `Pin`!
        let _ = holder.take_changes();
        // Simulate soft-pin expiry of `lease(2)` in `PeerExports` (`soft_leases` does NOT contain `lease(2)`):
        let (_, Reply::Answer { answer: p2, .. }) =
            holder.handle(&cache, Request::Pin { lease: lease(2) }, now + Duration::from_secs(3))
        else {
            panic!();
        };
        assert!(
            !p2.held,
            "synced lookup whose PeerExports soft lease expired MUST reply Pin {{ held: false }}"
        );

        // Simulate cache clear incrementing `plans` (`10 -> 11`) without `hard_leases` holding `lease(1)`:
        cache.plans += 1;
        let (_, Reply::Answer { answer: r1, .. }) = holder.handle(
            &cache,
            Request::Release {
                lease: lease(1),
                demote: true,
            },
            now + Duration::from_secs(3),
        ) else {
            panic!();
        };
        assert!(
            !r1.held,
            "Release must never vouch held=true once plan_calls incremented on cache clear"
        );
    }
}
