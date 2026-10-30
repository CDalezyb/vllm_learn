# NIXL push-mode KV transfer

The default NIXL connector is **pull-based**: the decode (D) instance
reads KV blocks from the prefill (P) instance via `NIXL READ` after
prefill completes. `NixlPushConnector` adds a **push-based** alternative
in which P writes the KV blocks directly into D's pre-allocated memory
via `NIXL WRITE`.

Enable push with `"kv_connector": "NixlPushConnector"`. Alternatively, use
`"kv_connector": "NixlConnector"` and set
`"kv_connector_extra_config": {"transfer_mode": "push"}`.

For the built-in `NixlConnector`, `transfer_mode` defaults to `"pull"` and accepts
only `"pull"` or `"push"`; other values raise `ValueError`.
Explicit `NixlPullConnector` and `NixlPushConnector` names ignore this option.
`kv_connector_module_path` takes precedence over built-in selection.
Under `MultiConnector`, set `transfer_mode` in the NIXL child's
`kv_connector_extra_config`.

This document describes the threading, queues, and scheduling
interactions specific to the push design. The pull-mode design is
unchanged; the push connector reuses the same handshake, NIXL agent
setup, and metadata path wherever possible.

## High-level flow

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Proxy
    participant DSched as D Scheduler
    participant DWorker as D Worker (main)
    participant DWriter as D Writer
    participant PWriter as P Writer
    participant PWorker as P Worker (main)
    participant PSched as P Scheduler

    Client->>Proxy: POST /v1/completions
    Proxy->>PSched: prefill leg (do_remote_decode=True, max_tokens=1)
    Proxy->>DSched: decode leg (do_remote_prefill=True, P coordinates)

    note over DSched,DWriter: D side - register blocks with P
    DSched->>DSched: update_state_after_alloc, stash registration, arm watchdog
    DSched->>DWorker: build_connector_meta -> meta.push_registrations
    DWorker->>DWriter: enqueue (req_id, reg_data) on _reg_send_inbox
    DWriter->>PWriter: NIXL send_notif PUSH_REG msgpack

    note over PSched,PWriter: P side - prefill, stage finished blocks
    PSched->>PSched: request_finished, stash blocks
    PSched->>PWorker: build_connector_meta -> meta.push_finished_blocks
    PWorker->>PWriter: enqueue (req_id, blocks) on _finished_blocks_inbox

    note over PWriter: P writer matches and WRITEs
    PWriter->>PWriter: get_new_notifs returns PUSH_REG, route via _handle_push_reg_notif
    alt PUSH_REG and finished blocks both present
        PWriter->>PWriter: pop matching pair, fire WRITE
    else only one side present
        PWriter->>PWriter: stash and wait, self-poll only when blocks unmatched
    end
    PWriter->>PWriter: _ensure_handshake to D (async; defer WRITE)
    PWriter->>PWriter: handshake callback re-queues on _deferred_push_inbox, wake
    PWriter->>DWriter: NIXL WRITE direct to D GPU + completion notif

    note over DWorker,DWriter: D side - completion accounting
    DWriter-->>DWorker: forward HB and completion notifs via _pending_completion_notifs
    DWorker->>DWorker: _get_new_notifs drains, HB extends lease, completion marks recv done
    DWorker->>DSched: update_connector_output(finished_recving)
    DSched->>DSched: clear watchdog deadline

    note over PWorker,PWriter: P side - reclaim
    PWorker->>PWorker: get_finished, drain _sending_transfers, queue eviction
    PWriter->>PWriter: drain _evict_finished_inbox, drop stale state
    PWorker->>PSched: update_connector_output(finished_sending)
    PSched->>PSched: free lease

    DWorker-->>Proxy: stream decode tokens
    Proxy-->>Client: response
```

## Threads

``NixlPushConnectorWorker`` introduces a single dedicated background
thread per worker (i.e. per TP rank), named ``nixl-push-writer``.
Each owns the new push-specific NIXL operations on its rank:

* ``nixl_wrapper.get_new_notifs()`` — receive notifications.
* ``nixl_wrapper.send_notif(...)`` for the ``PUSH_REG:<msgpack>`` (D
  side) and for the per-WRITE completion notif (P side).
* ``nixl_wrapper.make_prepped_xfer(...) / transfer(...)`` — submit the
  WRITE itself.

Heartbeats continue to go out from the engine main thread via the
existing base-worker ``_send_heartbeats`` plumbing inside
``start_load_kv``.

### Wake model

The writer thread blocks on ``_push_writer_wake`` (a
``threading.Event``) when it has no work. These callers set the
event:

1. **``start_load_kv``** (worker main thread, called once per engine
   step with the scheduler's metadata) — sets the wake only when the
   step actually hands the writer new work, i.e. when
   ``meta.push_registrations``, ``meta.push_finished_blocks`` or
   ``meta.push_chunkwise_finished`` is non-empty. This is the wake for
   new transfers.
2. **``get_finished``** (worker main thread, called once per engine
   step to report completions) — always sets the wake. The writer is
   the sole consumer of ``nixl_wrapper.get_new_notifs()`` for push,
   so this gives it a chance to drain inbound notifs (heartbeats from
   D, completion notifs after a WRITE, late-arriving ``PUSH_REG``)
   even when there is no new metadata to act on.
3. **Handshake-completion callback** (background handshake executor
   thread) — both handshakes run on the executor and never block the
   writer; their done-callbacks re-enqueue the deferred op and set the
   wake, since neither ``send_notif`` nor the NIXL WRITE may run off the
   writer thread:
    * the **D→P** handshake (before sending ``PUSH_REG``) re-enqueues the
      registration onto ``_reg_send_inbox``;
    * the **P→D** handshake (before a WRITE) re-enqueues the matched
      ``(req_id, blocks, reg_data)`` onto ``_deferred_push_inbox``;
    * the **P→D** handshake of a chunk-wise push queues a ``"ready"``
      message onto ``_cw_inbox``, with the error if it failed.

   On this second pass ``_ensure_handshake`` returns ``None`` (the agent
   is now connected), so the writer sends the ``PUSH_REG`` / issues the
   WRITE directly. If a handshake *failed*, the callback fails or drops
   the request instead of re-enqueuing (a chunk-wise push is failed by
   the writer on its ``"ready"``), so there is no retry loop (see
   Failure handling).
4. **Chunk-wise push** (worker main thread) — ``bind_chunkwise`` at the
   start of every step, and ``wait_for_save`` when a step's forward
   queued chunk-wise tasks.

In addition to event-driven wakes, the writer self-polls at
``_PUSH_WRITER_POLL_INTERVAL_MS = 1.0`` ms while there are P-side
finished blocks waiting for an unmatched ``PUSH_REG``, whole-request
WRITEs in flight under ``push_writer_progress``, or chunk-wise pushes
with WRITEs in flight or waiting for their ``PUSH_REG``. When all that
is left are chunk-wise WRITEs of a step still being computed, it blocks
on that step's CUDA event instead (a blocking-sync event: it sleeps
rather than spins); work handed to it meanwhile waits until the step is
computed.

When a request completes on P (lease expires or the WRITE finishes),
``get_finished`` enqueues the request id onto ``_evict_finished_inbox``,
which the writer drains to drop stale ``_push_finished_blocks`` /
``_pending_d_registrations`` and stop self-polling.

### Writer-side WRITE progress

NIXL's UCX backend sends a WRITE's completion notif only when the initiator
checks the transfer (``check_xfer_state``). By default P checks its WRITEs once
per engine step, from ``get_finished``, so while P runs a long prefill step D
waits for KV that has already landed. With
``"push_writer_progress": true`` in ``kv_connector_extra_config`` the writer
polls the in-flight WRITEs every ``_PUSH_WRITER_POLL_INTERVAL_MS`` instead and
hands the results to ``get_finished`` through ``_sending_results``.
Chunk-wise WRITEs (below) are always polled by the writer.

## Writer-local matching tables

| Table                          | Owner            | Holds                                                                  |
|--------------------------------|------------------|------------------------------------------------------------------------|
| `_pending_d_registrations`     | writer           | D registrations received from a remote D, waiting for P's blocks       |
| `_push_finished_blocks`        | writer           | P blocks staged by the scheduler, waiting for a remote D registration  |

Either side can arrive first. The writer matches in both directions:
when a ``PUSH_REG`` arrives we look up ``_push_finished_blocks``, and
when finished blocks arrive we look up ``_pending_d_registrations``.
Both lookups try an exact ``request_id`` match first, then fall back
to comparing the ids after stripping the trailing per-engine random
suffix (via ``get_base_request_id``). The fallback exists because the
proxy hands the same ``X-Request-Id`` to both legs, so P and D wrap it
into the same ``cmpl-<uuid>-<index>`` form and differ only by the
8-hex randomization suffix that ``input_processor.assign_request_id``
appends per engine. Stripping just that suffix normalizes both sides
to the same id while preserving the completion index (so multi-prompt
sub-requests stay distinct). It also works whether or not
``VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`` is set, which matters since
that env var is slated for removal upstream. Chunk-wise pushes waiting
for their ``PUSH_REG`` match the same way (``match_request_id``).

## Wire format

A push registration is sent as a NIXL notification:

```text
PUSH_REG:<msgpack-encoded dict>
```

Fields in the dict:

| Field                | Set by | Meaning                                                                |
|----------------------|--------|------------------------------------------------------------------------|
| ``request_id``       | D      | D's own vLLM request id; P's match key, echoed in the completion notif |
| ``decode_engine_id`` | D      | D's engine id (P uses this for the reverse handshake)                  |
| ``decode_host``      | D      | D's NIXL side-channel host                                             |
| ``decode_port``      | D      | D's NIXL side-channel port                                             |
| ``decode_tp_size``   | D      | D's tensor-parallel size                                               |
| ``local_block_ids``  | D      | per-group lists of D's *logical* block ids (preallocated)              |
| ``remote_engine_id`` | D      | P's engine id (for the existing P-side handshake)                      |
| ``remote_host``      | D      | P's NIXL side-channel host                                             |
| ``remote_port``      | D      | P's NIXL side-channel port                                             |
| ``remote_tp_size``   | D      | P's tensor-parallel size                                               |

D ships **logical** block ids; P expands them to physical block ids at
WRITE-submission time using the ratio learned during the NIXL
handshake (`remote_physical_blocks_per_logical`). This matches the
pull-mode contract — schedulers ship logical ids, workers expand to
physical at submission.

The completion notif sent from P to D after a WRITE is the existing
`<request_id>:<tp_size>` format used in pull mode (here ``request_id``
is D's own request id, taken from the registration), so the D-side
accounting code is unchanged.

## Pipeline parallelism and hybrid KV caches

A peer's KV regions do not line up with the local ones by index under
pipeline parallelism (PP) or a hybrid (HMA) KV layout:

* a PP-sharded prefiller (**P**) holds only a slice of the model's
  layers while the `PP=1` decoder (**D**) holds them all, so region
  counts differ;
* with HMA, layers of several cache groups can share one region, and
  which layers share a region can differ between P and D.

Both connectors therefore pair regions **by layer**, for every peer. Each
worker advertises which layers back each of its NIXL regions in the
handshake metadata (`NixlAgentMetadata.region_members`). On the
handshake, `_pair_regions` looks each layer up on both sides and records
one pair (remote region, local region, cache group) per layer,
deduplicated; a layer registered as two regions (K and V apart) pairs
them in order. The remote descriptor list keeps only the remote regions
that pair, and a transfer reads or writes each pair at its cache group's
blocks. A region shared by several groups thus transfers once per group
it holds a layer of, from one descriptor list.

```text
D (PP=1):  region 0 = [L0, L0.swa]   region 1 = [L1]   region 2 = [L2]
P stage 1: region 0 = [L2]

pairs (remote, local, group) on P: (2, 0, 0)  ->  D's region 2 only
```

Region capacities stay per region on both sides, so P and D need not
allocate the same number of blocks. Physical allocations are still
registered once, even when multiple layers share them. Mamba/SSM hybrids
pair the same way: each SSM region's conv and SSM-state descriptors are
built from the peer region its layers pair with, so P and D may group the
same layers into regions differently. A layer in different cache groups on
P and D is refused at the handshake for these models.

Packed MLA caches (block-outermost layouts such as DeepSeek-V4's) store the
pages of every layer for one block in a single row. A PP stage holds only its
own layers, so its rows are shorter and each layer sits at a different offset:

```text
D (PP=1),    block b: | L0 | L1 | L2 | L3 | L4 | L5 |
P (stage 2), block b: | L3 | L4 | L5 |
```

`PP=1` peers transfer whole rows. They also advertise where each layer's page
sits in the row (`NixlAgentMetadata.packed_member_layouts`). A PP producer
registers one region per layer, so it views each of D's rows as one region per
layer and pairs its pages with those. Page sizes must match; offsets, row
strides, and block counts may differ. Because PP stages can list the same
attention backends in a different order, packed push compares the sorted set of
backend names during the handshake.

Invariants enforced when pairing:

* the remote must hold every local layer; a missing layer fails the
  handshake rather than silently leaving that layer's KV stale. Layers
  only the remote holds are skipped on a PP stage (other stages own
  them) and fail the handshake otherwise;
* a remote that does not advertise its layers fails the handshake
  instead of falling back to pairing by index;
* a layer must span the same number of regions on both sides;
* every TP rank of one remote engine must pair the same way;
* a layer in different cache groups on P and D is refused by push. Each
  worker advertises every layer's cache group
  (`NixlAgentMetadata.layer_group_ids`), so this is caught even when the
  layer's region is shared by several groups.

Decode-side PP is unsupported because completions are counted per
consumer rank. Mamba/SSM hybrids are unsupported under PP. Packed PP push
requires MLA caches and equal P/D block sizes. MLA pages are replicated
across TP ranks, so producer TP1 to consumer TP2 is supported.

## Scheduler-side responsibilities

`NixlPushConnectorScheduler` extends the base scheduler with:

* **D side** — `update_state_after_alloc` stashes registration data in
  `_push_pending_registrations` and arms a soft watchdog
  (`_push_registration_deadlines`). `build_connector_meta` drains the
  stash into `meta.push_registrations` and any expired entries are
  dropped with a warning.
* **P side** — `request_finished` stashes block IDs in
  `_finished_request_blocks` (for the lease and for
  `has_pending_push_work`) and `_newly_finished_push_blocks` (for the
  next worker step via `meta.push_finished_blocks`).
* **Both sides** — `has_pending_push_work` keeps the engine main loop
  stepping while there is in-flight push state, so the writer always
  gets at least one wake per step.

`update_connector_output`:

* `finished_sending` (P side) clears the lease entry.
* `finished_recving` (D side) clears the watchdog deadline.

## Timeouts and watchdogs

Two per-request timers are armed on the scheduler:

* **D-side registration watchdog** — ``_push_registration_deadlines``.
  If a registered request does not see a push completion within
  ``push_registration_timeout`` seconds (defaults to
  ``decoder_kv_blocks_ttl``), ``build_connector_meta`` drops the stale
  registration and the pending entry, logs a warning, and stops trying
  to resend the registration. The corresponding request remains tracked
  in ``_reqs_need_recv``; it is the engine's request-level abort path
  (or the user / proxy timing out the HTTP call) that ultimately fails
  the request.
* **P-side block lease** — same ``_kv_lease_duration`` used by pull
  mode. ``request_finished`` sets the expiration in ``_reqs_need_send``
  and ``update_connector_output(finished_sending=...)`` clears it on
  successful WRITE. Stale leases are reaped by ``get_finished`` in the
  base worker, which then enqueues the eviction onto
  ``_evict_finished_inbox`` so the writer also stops self-polling.

## Failure handling

* **D-side handshake failure (P→D handshake before sending PUSH_REG)** —
  the future's done-callback calls ``_handle_failed_transfer(rid, None)``,
  which marks D's pre-allocated blocks invalid and enqueues onto
  ``_failed_recv_reqs`` so the next ``get_finished`` reports the
  request as a failed recv. Same recv-side accounting as pull mode.
* **D-side ``send_notif`` failure when shipping the PUSH_REG to P** —
  identical handling: ``_handle_failed_transfer`` marks the recv as
  failed.
* **P-side handshake failure (P→D handshake before a WRITE)** — the
  future's done-callback logs ``push_handshake_failed`` and drops the
  request without re-queuing. It deliberately does *not* call
  ``_handle_failed_transfer`` (there is no ``_recving_metadata`` entry to
  invalidate on the producer side, same reasoning as the WRITE-submission
  failure below). P's blocks are reclaimed by the ``_kv_lease_duration``
  lease and D's stale registration by its watchdog. For a chunk-wise
  push the callback queues the failure, and the writer fails the push.
* **P-side chunk-wise WRITE failure** (submission or completion) — the
  push is failed: no further WRITE goes out, in particular not the final
  one carrying D's completion notif, so D never takes a partial push for
  a complete one. P's blocks are reclaimed by the lease.
* **P-side WRITE submission failure** — the WRITE handle (if any) is
  released and ``xfer_stats.record_failed_transfer()`` bumps the
  failure counter. We deliberately do not call
  ``_handle_failed_transfer`` here: ``req_id`` on the P side has no
  entry in ``_recving_metadata`` (P is not the receiver), so the
  helper would put a P-local request id into ``_failed_recv_reqs``
  and trip the assertion in the base worker's ``get_finished``. The
  outbound WRITE is dropped on the floor; D's lease watchdog handles
  the missing completion.

## Chunk-wise push

By default P WRITEs a request's KV once, after its prefill finished and the
scheduler released it. With chunk-wise push, P WRITEs the KV of each prefill
chunk (scheduler step) right after the step computed it, while the next chunks
are computed. When the prefill ends only the last chunk's KV is left to send.
Enable it on the prefill instance only:

```json
{"kv_connector": "NixlPushConnector", "kv_role": "kv_producer",
 "kv_connector_extra_config": {"chunkwise_push": true}}
```

D needs no change. The overlap only happens when D registers its blocks before
or during the prefill, i.e. with a proxy that sends the P and D legs
concurrently (as `disagg_proxy_pushconnector_demo.py` does). It applies to the
requests that finish with their prefill (`max_tokens=1`, as P legs are sent);
the others are pushed whole.

* **Scheduler.** Every step, `build_connector_meta` ships in
  `meta.push_chunkwise_steps` the attention blocks each such request's prefill
  completes in that step (a partially filled block waits for the step that
  fills it); the last chunk ships the rest of the prompt's blocks, without the
  lookahead slots of speculative decoding, and the Mamba state. Block positions
  are in units of the transfer block size the workers use
  (`transfer_block_size`). Such requests skip the post-prefill WRITE; their
  block table still goes to the workers once they finished
  (`meta.push_chunkwise_finished`), for a D of other geometry (see below).
* **Worker.** Every step's metadata is bound before its forward
  (`bind_chunkwise`). At the end of the forward, `wait_for_save` records a CUDA
  event and queues one `_ChunkTask` per request on `_cw_inbox`. Descriptor ids
  are the ones of a whole-request WRITE of the task's blocks, across every
  layer, and Mamba state slots pair like in a whole-request WRITE.
* **Writer.** Tasks queue up until D's `PUSH_REG` is matched and the P->D
  handshake is done. The writer then WRITEs the queued tasks whose event fired,
  in one WRITE. It checks the events without blocking, so the WRITE of chunk i
  overlaps with the compute of the following chunks, and a step being computed
  never holds back other WRITEs. The request's last WRITE carries the
  completion notif. It is only posted once all earlier WRITEs of the request
  finished, so the notif implies all KV has landed on D, and once the next
  step's metadata was bound without cancelling the request. D's completion
  accounting is unchanged.
* **Preemption and abort.** A preempted or aborted request's blocks are freed
  before its lease. `bind_chunkwise` cancels its push before the forward that
  may reuse them: the scheduler reports preempted requests in
  `meta.push_chunkwise_preempted` (also a request listed as new again, e.g. a
  resumed streaming session), and aborted ones in `meta.reqs_not_processed`.
  Under async scheduling an abort reaches the worker one step late, possibly
  after the final WRITE went out; `bind_chunkwise` then waits for that WRITE (a
  fence), so D is never notified of KV read from reused blocks. A preempted
  request is pushed as a whole when it finishes. D's registration is handed to
  that push once the cancelled push's WRITEs are done, so the whole-request
  WRITE overwrites any chunk read from a reused block.
* **Completion on P.** A push usually completes before the scheduler finished
  the request. The worker only reports it in `finished_sending` once the
  request's lease (sent when the scheduler finishes it) reached the worker, so
  blocks are never released under a running request.

Chunk-wise WRITEs pair P and D blocks by position. If D's TP size, block size,
kernel-block ratio, source ranks or cache group layout differs from P's, the
request is pushed as a whole once it finished instead, like a request without
chunk-wise push (the scheduler's block table, under the lease); a warning is
logged once per D and reason. Chunk-wise push is disabled (with a warning) with
pipeline or context parallelism, host-buffer transfer,
`kv_load_failure_policy=recompute` (a request recomputed in place would leave
its pushed chunks stale), cache groups other than full attention, MLA and Mamba
(e.g. sliding window), and attention groups of different block sizes.

## Summary

The push design is a small, well-contained extension on top of the
existing NIXL connector:

* one new connector class, one new scheduler class, one new worker
  class — all subclasses of the existing base classes;
* one dedicated background thread per worker;
* a few cross-thread queues, each with a single consumer: the writer, or
  the engine main thread for the writer's results
  (``_pending_completion_notifs``, ``_sending_results``, ``_cw_done``);
  most have one producer, except the replay queues fed by both the engine
  main thread and a handshake-completion callback: ``_reg_send_inbox``
  (registrations replayed after their D→P handshake),
  ``_deferred_push_inbox`` (matched pushes replayed after their P→D
  handshake) and, with chunk-wise push, ``_cw_inbox`` (its handshake's
  ``"ready"``);
* one new notification type (`PUSH_REG:<msgpack>`).

Behavior on the engine main thread is otherwise unchanged, except that with
chunk-wise push `bind_chunkwise` may wait for a cancelled request's final WRITE
before a forward. The writer thread is event-driven and idle when there is no
push work.
