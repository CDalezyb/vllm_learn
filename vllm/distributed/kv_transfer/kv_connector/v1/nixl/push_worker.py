# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated ``nixl-push-writer`` thread owns all push-related NIXL ops:
calls ``get_new_notifs`` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
``send_notif``, matches D registrations with P finished blocks, and
issues WRITE transfers via ``make_prepped_xfer`` / ``transfer``.

The engine main thread feeds the writer through queues:
``_reg_send_inbox`` (D-side regs to send), ``_finished_blocks_inbox``
(P-side blocks from metadata) and ``_pending_completion_notifs``
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds ``_deferred_push_inbox`` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every ``_PUSH_WRITER_POLL_INTERVAL_MS``
while something only polling reveals is pending: unmatched
``_push_finished_blocks`` (P-side blocks waiting for a D PUSH_REG notif),
whole-request WRITEs under ``push_writer_progress``, and chunk-wise pushes
with WRITEs in flight or waiting for a PUSH_REG. When all that is left are
chunk-wise WRITEs waiting for a step to be computed, it sleeps on that step's
CUDA event (other work handed to it meanwhile waits until the step is
computed). Otherwise it waits on ``_push_writer_wake``: the engine main thread
sets it from ``start_load_kv`` (when handing it new work), from
``get_finished`` (so each engine step gives the writer a chance to drain NIXL
notifs) and, with chunk-wise push, from ``bind_chunkwise`` and
``wait_for_save``; the handshake-completion callbacks set it after queueing a
deferred PUSH_REG send, a deferred push WRITE or a chunk-wise "ready". When a
request's lease expires (the base worker reports it via ``done_sending``) or
the WRITE completes, ``get_finished`` enqueues an eviction onto
``_evict_finished_inbox`` so the writer drops any leftover
``_push_finished_blocks`` / ``_pending_d_registrations`` and stops
self-polling.

Chunk-wise push (P side, ``chunkwise_push``): instead of one WRITE after the
prefill, ``wait_for_save`` queues a ``_ChunkTask`` per request at the end of
every prefill step (chunk), with the blocks the step completed and a CUDA event
recorded after the step's forward (``_cw_inbox``). The writer WRITEs them once
the event fired (it never waits on one while it has other work), so chunk i's
KV moves while chunk i+1 is computed (Mamba state only with the last chunk).
Tasks queue up until D's PUSH_REG arrives. The request's last WRITE carries the
completion notif. It is posted once all its earlier WRITEs are done, so D never
sees the notif before the data, and once the next step's metadata was bound
without cancelling the request (see ``bind_chunkwise``). Preempted requests,
and requests to a D of other geometry, are pushed whole once finished.
"""

import contextlib
import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
    _is_ssm_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    chunkwise_push_unsupported_reason,
    match_request_id,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0
# Chunk-wise push: longest wait, before a forward, for the final WRITE of a
# request preempted or aborted since the last step (see bind_chunkwise).
_CHUNKWISE_FENCE_TIMEOUT_S = 10.0


@dataclass
class _ChunkTask:
    """WRITE of the KV one prefill step (chunk) completed for a request."""

    req_id: ReqId
    # Worker step count when queued, see _cw_final_due.
    seq: int
    # Kernel-block index of the request's first attention block in the task.
    start: int
    # Per group, kernel block ids of the attention blocks the step completed
    # and, with the last chunk, of the Mamba state.
    kernel_block_ids: BlockIds
    num_prompt_tokens: int
    # Last chunk: carries the Mamba state and the completion notif.
    final: bool
    # Recorded after the step's forward: the blocks are computed once it fires.
    event: "torch.cuda.Event"


@dataclass
class _ChunkSignal:
    """Writer control message for a chunk-wise request."""

    req_id: ReqId
    # "drop": aborted or lease gone; "fallback": preempted; "finished": the
    # request finished (``blocks``); "ready": P->D handshake done (``error``).
    kind: str
    error: BaseException | None = None
    # "drop" / "fallback": set once the request's in-flight WRITEs are done.
    fence: threading.Event | None = None
    blocks: BlockIds | None = None


@dataclass
class _ChunkwiseWrite:
    """Writer-thread state of one chunk-wise request."""

    num_prompt_tokens: int
    reg: dict[str, Any] | None = None
    # Handshaked and geometry resolved.
    ready: bool = False
    # D's geometry rules out chunk-wise WRITEs: pushed whole once finished.
    monolithic: bool = False
    # "drop" (aborted, lease gone) or "fallback" (preempted, pushed whole).
    cancelled: str | None = None
    failed: bool = False
    # The final WRITE (with D's notif) was posted.
    final_posted: bool = False
    # Tasks not WRITTEN yet, in step order.
    pending: deque[_ChunkTask] = field(default_factory=deque)
    handles: list[TransferHandle] = field(default_factory=list)
    # Cancellation fences, set once the in-flight WRITEs are done.
    fences: list[threading.Event] = field(default_factory=list)
    # The request's block table once it finished, for a whole-request push.
    finished_blocks: BlockIds | None = None
    # Resolved on ``ready``.
    dst_engine_id: str = ""
    local_handle: int = 0
    remote_handle: int = 0
    remote_agent: str = ""
    remote_physical_per_logical: int = 1
    notif: bytes = b""
    remote_block_ids: BlockIds = ()
    # Attention kernel-block offset between P's and D's block lists (D only
    # allocates the suffix it has not cached).
    fa_offset: int = 0

    @property
    def pushing(self) -> bool:
        """Neither cancelled nor failed."""
        return self.cancelled is None and not self.failed

    @property
    def awaiting_registration(self) -> bool:
        """Pushing, but D's PUSH_REG has not arrived yet."""
        return self.reg is None and self.pushing


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

    # Distinguishes push from pull in the NIXL compatibility hash.
    _TRANSFER_MODE: str = "push"

    _supports_pp_hma = True

    # Set from ``chunkwise_push`` in kv_connector_extra_config (P side).
    chunkwise_push: bool = False
    # Set from ``push_writer_progress`` in kv_connector_extra_config.
    _writer_progress: bool = False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Heartbeat handshakes to a PP-sharded producer must be notif-only,
        # like the PUSH_REG path.
        self._hb_handshake_notif_only = True

        # Push-specific state.
        # P-side: outgoing WRITE handles awaiting completion, keyed by
        # request_id. Mutated by writer (submit) and main thread
        # (``_pop_done_transfers``); guarded by
        # ``_sending_transfers_lock``.
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._send_failures: set[ReqId] = set()
        self._sending_transfers_lock = threading.Lock()

        # Writer-thread owned matching state.
        # P-side: finished request blocks received from scheduler metadata
        # that have not yet been matched with an incoming D registration.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations received via NIXL notification that have
        # not yet been matched with a finished P request.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}

        # Cross-thread channels.
        self._reg_send_inbox: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._finished_blocks_inbox: queue.Queue[tuple[str, BlockIds]] = queue.Queue()
        self._pending_completion_notifs: queue.Queue[bytes] = queue.Queue()
        # Main thread → writer: req_ids whose lease has expired or whose
        # WRITE has completed. Writer drops them from
        # ``_push_finished_blocks`` so an unmatched entry doesn't keep the
        # writer busy-polling forever.
        self._evict_finished_inbox: queue.Queue[str] = queue.Queue()
        # Handshakes that have just completed and are ready for the WRITE on wthread
        self._deferred_push_inbox = queue.Queue[tuple[str, BlockIds, dict[str, Any]]]()

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

        # P-side: poll the WRITEs' completion on the writer thread (every
        # poll interval) instead of once per engine step. NIXL's UCX backend
        # only sends a WRITE's notif when the initiator checks the transfer,
        # so per-step polling delays D by up to a (long prefill) step.
        self._writer_progress = bool(
            self.kv_transfer_config.get_from_extra_config("push_writer_progress", False)
        )
        # Writer -> main thread: (done, failed) request ids of the WRITEs.
        self._sending_results: queue.Queue[tuple[set[str], set[str]]] = queue.Queue()

        # Chunk-wise push (P side), see module docstring.
        self.chunkwise_push = bool(
            self.kv_transfer_config.get_from_extra_config("chunkwise_push", False)
        )
        if self.chunkwise_push and (
            reason := chunkwise_push_unsupported_reason(vllm_config, kv_cache_config)
        ):
            logger.warning_once("Disabling NIXL chunk-wise push: %s.", reason)
            self.chunkwise_push = False
        self._init_chunkwise_state()
        # Groups indexed by token (attention), as opposed to Mamba state.
        self._cw_attn_groups = [
            g for g, t in enumerate(self._group_spec_types) if not _is_ssm_spec(t)
        ]
        if self.chunkwise_push:
            logger.info("NIXL chunk-wise push enabled (rank %d)", self.tp_rank)

    def _init_chunkwise_state(self) -> None:
        # Main thread -> writer, in step order.
        self._cw_inbox: queue.Queue[_ChunkTask | _ChunkSignal] = queue.Queue()
        # Main thread: number of steps whose metadata was bound.
        self._cw_seq = 0
        # Main thread -> writer: preempted or aborted requests, until the
        # writer handled their "fallback" / "drop" message.
        self._cw_cancelled: set[ReqId] = set()
        # Writer -> main thread: requests whose final WRITE is in flight.
        self._cw_final_inflight: set[ReqId] = set()
        # Writer -> main thread: requests whose chunk-wise push completed.
        self._cw_done: queue.Queue[ReqId] = queue.Queue()
        # Main thread: completed pushes awaiting the scheduler's lease, so that
        # blocks are only released once the request has finished.
        self._cw_completed: set[ReqId] = set()
        # Writer-thread owned per-request state.
        self._cw_writes: dict[ReqId, _ChunkwiseWrite] = {}

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._mixed_mem_types:
            raise NotImplementedError(
                "NixlPushConnector does not support mixed-memory KV caches"
            )
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
        local_region_ids: Sequence[int] | None = None,
    ) -> None:
        super()._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size, local_region_ids
        )
        assert self.transfer_topo is not None
        if (
            self._head_sharded_draft_regions
            and self.transfer_topo.tp_size != remote_tp_size
        ):
            raise NotImplementedError(
                "NIXL push mode does not support a head-sharded draft's KV "
                "under an MLA target with heterogeneous TP. Use pull mode."
            )

    def shutdown(self) -> None:
        self._push_writer_stop.set()
        # Unblock the writer if it's waiting in the no-active-state branch.
        self._push_writer_wake.set()
        writer_alive = False
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            writer_alive = self._push_writer_thread.is_alive()
            self._push_writer_thread = None
        with self._sending_transfers_lock:
            for handles in self._sending_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
            self._send_failures.clear()
        # The chunk-wise state belongs to the writer: leave it to a stuck one.
        if not writer_alive:
            for state in self._cw_writes.values():
                for handle in state.handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._cw_writes.clear()
        super().shutdown()

    # --- Engine-main-thread entry point -------------------------------- #

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Pre-process metadata; defer NIXL ops to the writer thread."""
        if self.pcp_rank > 0 and not self.pcp_dcp_sharded:
            return

        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            # Update last activity from this remote (same as pull), but only for
            # an already-connected engine so we never leave an _engine_last_active
            # entry without a _remote_agents entry (breaks _cleanup_remote_engine).
            if remote_engine_id in self._remote_agents:
                self._engine_last_active[remote_engine_id] = time.perf_counter()

            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            self._recving_metadata[req_id] = meta

        # --- D-side: registrations to send to P via NIXL ---
        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        # --- P-side: newly finished blocks awaiting a D registration match ---
        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                self._finished_blocks_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        # Batch + lease tracking (same as pull).
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send
        # Rebase scheduler-clock deadlines onto this worker's clock — see the
        # equivalent block in pull_worker.start_load_kv for the rationale.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        if self.chunkwise_push and metadata.push_chunkwise_finished:
            for req_id, block_ids in metadata.push_chunkwise_finished.items():
                self._cw_inbox.put(_ChunkSignal(req_id, "finished", blocks=block_ids))
            self._push_writer_wake.set()

        # Heartbeats still leave from the main thread (base worker behaviour).
        self._send_heartbeats(metadata)

    # --- Chunk-wise push (engine main thread) -------------------------- #

    def bind_chunkwise(self, metadata: NixlConnectorMetadata) -> None:
        """Start of every step, before its forward: cancel the pushes of the
        requests preempted or aborted since the last step, then count the step.

        A preemption or an abort frees a request's blocks before its lease,
        and reaches this hook before the forward that may reuse them. The
        writer posts no WRITE for a cancelled request, and this waits for its
        final WRITE if in flight (a fence), so D is never notified of KV read
        from reused blocks. Hence once the step after a request's last chunk
        was counted without cancelling it, its final WRITE can go out (see
        _cw_final_due). An earlier WRITE reading reused blocks is harmless: D
        is not notified, or a whole-request push overwrites it.
        """
        preempted = metadata.push_chunkwise_preempted
        fences: list[tuple[ReqId, threading.Event]] = []
        for req_id in preempted | metadata.reqs_not_processed:
            self._cw_cancelled.add(req_id)
            # After the line above: a final WRITE the writer is about to post
            # shows up here, or it sees the cancellation (_cw_write_tasks).
            fence = threading.Event() if req_id in self._cw_final_inflight else None
            kind = "fallback" if req_id in preempted else "drop"
            self._cw_inbox.put(_ChunkSignal(req_id, kind, fence=fence))
            if fence is not None:
                fences.append((req_id, fence))
        # After the cancellations: the writer reads the count first.
        self._cw_seq += 1
        self._push_writer_wake.set()
        if (
            not fences
            or self._push_writer_thread is None
            or self._push_writer_stop.is_set()
        ):
            return
        deadline = time.monotonic() + _CHUNKWISE_FENCE_TIMEOUT_S
        for req_id, fence in fences:
            if not fence.wait(max(0.0, deadline - time.monotonic())):
                logger.warning(
                    "NIXL chunk-wise push: the final WRITE of %s is still in "
                    "flight after %.0fs; its blocks may be reused while read.",
                    req_id,
                    _CHUNKWISE_FENCE_TIMEOUT_S,
                )

    def wait_for_save(self, metadata: NixlConnectorMetadata) -> None:
        """End of a step's forward: queue the blocks it completed for WRITE."""
        if not metadata.push_chunkwise_steps:
            return
        # Blocking-sync: a writer sleeping on it doesn't spin.
        event = torch.cuda.Event(blocking=True)
        event.record()
        ratio = self._physical_blocks_per_logical_kv_block
        for req_id, step in metadata.push_chunkwise_steps.items():
            self._cw_inbox.put(
                _ChunkTask(
                    req_id=req_id,
                    seq=self._cw_seq,
                    start=step.start_block * ratio,
                    kernel_block_ids=self._logical_to_kernel_block_ids(
                        self._as_grouped_block_ids(step.block_ids), ratio
                    ),
                    num_prompt_tokens=step.num_prompt_tokens,
                    final=step.is_last,
                    event=event,
                )
            )
        self._push_writer_wake.set()

    # --- Writer thread ------------------------------------------------- #

    def _push_writer_loop(self) -> None:
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0
        if self.chunkwise_push:
            # Like the handshake thread: the CUDA event calls must not create
            # a context on another device.
            current_platform.set_device(self.device_id)

        while not self._push_writer_stop.is_set():
            try:
                # 0. Chunk-wise tasks and control messages, in order.
                if self.chunkwise_push:
                    self._cw_process_inbox()

                # 1. D registrations to send.
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Deferred P→D pushes whose handshake just completed; do xfer now
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

                # 3b. Evict finished blocks for requests that have either
                # completed (WRITE acknowledged) or whose lease expired
                # without a D registration.  Drop pending registrations
                # for the same reason so we don't leak state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)
                    if self.chunkwise_push:
                        self._cw_cancel(rid, "drop")

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)

                # 5. Chunk-wise WRITEs: completions, then the ready ones.
                if self.chunkwise_push:
                    self._cw_progress()

                # 6. Whole-request WRITE completions.
                if self._writer_progress and self._sending_transfers:
                    self._poll_sending_transfers()
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while something has no other wake source (see
            # module docstring).
            if (
                self._push_finished_blocks
                or (self._writer_progress and self._sending_transfers)
                or (self.chunkwise_push and self._cw_needs_poll())
            ):
                self._push_writer_stop.wait(timeout=sleep_s)
            elif self.chunkwise_push and (event := self._cw_next_event()):
                # Nothing to do before this step is computed: sleep on it
                # rather than poll (GIL / CUDA driver contention).
                event.synchronize()
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _poll_sending_transfers(self) -> None:
        """Writer-side WRITE polling (``push_writer_progress``); results go
        to the engine thread's ``get_transfer_results``."""
        with self._sending_transfers_lock:
            done, failed = self._pop_done_transfers(self._sending_transfers)
        if done or failed:
            self._sending_results.put((done, failed))

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        try:
            reg_data = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return
        self._route_push_registration(reg_data)

    def _route_push_registration(self, reg_data: dict[str, Any]) -> None:
        """Match a D registration with finished P blocks or a chunk-wise push,
        or keep it until one of them shows up."""
        rid = reg_data["request_id"]
        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        elif self.chunkwise_push and (
            (cw_id := self._cw_match_unregistered(rid)) is not None
        ):
            state = self._cw_writes[cw_id]
            with self._cw_guard(cw_id, state):
                self._cw_attach(cw_id, state, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data

    # --- Chunk-wise push (writer thread) ------------------------------- #

    @contextlib.contextmanager
    def _cw_guard(self, req_id: str, state: _ChunkwiseWrite) -> Iterator[None]:
        """Fail one request's chunk-wise push on error: D is never notified of
        a partial push, and the other requests go on."""
        try:
            yield
        except Exception as e:
            self._log_failure(
                failure_type="chunkwise_push_failed", req_id=req_id, error=e
            )
            self._cw_fail(state)

    def _cw_process_inbox(self) -> None:
        """Queue the steps' tasks and apply the control messages, in order."""
        while True:
            try:
                item = self._cw_inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _ChunkTask):
                self._cw_handle_task(item)
            elif item.kind in ("drop", "fallback"):
                self._cw_cancel(item.req_id, item.kind, item.fence)
            elif (state := self._cw_writes.get(item.req_id)) is None:
                continue
            elif item.kind == "finished":
                state.finished_blocks = item.blocks
                if state.monolithic and state.pushing:
                    self._cw_push_whole(item.req_id, state)
            elif item.error is not None:
                self._log_failure(
                    failure_type="push_handshake_failed",
                    req_id=item.req_id,
                    error=item.error,
                )
                self._cw_fail(state)
            else:
                self._cw_prepare(item.req_id, state)

    def _cw_handle_task(self, task: _ChunkTask) -> None:
        state = self._cw_writes.get(task.req_id)
        if state is None:
            state = self._cw_writes[task.req_id] = _ChunkwiseWrite(
                num_prompt_tokens=task.num_prompt_tokens
            )
            if (reg := self._pop_matching_registration(task.req_id)) is not None:
                with self._cw_guard(task.req_id, state):
                    self._cw_attach(task.req_id, state, reg)
        if state.pushing and not state.monolithic:
            # WRITTEN by _cw_progress once D is ready and the step computed.
            state.pending.append(task)

    def _cw_match_unregistered(self, request_id: str) -> str | None:
        """P request id of a chunk-wise push still waiting for *request_id*'s
        registration."""
        return match_request_id(
            request_id,
            [
                req_id
                for req_id, state in self._cw_writes.items()
                if state.awaiting_registration
            ],
        )

    def _cw_attach(self, req_id: str, state: _ChunkwiseWrite, reg: dict) -> None:
        """Pair a D registration with a chunk-wise push; handshake if needed."""
        state.reg = reg
        fut = self._ensure_handshake(
            reg["decode_engine_id"],
            reg["decode_host"],
            reg["decode_port"],
            reg["decode_tp_size"],
        )
        if fut is None:
            self._cw_prepare(req_id, state)
            return

        def _on_handshake(f: Future, rid: str = req_id) -> None:
            self._cw_inbox.put(_ChunkSignal(rid, "ready", f.exception()))
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _cw_prepare(self, req_id: str, state: _ChunkwiseWrite) -> None:
        """Resolve the WRITE geometry once D registered and is handshaked;
        _cw_progress then WRITEs the tasks.

        Chunk-wise WRITEs pair blocks by position. A D of other geometry gets
        the whole request in one WRITE once it finished instead.
        """
        if not state.pushing:
            return
        with self._cw_guard(req_id, state):
            assert self.transfer_topo is not None and state.reg is not None
            reg = state.reg
            engine_id = reg["decode_engine_id"]
            info = self.transfer_topo.get_engine_info(engine_id)
            plan = self.tp_mappings[engine_id]
            remote_block_ids = self._logical_to_kernel_block_ids(
                self._as_grouped_block_ids(reg["local_block_ids"]),
                info.remote_physical_blocks_per_logical,
            )
            ratio = self._physical_blocks_per_logical_kv_block
            num_local = cdiv(state.num_prompt_tokens, self.block_size * ratio) * ratio
            num_remote = max(
                (len(remote_block_ids[g]) for g in self._cw_attn_groups), default=0
            )
            self._engine_last_active[engine_id] = time.perf_counter()
            state.ready = True
            if reason := self._cw_geometry_mismatch(
                engine_id, info, plan, remote_block_ids
            ):
                logger.warning_once(
                    "NIXL chunk-wise push: requests to %s are pushed whole once "
                    "they finished (%s).",
                    engine_id,
                    reason,
                )
                self._cw_go_whole(req_id, state)
                return
            if num_remote > num_local:
                logger.debug(
                    "NIXL chunk-wise push: %s is pushed whole, D allocated %d "
                    "blocks for %d prompt blocks.",
                    req_id,
                    num_remote,
                    num_local,
                )
                self._cw_go_whole(req_id, state)
                return
            rank = plan.all_source_ranks[0]
            with self._handshake_lock:
                state.remote_agent = self._remote_agents[engine_id][(0, rank)]
            state.dst_engine_id = engine_id
            state.local_handle = self.src_xfer_handles_by_block_size[self.block_size]
            state.remote_handle = self.dst_xfer_side_handles[engine_id][rank]
            state.remote_physical_per_logical = info.remote_physical_blocks_per_logical
            state.notif = f"{reg['request_id']}:{self.world_size}".encode()
            state.remote_block_ids = remote_block_ids
            state.fa_offset = num_local - num_remote

    def _cw_geometry_mismatch(
        self, engine_id: str, info: Any, plan: Any, remote_block_ids: BlockIds
    ) -> str | None:
        """Why blocks can't be WRITTEN to *engine_id* by position, if so."""
        assert self.transfer_topo is not None
        if (tp_ratio := self.transfer_topo.tp_ratio(info.remote_tp_size)) != 1:
            return f"TP ratio {tp_ratio}"
        if info.remote_block_size != self.block_size:
            return f"block size {info.remote_block_size} vs {self.block_size}"
        ratio = self._physical_blocks_per_logical_kv_block
        if info.remote_physical_blocks_per_logical != ratio:
            return (
                f"{info.remote_physical_blocks_per_logical} kernel blocks per "
                f"block vs {ratio}"
            )
        if len(plan.all_source_ranks) != 1:
            return f"{len(plan.all_source_ranks)} source ranks"
        if not self._dst_pairs[engine_id].groups_match:
            return "different cache group layout"
        if len(remote_block_ids) != len(self._group_spec_types):
            return f"{len(remote_block_ids)} cache groups"
        return None

    def _cw_go_whole(self, req_id: str, state: _ChunkwiseWrite) -> None:
        state.monolithic = True
        state.pending.clear()
        if state.finished_blocks is not None:
            self._cw_push_whole(req_id, state)

    def _cw_push_whole(self, req_id: str, state: _ChunkwiseWrite) -> None:
        """Push a finished request whole, like a request without chunk-wise
        push: the scheduler's block table, under the request's lease."""
        assert state.reg is not None and state.finished_blocks is not None
        del self._cw_writes[req_id]
        self._do_start_push_kv(req_id, state.finished_blocks, state.reg)

    def _cw_final_due(self, task: _ChunkTask) -> bool:
        """The next step's metadata was bound: had it cancelled the request,
        _cw_write_tasks sees it (read after the step count, see
        bind_chunkwise)."""
        return self._cw_seq > task.seq

    def _cw_post_ready(self, req_id: str, state: _ChunkwiseWrite) -> None:
        """WRITE the queued tasks whose step is computed, in one WRITE. The
        final one, with D's notif, waits until it is due and the request's
        earlier WRITEs are done, so the notif implies all of the KV landed."""
        ready: list[_ChunkTask] = []
        for task in state.pending:
            if not task.event.query():
                break
            ready.append(task)
        if not ready:
            return
        notif = b""
        if ready[-1].final:
            if not state.handles and self._cw_final_due(ready[-1]):
                notif = state.notif
            else:
                ready.pop()
        if ready and self._cw_write_tasks(req_id, state, ready, notif=notif):
            for _ in ready:
                state.pending.popleft()
            if notif:
                state.final_posted = True

    def _cw_write_tasks(
        self,
        req_id: str,
        state: _ChunkwiseWrite,
        tasks: list[_ChunkTask],
        notif: bytes = b"",
    ) -> bool:
        """One WRITE of *tasks*' blocks, with D's *notif* for the final one
        (just the notif if nothing is left to write). Returns whether it was
        posted."""
        desc_ids = [ids for task in tasks if (ids := self._cw_desc_ids(state, task))]
        if notif:
            # Before the cancellation check, see bind_chunkwise.
            self._cw_final_inflight.add(req_id)
        try:
            if self._push_writer_stop.is_set() or req_id in self._cw_cancelled:
                return False
            if desc_ids:
                return self._cw_write(
                    req_id,
                    state,
                    np.concatenate([local for local, _ in desc_ids]),
                    np.concatenate([remote for _, remote in desc_ids]),
                    notif=notif,
                )
            if notif:
                try:
                    self.nixl_wrapper.send_notif(state.remote_agent, notif_msg=notif)
                except Exception as e:
                    self._log_failure(
                        failure_type="notification_failed", req_id=req_id, error=e
                    )
                    self.xfer_stats.record_failed_notification()
                    self._cw_fail(state)
                    return False
            return True
        finally:
            if not state.handles:
                self._cw_final_inflight.discard(req_id)

    def _cw_desc_ids(
        self, state: _ChunkwiseWrite, task: _ChunkTask
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Local/remote descriptor ids of the blocks *task* WRITEs."""
        offset = state.fa_offset
        ssm = [_is_ssm_spec(spec_type) for spec_type in self._group_spec_types]
        local_ids: list[list[int]] = []
        remote_ids: list[list[int]] = []
        for g, is_ssm in enumerate(ssm):
            if is_ssm:
                local_ids.append([])
                remote_ids.append([])
                continue
            # D allocated only the suffix it has not cached.
            local = task.kernel_block_ids[g]
            remote = state.remote_block_ids[g]
            lo = max(task.start, offset)
            hi = max(lo, task.start + len(local))
            local_ids.append(list(local[lo - task.start : hi - task.start]))
            remote_ids.append(list(remote[lo - offset : hi - offset]))
        if task.final and any(ssm):
            # Pair the Mamba state slots like a whole-request WRITE does.
            remote_ssm, local_ssm = self._apply_prefix_caching(
                decode_block_ids=[
                    list(blocks) if is_ssm else []
                    for blocks, is_ssm in zip(state.remote_block_ids, ssm)
                ],
                prefill_block_ids=[
                    list(blocks) if is_ssm else []
                    for blocks, is_ssm in zip(task.kernel_block_ids, ssm)
                ],
                decode_physical_per_logical=state.remote_physical_per_logical,
                prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
            )
            for g, is_ssm in enumerate(ssm):
                if is_ssm:
                    local_ids[g] = list(local_ssm[g])
                    remote_ids[g] = list(remote_ssm[g])
        for g, (local, remote) in enumerate(zip(local_ids, remote_ids)):
            if len(local) != len(remote):
                raise ValueError(
                    f"chunk-wise push block mismatch for {task.req_id} group {g}: "
                    f"{len(local)} local vs {len(remote)} remote blocks"
                )
        if not any(local_ids):
            return None
        # Same block size as D (see _cw_geometry_mismatch).
        return self._write_desc_ids(
            local_ids,
            remote_ids,
            state.dst_engine_id,
            remote_physical_per_logical=state.remote_physical_per_logical,
            block_size_ratio=1,
        )

    def _cw_write(
        self,
        req_id: str,
        state: _ChunkwiseWrite,
        local_descs: np.ndarray,
        remote_descs: np.ndarray,
        notif: bytes = b"",
    ) -> bool:
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                state.local_handle,
                local_descs,
                state.remote_handle,
                remote_descs,
                notif_msg=notif,
            )
            self.nixl_wrapper.transfer(handle)
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=req_id,
                msg="Chunk-wise push WRITE submission failed",
                error=e,
                dst_engine_id=state.dst_engine_id,
            )
            self._cw_fail(state)
            if not self._handle_failed_transfer(req_id, handle):
                # Only a submitted transfer can need retaining.
                assert handle is not None
                state.handles.append(handle)
            return False
        state.handles.append(handle)
        return True

    def _cw_fail(self, state: _ChunkwiseWrite) -> None:
        """Stop pushing; D is not notified and P frees the blocks when the
        lease expires (same as a failed whole-request push). The state stays
        until dropped, to absorb the request's later tasks."""
        state.failed = True
        state.pending.clear()

    def _cw_cancel(
        self, req_id: str, kind: str, fence: threading.Event | None = None
    ) -> None:
        """Stop a request's chunk-wise push: it was preempted ("fallback": it
        is pushed whole once it finished), or aborted or its lease is gone
        ("drop"). Its state goes once its WRITEs are done, which is when
        *fence* is set (see bind_chunkwise)."""
        self._cw_cancelled.discard(req_id)
        state = self._cw_writes.get(req_id)
        if state is None:
            if fence is not None:
                fence.set()
            return
        if state.cancelled != "drop":
            state.cancelled = kind
        state.pending.clear()
        if fence is not None:
            state.fences.append(fence)

    def _cw_progress(self) -> None:
        """Poll the in-flight WRITEs (the UCX backend only progresses a
        transfer, and sends its notif, when it is checked), WRITE the tasks
        that are ready and retire the requests that are done. Never blocks."""
        in_flight = {
            req_id: state.handles
            for req_id, state in self._cw_writes.items()
            if state.handles
        }
        if in_flight:
            _, failed = self._pop_done_transfers(in_flight)
            for req_id, state in self._cw_writes.items():
                if state.handles:
                    state.handles = in_flight.get(req_id, [])
                if req_id in failed:
                    self._cw_fail(state)
        for req_id, state in self._cw_writes.items():
            if not state.handles:
                self._cw_final_inflight.discard(req_id)
            if state.ready and state.pushing and not state.monolithic:
                with self._cw_guard(req_id, state):
                    self._cw_post_ready(req_id, state)
        self._cw_retire()

    def _cw_retire(self) -> None:
        """Once their WRITEs are done: report the completed pushes, release
        the fences of the cancelled ones and hand a preempted request's
        registration to its whole-request push."""
        for req_id, state in list(self._cw_writes.items()):
            if state.handles:
                continue
            if state.cancelled is None:
                if state.final_posted and not state.failed:
                    del self._cw_writes[req_id]
                    self._cw_done.put(req_id)
                continue
            del self._cw_writes[req_id]
            for fence in state.fences:
                fence.set()
            if state.cancelled == "drop" or state.reg is None:
                continue
            if state.final_posted:
                # D was notified: a second push could overwrite blocks it uses.
                logger.warning(
                    "NIXL chunk-wise push: %s was preempted after its push; "
                    "not pushing it again.",
                    req_id,
                )
            else:
                self._route_push_registration(state.reg)

    def _cw_needs_poll(self) -> bool:
        """In-flight WRITEs to check, or a PUSH_REG to wait for: progress only
        polling reveals."""
        return any(
            state.handles or state.awaiting_registration
            for state in self._cw_writes.values()
        )

    def _cw_next_event(self) -> "torch.cuda.Event | None":
        """The event of the oldest step a WRITE waits for, if any."""
        oldest: _ChunkTask | None = None
        for state in self._cw_writes.values():
            if not (state.ready and state.pushing and state.pending):
                continue
            task = state.pending[0]
            if (oldest is None or task.seq < oldest.seq) and not task.event.query():
                oldest = task
        return None if oldest is None else oldest.event

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. ``send_notif`` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto ``_reg_send_inbox`` once it
        completes (at which point ``_ensure_handshake`` returns ``None`` and we
        send directly)."""
        remote_pp_size = reg_data.get("remote_pp_size", 1)
        fut = self._ensure_handshake(
            reg_data["remote_engine_id"],
            reg_data["remote_host"],
            reg_data["remote_port"],
            reg_data["remote_tp_size"],
            pp_size=remote_pp_size,
            # D only ever sends PUSH_REG notifs to P and never reads or writes
            # P's memory in push mode, so it never needs the transfer
            # descriptors set up by the full add_remote_agent path.
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[tuple[dict[tuple[int, int], str], float]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._failed_recv_reqs.put(rid)
                return
            # Re-queue for the writer to send now that the handshake is done.
            self._reg_send_inbox.put((rid, rd))
            # Wake the writer so it sends the PUSH_REG promptly even if
            # otherwise parked.
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        # _remote_agents is mutated on other threads; snapshot under the lock.
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )
                self.xfer_stats.record_failed_notification()
                # Earlier registrations may still trigger WRITEs into D's blocks.
                # Keep the receive pending until those writes are finished.
        logger.debug(
            "Sent PUSH_REG for %s to engine %s (%dB)", req_id, engine_id, len(notif_msg)
        )

    # --- Matching helpers --------------------------------------------- #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Pop the D-side registration matching *request_id*.

        Exact key first, then a match after stripping the random suffix from
        both sides. No match leaves the request unmatched (push not started).
        """
        reg_id = match_request_id(request_id, self._pending_d_registrations)
        if reg_id is None:
            return None
        return self._pending_d_registrations.pop(reg_id)

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Pop the P-side finished blocks matching *request_id*.

        Same lookup as ``_pop_matching_registration``: exact key, then a
        match after stripping the random suffix from both sides.
        """
        fin_id = match_request_id(request_id, self._push_finished_blocks)
        if fin_id is None:
            return None
        return fin_id, self._push_finished_blocks.pop(fin_id)

    # --- WRITE transfer logic (writer thread) ------------------------- #

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node.

        The P→D handshake runs on the base worker's background executor.
        If it isn't ready yet we register a completion callback, defer the
        WRITE, and re-drive this request via ``_deferred_push_inbox`` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors ``_send_registration_to_p``).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]

        # Runs on the background executor; defer the WRITE until it's ready.
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                rid: str = request_id,
                blocks: BlockIds = local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (e := f.exception()) is not None:
                    # The engine reclaims the blocks via the TTL so we dont free here
                    self._log_failure(
                        failure_type="push_handshake_failed", req_id=rid, error=e
                    )
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return
        # Keep the engine alive while it is actively receiving pushes, mirroring
        # how pull-mode transfers touch _engine_last_active in start_load_kv.
        self._engine_last_active[decode_engine_id] = time.perf_counter()

        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )

        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        ``BlockIds`` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_do_start_push_kv`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        if not self._dst_pairs[engine_id].groups_match:
            raise NotImplementedError(
                "NixlPushConnector does not support different producer and "
                "consumer cache-group layouts"
            )

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)

        num_groups = len(local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (self._is_csa_linear and tp_ratio < 0)
                or (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        handles: list[int] = []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)

    def _write_desc_ids(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        dst_engine_id: str,
        remote_physical_per_logical: int,
        block_size_ratio: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local / remote descriptor ids of a WRITE of the kernel blocks
        *local_block_ids* into *remote_block_ids*, paired by position. Pair k
        writes local region pairs.local[k] into remote region pairs.remote[k]
        at its cache group's blocks."""
        pairs = self._dst_pairs[dst_engine_id]
        local_descs = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.dst_region_num_blocks[self.engine_id],
            pairs=(pairs.local, pairs.group),
        )
        remote_descs = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_physical_per_logical,
            region_num_blocks=self.dst_region_num_blocks[dst_engine_id],
            pairs=(pairs.remote, pairs.group),
            ssm_regions=pairs.remote_ssm_regions,
        )
        assert len(local_descs) == len(remote_descs)
        return local_descs, remote_descs

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or ``None`` if nothing was submitted.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )

        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Prefix caching: D allocated only uncached blocks, so on a partial hit it
        # sends fewer than P's. End-trim P's blocks to that same suffix so we WRITE only
        # the uncomputed tail into D's slots. Runs on kernel ids, post-expansion.
        remote_block_ids, local_block_ids = self._apply_prefix_caching(
            decode_block_ids=remote_block_ids,
            prefill_block_ids=local_block_ids,
            decode_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        assert len(local_block_ids) == len(remote_block_ids), (
            f"push group-count mismatch for {request_id}: {len(local_block_ids)} "
            f"local vs {len(remote_block_ids)} remote groups"
        )
        for i in range(len(local_block_ids)):
            assert len(local_block_ids[i]) == len(remote_block_ids[i]), (
                f"push block-count mismatch for {request_id} group {i}: "
                f"{len(local_block_ids[i])} local vs "
                f"{len(remote_block_ids[i])} remote blocks"
            )

        local_block_descs_ids, remote_block_descs_ids = self._write_desc_ids(
            local_block_ids,
            remote_block_ids,
            dst_engine_id,
            remote_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            block_size_ratio=block_size_ratio,
        )

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; we
            # don't have a ``_recving_metadata`` entry to invalidate, so
            # we just release the handle and let the engine reschedule
            # via the lease / watchdog.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    def _pop_chunkwise_done(self) -> set[str]:
        """Chunk-wise pushes to report as sent. A push usually completes
        before the scheduler finished its request, so it is only reported once
        the request's lease (sent on finish) has reached this worker."""
        while True:
            try:
                self._cw_completed.add(self._cw_done.get_nowait())
            except queue.Empty:
                break
        # Expired leases were already reported; aborted requests get none.
        self._cw_completed &= self._reqs_to_process
        done = {req_id for req_id in self._cw_completed if req_id in self._reqs_to_send}
        self._cw_completed -= done
        return done

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns ``nixl_wrapper.get_new_notifs`` for push; PUSH_REG
        notifs are handled there. P learns of finished pushes from its WRITE
        handles, so this never reports finished sends.
        """
        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            req_id, tp_size = msg.rsplit(":", 1)

            if (meta := self._recving_metadata.get(req_id)) is None:
                # The lease may have expired before the notif arrived.
                self.xfer_stats.record_notification_after_expiry()
                logger.error(
                    "Unrecognized request %s notif (may have expired).", req_id
                )
                continue
            # Consumer waits for one notif per producer rank writing
            # here: pp_size stages * producers-per-consumer (>1 when
            # producer TP > consumer TP; tp_size is the producer TP).
            producers_per_consumer = max(1, int(tp_size) // self.world_size)
            expected_notifs = meta.pp_size * producers_per_consumer
            self.consumer_notification_counts_by_req[req_id] += 1
            if self.consumer_notification_counts_by_req[req_id] < expected_notifs:
                continue
            del self.consumer_notification_counts_by_req[req_id]
            # P drove the transfer (we own no NIXL handle), so
            # materialise an empty ``_recving_transfers`` entry for
            # ``_pop_done_transfers`` to report done.
            self._recving_transfers.setdefault(req_id, [])
        return set()

    def get_transfer_results(self) -> KVConnectorTransferResults:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        results = super().get_transfer_results()
        done_sending = results.finished_sending

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            if self._writer_progress:
                done_pushing, failed_pushing = set(), set()
                while True:
                    try:
                        done, failed = self._sending_results.get_nowait()
                    except queue.Empty:
                        break
                    done_pushing |= done
                    failed_pushing |= failed
            else:
                done_pushing, failed_pushing = self._pop_done_transfers(
                    self._sending_transfers
                )
            # Remember failures until the final sibling WRITE completes.
            self._send_failures.update(failed_pushing)
            successful = {
                req_id
                for req_id in done_pushing - self._send_failures
                if req_id in self._reqs_to_send or req_id in self._reqs_to_process
            }
            self._send_failures.difference_update(done_pushing | done_sending)
        # Expired requests were already reported, even if their WRITEs finish later.
        done_pushing = successful
        if self.chunkwise_push:
            done_pushing |= self._pop_chunkwise_done()
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results
