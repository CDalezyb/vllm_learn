# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import math
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Any

import regex as re
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket."""
    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)


def match_request_id(request_id: str, candidates: Collection[str]) -> str | None:
    """The candidate equal to *request_id*, else the first one equal to it
    once both lose their random suffix (see get_base_request_id): P and D get
    the same request id from the proxy, and each engine appends its own."""
    if request_id in candidates:
        return request_id
    base_id = get_base_request_id(request_id)
    return next(
        (req_id for req_id in candidates if get_base_request_id(req_id) == base_id),
        None,
    )


def transfer_block_size(kv_cache_config: "KVCacheConfig", default: int) -> int:
    """Logical block size of NIXL transfers: the lcm of the transfer groups'
    block sizes, Mamba and circular-buffer groups aside."""
    block_sizes = [
        group.kv_cache_spec.block_size
        for group in kv_cache_config.transfer_groups
        if get_representative_spec_type(group.kv_cache_spec)
        not in (MambaSpec, CircularBufferSpec)
    ]
    return math.lcm(*block_sizes) if block_sizes else default


# Cache group types whose KV can be WRITTEN chunk by chunk: token-indexed full
# attention, and Mamba state (sent once, with the last prefill chunk).
_CHUNKWISE_SPEC_TYPES = (FullAttentionSpec, MLAAttentionSpec, MambaSpec)


def chunkwise_push_unsupported_reason(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> str | None:
    """Why chunk-wise push can't run with this config (None if it can)."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    parallel_config = vllm_config.parallel_config
    if not kv_transfer_config.is_kv_producer:
        return "it only applies to the prefill (kv_producer) instance"
    if kv_transfer_config.kv_buffer_device == "cpu":
        return "host-buffer KV transfer is not supported"
    if kv_transfer_config.kv_load_failure_policy == "recompute":
        # Recomputing a request in place would leave its pushed chunks stale.
        return "kv_load_failure_policy=recompute is not supported"
    if parallel_config.pipeline_parallel_size > 1:
        return "pipeline parallelism is not supported"
    if (
        parallel_config.prefill_context_parallel_size > 1
        or parallel_config.decode_context_parallel_size > 1
    ):
        return "context parallelism is not supported"
    attn_block_sizes = set()
    for group in kv_cache_config.transfer_groups:
        spec_type = get_representative_spec_type(group.kv_cache_spec)
        if spec_type not in _CHUNKWISE_SPEC_TYPES or any(
            getattr(spec, "tp_replicated", False)
            for spec in iter_layer_specs(group.kv_cache_spec)
        ):
            return f"KV cache group type {spec_type.__name__} is not supported"
        if spec_type is not MambaSpec:
            attn_block_sizes.add(group.kv_cache_spec.block_size)
    if len(attn_block_sizes) > 1:
        return "attention groups with different block sizes are not supported"
    return None
