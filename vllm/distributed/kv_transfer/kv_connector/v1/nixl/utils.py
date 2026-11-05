# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import regex as re
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
    is_full_attention_spec,
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


def chunked_push_unsupported_reason(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    use_host_buffer: bool,
) -> str | None:
    """Why chunked push cannot run with this configuration, or ``None`` if it
    can. The scheduler and the workers both decide with this function, so they
    always agree on whether chunked push is enabled."""
    assert vllm_config.kv_transfer_config is not None
    kv_role = vllm_config.kv_transfer_config.kv_role
    if kv_role != "kv_producer":
        return f"kv_role is {kv_role!r}, but chunked push runs only on a kv_producer"
    if use_host_buffer:
        return (
            "the KV buffer is on the host (kv_buffer_device='cpu'), but chunked "
            "push needs a device KV buffer"
        )
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size != 1:
        return (
            f"pipeline_parallel_size is {parallel.pipeline_parallel_size}, but "
            "chunked push needs 1"
        )
    if parallel.prefill_context_parallel_size != 1:
        return (
            "prefill_context_parallel_size is "
            f"{parallel.prefill_context_parallel_size}, but chunked push needs 1"
        )
    groups = kv_cache_config.transfer_groups
    if len(groups) != 1 or not is_full_attention_spec(groups[0].kv_cache_spec):
        kinds = ", ".join(type(group.kv_cache_spec).__name__ for group in groups)
        return (
            f"the KV cache transfer groups are [{kinds}], but chunked push "
            "supports a single full-attention (or MLA) group"
        )
    return None
