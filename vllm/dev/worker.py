# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side entry points for a plank swap.

These run inside the worker process through `collective_rpc`, so they see the
same op registry and CUDA graphs the model forward uses. Arguments arrive as
JSON strings because the dev RPC endpoint only forwards strings.
"""

import json
from pathlib import Path
from typing import Any

from vllm.dev import ops as dev_ops
from vllm.dev.state import bump_epoch, dev_state, dev_enabled


def _require_enabled() -> None:
    if not dev_enabled():
        raise RuntimeError("VLLM_DEV_HOTSWAP is not enabled in this worker")


def apply_ops(payload_json: str) -> dict[str, Any]:
    """Import an op module and install its dispatch priority.

    Args:
        payload_json: JSON object with optional keys `module` (path to a
            Python file to import before selecting providers) and `priority`
            (op name to provider list).

    Returns:
        The installed priority, the swap epoch, and the module hash.
    """
    _require_enabled()
    payload = json.loads(payload_json)
    module_path = payload.get("module")
    priority = payload.get("priority") or {}

    module_name = None
    module_sha256 = None
    if module_path:
        module_name, module_sha256 = dev_ops.import_module(module_path)
        dev_state().module_sha256[str(Path(module_path).expanduser().resolve())] = (
            module_sha256
        )

    installed = dev_ops.set_priority(priority) if priority else {}
    epoch = bump_epoch(f"ops {sorted(installed)}")
    return {
        "epoch": epoch,
        "installed": installed,
        "module": module_name,
        "module_sha256": module_sha256,
        "ops": dev_ops.describe(sorted(installed)),
    }


def identity() -> dict[str, Any]:
    """What this worker is currently running."""
    _require_enabled()
    return {
        "epoch": dev_state().epoch,
        "priority": dev_ops.priority_signature(),
        "modules": dict(dev_state().module_sha256),
    }
