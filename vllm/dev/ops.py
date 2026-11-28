# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registering and selecting operator implementations at run time.

The IR op registry already models "one op, several providers, one dispatch
priority". Swapping an operator is therefore two steps: import a module that
calls ``register_impl``, then install a priority that puts the new provider
first. Nothing here parses or compiles anything; the swapped-in module is
expected to be prebuilt for the running process (a Python module, a Triton
kernel whose cache is warm, or a CUDA extension already loaded).
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from vllm import ir
from vllm.dev.state import dev_state
from vllm.ir.op import IrOp
from vllm.logger import init_logger

logger = init_logger(__name__)


def iter_ops() -> dict[str, IrOp]:
    """Every IR op that can be swapped, by name."""
    found: dict[str, IrOp] = {}
    for name in dir(ir.ops):
        candidate = getattr(ir.ops, name)
        if isinstance(candidate, IrOp):
            found[name] = candidate
    return found


def get_op(name: str) -> IrOp:
    """Resolve one op by name.

    Args:
        name: Attribute name under `vllm.ir.ops`, e.g. `rms_norm`.

    Raises:
        KeyError: The name is not an IR op.
    """
    op = getattr(ir.ops, name, None)
    if not isinstance(op, IrOp):
        raise KeyError(f"unknown IR op {name!r}; known: {sorted(iter_ops())}")
    return op


def describe(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Providers and active priority for the named ops, or all of them."""
    ops = iter_ops()
    selected = sorted(ops) if names is None else sorted(names)
    described: dict[str, dict[str, Any]] = {}
    for name in selected:
        op = ops.get(name)
        if op is None:
            continue
        described[name] = {
            "providers": sorted(op.impls),
            "priority": op.get_priority(),
        }
    return described


def import_module(path: str) -> tuple[str, str]:
    """Import a file as a module so it can register impls.

    The module is imported under a name derived from its content hash, so a
    second swap of the same file is a no-op while an edited file is a fresh
    import.

    Args:
        path: File to import.

    Returns:
        The module name and the sha256 of the file.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"no such op module: {resolved}")
    data = resolved.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    name = f"vllm_dev_ops_{resolved.stem}_{sha256[:12]}"
    if name in sys.modules:
        return name, sha256
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    dev_state().module_sha256[str(resolved)] = sha256
    logger.info("dev hotswap: imported op module %s (%s)", resolved, sha256[:12])
    return name, sha256


def set_priority(priority: dict[str, list[str]]) -> dict[str, list[str]]:
    """Install a dispatch priority per op.

    Args:
        priority: Op name to the providers to try first, in order. `native`
            should end the list unless the last provider handles every
            argument shape.

    Returns:
        The priority actually installed per op, after unsupported providers
        were filtered out.

    Raises:
        KeyError: An op name is not an IR op.
        AssertionError: A provider is not registered for its op.
    """
    installed: dict[str, list[str]] = {}
    for name, providers in priority.items():
        op = get_op(name)
        op.set_default(list(providers))
        installed[name] = op.get_priority()
        dev_state().op_priority[name] = installed[name]
    return installed


def priority_signature() -> dict[str, list[str]]:
    """The installed priority of every op that is not on `native` alone."""
    signature = {}
    for name, op in iter_ops().items():
        priority = op.get_priority()
        if priority and priority != ["native"]:
            signature[name] = priority
    return signature
