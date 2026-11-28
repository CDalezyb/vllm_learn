# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for swapping IR op providers without a restart."""

import pytest
import torch

from vllm import ir
from vllm.dev import ops as dev_ops

PROVIDER = "dev_test_provider"

VARIANT_MODULE = """
import torch
from vllm import ir


@ir.ops.rms_norm.register_impl("dev_test_provider")
def impl(x, weight, epsilon, variance_size=None):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    out = (x.float() * torch.rsqrt(variance + epsilon)).to(x.dtype)
    return out if weight is None else out * weight
"""


@pytest.fixture(autouse=True)
def restore_priority():
    """Leave every op on the priority it had before the test."""
    saved = {name: op.get_priority() for name, op in dev_ops.iter_ops().items()}
    yield
    for name, priority in saved.items():
        dev_ops.get_op(name).set_default(priority or ["native"])


def write_variant_module(tmp_path) -> str:
    module = tmp_path / "variants.py"
    module.write_text(VARIANT_MODULE)
    return str(module)


def test_iter_ops_finds_layernorm_ops():
    assert {"rms_norm", "fused_add_rms_norm"} <= set(dev_ops.iter_ops())


def test_import_module_registers_provider(tmp_path):
    name, sha256 = dev_ops.import_module(write_variant_module(tmp_path))

    assert name.startswith("vllm_dev_ops_variants_")
    assert len(sha256) == 64
    assert PROVIDER in dev_ops.get_op("rms_norm").impls


def test_swapped_provider_computes_the_native_values(tmp_path):
    dev_ops.import_module(write_variant_module(tmp_path))
    installed = dev_ops.set_priority({"rms_norm": [PROVIDER, "native"]})

    assert installed["rms_norm"] == [PROVIDER]
    x = torch.randn(4, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    swapped = ir.ops.rms_norm(x, weight, 1e-6)
    native = dev_ops.get_op("rms_norm").impls["native"].impl_fn(x, weight, 1e-6)
    torch.testing.assert_close(swapped, native)


def test_set_priority_rejects_unknown_provider():
    with pytest.raises(ValueError, match="has no provider"):
        dev_ops.set_priority({"rms_norm": ["not_registered"]})


def test_unknown_op_is_rejected():
    with pytest.raises(KeyError, match="unknown IR op"):
        dev_ops.get_op("not_an_op")
