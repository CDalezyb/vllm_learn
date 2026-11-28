# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample op module for a plank swap.

Each provider below computes the same values as the native `rms_norm`; they
differ only in which kernels do the work. `dev_twice` is deliberately slower,
so a swap to it must show up as a slower decode step with identical tokens.
That pair is what makes a swap check meaningful: the tokens prove the math did
not change, the time proves the swap reached the measured path.
"""

import torch

from vllm import ir


@ir.ops.rms_norm.register_impl("dev_torch")
def torch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    if variance_size is not None:
        variance = x[..., :variance_size].float().pow(2).mean(
            dim=-1, keepdim=True
        )
    out = (x.float() * torch.rsqrt(variance + epsilon)).to(x.dtype)
    if weight is not None:
        out = out * weight
    return out


@ir.ops.rms_norm.register_impl("dev_twice")
def twice_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    """The native impl run twice; same output, roughly double the time."""
    out = ir.ops.rms_norm.impls["native"].impl_fn(
        x, weight, epsilon, variance_size
    )
    return ir.ops.rms_norm.impls["native"].impl_fn(
        x, weight, epsilon, variance_size
    ) * 0 + out
