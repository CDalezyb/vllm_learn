# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Development-time swapping of engine planks on a live engine.

A *plank* is a piece of the engine an agent or an engineer iterates on: a
scheduler budget, an operator implementation, or (through a restart) a whole
worker. Swapping a plank must be much cheaper than restarting the engine, and
it must never leave the engine running code it does not report.

Everything here is gated behind ``VLLM_DEV_HOTSWAP=1`` and is meant for a
development host: the control endpoints are unauthenticated and can run
arbitrary Python from a path the caller names.
"""

from vllm.dev.state import DevState, bump_epoch, dev_enabled, dev_state

__all__ = ["DevState", "bump_epoch", "dev_enabled", "dev_state"]
