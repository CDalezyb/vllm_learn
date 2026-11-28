# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-process bookkeeping for plank swaps.

The epoch is the load-bearing part. A CUDA graph replays recorded kernel
launches and never re-enters Python, so swapping an op implementation cannot
affect an already captured graph. Every swap bumps the epoch; a capture records
the epoch it was made under; a replay under a different epoch raises instead of
silently measuring the old kernel.
"""

import os
from dataclasses import dataclass, field
from typing import Any


def dev_enabled() -> bool:
    """Whether dev plank swapping is enabled for this process."""
    return os.getenv("VLLM_DEV_HOTSWAP", "0").lower() not in ("", "0", "false", "no")


@dataclass
class DevState:
    #: Bumped on every swap that can change what a captured graph runs.
    epoch: int = 0
    #: op name -> provider priority that was last installed.
    op_priority: dict[str, list[str]] = field(default_factory=dict)
    #: sha256 of the last op module imported, per module path.
    module_sha256: dict[str, str] = field(default_factory=dict)
    #: Human-readable log of swaps, newest last. Bounded by ``max_log``.
    swaps: list[dict[str, Any]] = field(default_factory=list)
    max_log: int = 64

    def record(self, **entry: Any) -> None:
        self.swaps.append(entry)
        if len(self.swaps) > self.max_log:
            del self.swaps[: -self.max_log]


_STATE = DevState()


def dev_state() -> DevState:
    """The process-wide dev state."""
    return _STATE


def bump_epoch(reason: str) -> int:
    """Bump the swap epoch and record why."""
    _STATE.epoch += 1
    _STATE.record(event="epoch", epoch=_STATE.epoch, reason=reason)
    return _STATE.epoch
