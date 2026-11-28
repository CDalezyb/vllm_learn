# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dev endpoints that swap a plank of a running engine.

Swapping has two halves that must not drift apart: installing the change on
every worker, and making the captured CUDA graphs run the new code. Both are
reported back so a caller can record which engine it measured.
"""

import json
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.dev.state import dev_enabled
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _require_enabled() -> None:
    if not dev_enabled():
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail="VLLM_DEV_HOTSWAP is not enabled",
        )


async def _body(raw_request: Request) -> dict:
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request body must be a JSON object",
        )
    return body


@router.get("/dev/hotswap/state")
async def state(raw_request: Request) -> JSONResponse:
    """What every worker is running: swap epoch, op priority, op modules."""
    _require_enabled()
    identities = await engine_client(raw_request).collective_rpc("dev_identity")
    return JSONResponse(content={"workers": identities})


@router.post("/dev/hotswap/sched")
async def sched(raw_request: Request) -> JSONResponse:
    """Change the scheduler token budget within the started envelope."""
    _require_enabled()
    body = await _body(raw_request)
    budget = body.get("token_budget")
    if not isinstance(budget, int):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="token_budget must be an integer",
        )
    result = await engine_client(raw_request).dev_set_token_budget(budget)
    return JSONResponse(content=result)


@router.post("/dev/hotswap/ops")
async def ops(raw_request: Request) -> JSONResponse:
    """Import an op module and put its providers first on every worker.

    Body keys: `module` (optional path imported first) and `priority` (op name
    to provider list). Captured graphs are left stale on purpose: the reply
    says so, and replaying them raises until `/dev/hotswap/recapture` runs.
    """
    _require_enabled()
    body = await _body(raw_request)
    payload = json.dumps(
        {"module": body.get("module"), "priority": body.get("priority") or {}}
    )
    results = await engine_client(raw_request).collective_rpc(
        "dev_apply_ops", args=(payload,)
    )
    return JSONResponse(
        content={"workers": results, "graphs_stale": True},
    )


@router.post("/dev/hotswap/recapture")
async def recapture(raw_request: Request) -> JSONResponse:
    """Re-capture the decode graphs so they run the ops in force now."""
    _require_enabled()
    results = await engine_client(raw_request).collective_rpc(
        "dev_recapture_cudagraphs"
    )
    return JSONResponse(content={"workers": results})


@router.post("/dev/hotswap/apply")
async def apply(raw_request: Request) -> JSONResponse:
    """Swap ops and re-capture the graphs in one call.

    Body: `module`, `priority`, and `recapture` (default true).
    """
    _require_enabled()
    body = await _body(raw_request)
    payload = json.dumps(
        {"module": body.get("module"), "priority": body.get("priority") or {}}
    )
    client = engine_client(raw_request)
    swapped = await client.collective_rpc("dev_apply_ops", args=(payload,))
    recaptured = None
    if body.get("recapture", True):
        recaptured = await client.collective_rpc("dev_recapture_cudagraphs")
    return JSONResponse(content={"ops": swapped, "recapture": recaptured})


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
