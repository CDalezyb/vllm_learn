# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command line for swapping planks on a running dev engine.

Examples:
    python -m vllm.dev.cli state
    python -m vllm.dev.cli sched 4096
    python -m vllm.dev.cli apply --module vllm/dev/examples/rms_norm_variants.py \\
        --op rms_norm --provider dev_twice
    python -m vllm.dev.cli verify --repeat 3
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


def request(base: str, path: str, payload: dict | None = None, timeout: float = 600):
    url = base.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path}: HTTP {e.code}: {e.read().decode()}") from e
    return json.loads(body) if body else {}


def model_id(base: str) -> str:
    return request(base, "/v1/models")["data"][0]["id"]


def complete(base: str, prompt: str, max_tokens: int) -> tuple[list[int], float]:
    payload = {
        "model": model_id(base),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    start = time.perf_counter()
    body = request(base, "/v1/completions", payload)
    elapsed = time.perf_counter() - start
    choice = body["choices"][0]
    token_ids = choice.get("token_ids")
    if token_ids is None:
        token_ids = [ord(c) for c in choice["text"]]
    return token_ids, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vllm.dev.cli")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    subs = parser.add_subparsers(dest="cmd", required=True)

    subs.add_parser("state", help="op priority and swap epoch per worker")

    sched = subs.add_parser("sched", help="set the scheduler token budget")
    sched.add_argument("budget", type=int)

    ops = subs.add_parser("apply", help="swap an op provider and re-capture")
    ops.add_argument("--module", help="Python file to import before selecting")
    ops.add_argument("--op", default="rms_norm")
    ops.add_argument("--provider", required=True)
    ops.add_argument("--native-fallback", action="store_true", default=True)
    ops.add_argument("--no-recapture", action="store_true")

    subs.add_parser("recapture", help="re-capture decode graphs only")

    verify = subs.add_parser("verify", help="greedy run: tokens and time")
    verify.add_argument("--prompt", default="The capital of France is")
    verify.add_argument("--max-tokens", type=int, default=32)
    verify.add_argument("--repeat", type=int, default=3)

    args = parser.parse_args(argv)
    base = args.base

    if args.cmd == "state":
        print(json.dumps(request(base, "/dev/hotswap/state"), indent=2))
        return 0

    if args.cmd == "sched":
        start = time.perf_counter()
        result = request(base, "/dev/hotswap/sched", {"token_budget": args.budget})
        result["elapsed_s"] = round(time.perf_counter() - start, 3)
        print(json.dumps(result))
        return 0

    if args.cmd == "recapture":
        start = time.perf_counter()
        result = request(base, "/dev/hotswap/recapture", {})
        result["elapsed_s"] = round(time.perf_counter() - start, 3)
        print(json.dumps(result))
        return 0

    if args.cmd == "apply":
        priority = [args.provider]
        if args.native_fallback:
            priority.append("native")
        payload = {
            "module": args.module,
            "priority": {args.op: priority},
            "recapture": not args.no_recapture,
        }
        start = time.perf_counter()
        result = request(base, "/dev/hotswap/apply", payload)
        result["elapsed_s"] = round(time.perf_counter() - start, 3)
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "verify":
        runs = []
        for _ in range(args.repeat):
            tokens, elapsed = complete(base, args.prompt, args.max_tokens)
            runs.append(
                {
                    "digest": hashlib.sha256(
                        ",".join(map(str, tokens)).encode()
                    ).hexdigest()[:16],
                    "seconds": round(elapsed, 3),
                    "tokens_per_s": round(len(tokens) / elapsed, 1),
                }
            )
        digests = {run["digest"] for run in runs}
        times = sorted(run["seconds"] for run in runs)
        print(
            json.dumps(
                {
                    "runs": runs,
                    "identical": len(digests) == 1,
                    "median_seconds": times[len(times) // 2],
                    "median_tokens_per_s": sorted(
                        run["tokens_per_s"] for run in runs
                    )[len(runs) // 2],
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
