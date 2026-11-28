# Dev-time plank swapping

Plank swapping changes one piece of a running engine — a scheduler budget or an
operator implementation — without restarting it. It exists to shorten the loop
"change a thing, measure it" for development hosts, not for serving: the
endpoints run code from a path the caller names and are unauthenticated.

It is gated twice. The routes are attached only with `VLLM_SERVER_DEV_MODE=1`,
and every handler answers only when `VLLM_DEV_HOTSWAP=1` is set in the engine's
environment.

## What a swap costs

A restart pays for weight loading, KV cache allocation, CUDA graph capture and
interpreter start. A scheduled budget swap pays for none of them: it changes a
value the next scheduling step reads. An operator swap pays for re-capturing the
decode graphs, because a replay runs the launches recorded at capture time and
never re-enters Python.

| Plank | What runs again | Typical cost |
| --- | --- | --- |
| Scheduler token budget | one scheduling step | milliseconds |
| Operator provider (prefill) | nothing; prefill is eager by default | under a second |
| Operator provider (decode) | decode graph capture | seconds |

## Scheduler token budget

The budget may only go down. `--max-num-batched-tokens` sizes the input buffers,
the block table and the CUDA graphs at startup; a wider budget would address
memory that was never allocated. To sweep upward, start the engine with the
widest value you may need and sweep below it.

```bash
python -m vllm.dev.cli sched 4096
```

Raising it above the startup value fails with the envelope in the message.

## Operator provider

An operator is swapped in two steps: import a module that registers an
implementation on the IR op, then install a dispatch priority that puts it
first. The module must be prebuilt — a Python module, a Triton kernel whose
cache is warm, or an already loaded CUDA extension. Neither step compiles.

```bash
python -m vllm.dev.cli apply --module path/to/variants.py \
    --op rms_norm --provider my_provider
```

The reply carries two things worth keeping: the installed priority per worker,
and the re-capture result. A swap without re-capture leaves the graphs stale;
replaying one raises

```
CUDA graph ... was captured under op swap epoch N, the runtime is at M.
```

That error is the point. A replay runs the old kernel, so measuring it would
report the old provider's cost under the new provider's name. Graphs are
re-captured with:

```bash
python -m vllm.dev.cli recapture
```

## Checking a swap

`vllm/dev/examples/rms_norm_variants.py` registers a numerically identical
provider and a deliberately slower one. Swapping to the slow one must change the
timing and leave the tokens untouched:

```bash
python -m vllm.dev.cli apply --module vllm/dev/examples/rms_norm_variants.py \
    --op rms_norm --provider dev_twice
python -m vllm.dev.cli verify --repeat 3
```

`verify` prints a token digest per run, whether all runs agree, and the median
time. A swap is only evidence when both move in the direction you expect: the
digest stays fixed and the time changes.

## Limits

- CUDA graphs must be re-captured after every operator swap; there is no partial
  invalidation, so a swap re-captures every decode graph the engine holds.
- A swap does not touch frameworks the engine only compiles once, such as
  `torch.compile` artifacts whose guards captured the old provider.
- Swapping while requests are in flight is not guarded. Pause or drain first,
  as with any other development-time change.
