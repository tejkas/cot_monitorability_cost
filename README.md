# CoT Monitorability & Cost

Do activation probes remain a viable safety monitor when chain-of-thought text becomes
unreadable — and what do they actually cost inside a real serving engine?

A **legibility-gradient** experiment on **Qwen3-8B**: generate CoT traces, degrade their readability
across four tiers, and measure how a **residual-stream activation probe** and a **text-reading
monitor** diverge — then benchmark what each monitor costs under continuous batching, where the
"activations are free" assumption meets CUDA graphs.

See **[DESIGN.md](DESIGN.md)** for the full experiment design, research questions, and phased plan.
Background literature is in **[papers/](papers/)**.

## Status
**Phase 0 — scaffold.** Nothing trained yet.

## Layout
```
src/cotmon/     # package: config now; phase modules added as we build
scripts/        # entry-point scripts, one per phase (00_smoke_test.py, ...)
data/           # traces + activations (gitignored)
results/        # figures + metrics (gitignored)
papers/         # background PDFs
DESIGN.md       # the plan
```

## Setup
Compute is **cloud-rented GPU** (A100/H100). This machine is for authoring; runs happen on the box.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (run first)
```bash
python scripts/00_smoke_test.py                          # Qwen3-8B (on the GPU box)
python scripts/00_smoke_test.py --model Qwen/Qwen3-0.6B  # quick local code-path check
```
Confirms: the model loads and generates, thinking mode emits a `<think>` block, and residual-stream
activations extract with the expected shape.
