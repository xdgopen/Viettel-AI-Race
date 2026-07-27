# Round 1 (Sơ loại) - LLM Inference Optimization Challenge

Serving stack + local tooling for **LiquidAI/LFM2.5-1.2B-Instruct** on vLLM,
targeting the spec in [`../docs/requirement.html`](../docs/requirement.html)
(updated 18/07/2026). See [`../docs/PLANNING.md`](../docs/PLANNING.md) for
the full gap analysis and architecture rationale behind this layout.

## Layout

```
round-1/
├── docker-compose.yml                   # submission artifact - mirrors BTC's sample verbatim + optimization flags
├── docker-compose.override.example.yml  # local-only: model volume mount + 3-CPU/8GB-RAM/GPU limits
├── Dockerfile                           # only needed if you customize beyond flag-tuning
├── requirements.txt
├── config/ers_config.py                 # single source of truth for every scoring constant
├── benchmark/{scoring,report}.py        # ERS/accuracy math + shared JSON report shape
├── workload/{schema,generate_trace,text_fill,replay_trace}.py  # trace tooling
├── sweep/sweep_params.py                # parameter grid search
├── accuracy/{run_gpqa_local.py,gpqa_subset.jsonl,LM_EVAL_WIRING.md}
├── input/trace-descriptor.sample.jsonl  # synthetic trace, checked in
└── docs/{OPTIMIZATION_NOTES.md,COLAB.md,VAST_TESTING_GUIDE.md}
```

## 1. Local setup

```bash
cd round-1
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Point docker-compose.override.yml at your local model copy (see the
# .example file for the fields to fill in) - never commit or submit this file.
cp docker-compose.override.example.yml docker-compose.override.yml
# edit the volumes: path to wherever you downloaded LiquidAI/LFM2.5-1.2B-Instruct

docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
docker compose logs -f model   # wait for "Uvicorn running on http://0.0.0.0:8000"
curl -fsS http://localhost:8000/health
```

BTC mounts the model at `/model` automatically in the graded environment -
the override file's volume mount only exists for local/rented-GPU testing.

## 2. Generate or reuse a trace

`input/trace-descriptor.sample.jsonl` is checked in, generated from the
**real published workload spec** in
[`../docs/grading-workload-spec.json`](../docs/grading-workload-spec.json)
(70 conversations × 6 turns = 420 requests; worst-case context 4700 tokens).
Regenerate it if that spec file ever changes:

```bash
python3 workload/generate_trace.py --from-spec ../docs/grading-workload-spec.json --output input/trace-descriptor.sample.jsonl
python3 workload/schema.py input/trace-descriptor.sample.jsonl   # prints validation warnings + summary stats
```

`generate_trace.py` also supports sampling a random-range trace (no
`--from-spec`) for stress-testing shapes the published spec doesn't cover -
see `--help`. Either way, `docs/PLANNING.md` section 5 still lists a couple
of assumptions the spec file doesn't pin down (turn-sequencing semantics,
pinned-output enforcement mechanism).

## 3. Benchmark locally

```bash
python3 workload/replay_trace.py \
  --trace input/trace-descriptor.sample.jsonl \
  --url http://localhost:8000/v1/chat/completions \
  --output results/run.json
```

This replays each conversation turn-by-turn (real message history, not
one-shot requests) and prints an ERS report using the exact thresholds in
`config/ers_config.py`.

## 4. Sweep serving parameters

```bash
python3 sweep/sweep_params.py --max-num-seqs 16 24 32 --max-num-batched-tokens 4096 8192 16384 --repeats 3
```

Ranks candidates by the spec's tie-break order (ERS, then p95 TTFT, then
throughput) instead of raw ERS alone. Results land in `results/ranking.json`.

## 5. Sanity-check the accuracy gate (optional, before choosing your ≤5 final picks)

```bash
python3 accuracy/run_gpqa_local.py --url http://localhost:8000/v1/chat/completions --ers 0.72
```

This is a plumbing smoke test against 8 self-authored placeholder
questions, **not** the real GPQA Diamond set - see
`accuracy/LM_EVAL_WIRING.md` for a closer (still unofficial) approximation
via `lm-evaluation-harness`.

## 6. Submit

1. If you only tuned flags, submit `vllm/vllm-openai:v0.22.1` directly. If
   you customized the image, `docker build` from `Dockerfile`, push
   **public** to Docker Hub, and update `image:` in `docker-compose.yml`.
2. Paste the final `docker-compose.yml` into the BTC portal. Do not touch
   the `entrypoint`/`--model`/`--served-model-name`/`--host`/`--port` lines
   (marked `#Don't change this to vllm-server`).
3. Confirm one more time locally: `docker compose config`, then
   `down && up -d` and re-check `health` + a full `replay_trace.py` run.

## Testing on rented GPUs (Vast.ai)

See [`docs/VAST_TESTING_GUIDE.md`](docs/VAST_TESTING_GUIDE.md) - covers
picking a GPU tier (Tesla/Ampere/Ada/Hopper) by budget vs. fidelity to the
graded MiG H200 slice, what each tier can and can't validate, and the same
workflow above adapted to a rented box.

## Further reading

- [`docs/OPTIMIZATION_NOTES.md`](docs/OPTIMIZATION_NOTES.md) - rationale for
  each serving flag and which spec-listed optimization directions are still
  unexplored.
- [`docs/COLAB.md`](docs/COLAB.md) - free-GPU alternative to a rented box,
  useful for quick functional checks (not for tuning - T4 numbers won't
  transfer to the H200 slice).
