# Step-by-Step: Running & Checking Output on Rented Compute

Practical runbook for standing up the `round-1/` serving stack on a rented
Vast.ai GPU and reading the score it produces. For the *why* behind flag
choices and rented-hardware-vs-H200 caveats, see
[`round-1/docs/OPTIMIZATION_NOTES.md`](../round-1/docs/OPTIMIZATION_NOTES.md)
and [`round-1/docs/VAST_TESTING_GUIDE.md`](../round-1/docs/VAST_TESTING_GUIDE.md) -
this doc is the condensed command sequence to actually run.

## Step 1 — Rent the instance

`LiquidAI/LFM2.5-1.2B-Instruct` is small enough that almost any GPU with
≥16GB VRAM can run it - the tier you pick mainly affects how much you can
trust the `--kv-cache-dtype=fp8` numbers, not whether the benchmark runs at
all. Pick based on budget vs. fidelity to the graded Hopper-architecture H200:

| Tier | Example listings | FP8 fidelity |
|---|---|---|
| Best | H100 | Native, same generation as H200 |
| Good | L4, L40/L40S, RTX 4090 | Native (Ada) |
| Middle | A10, A100, A40, RTX 3090 | KV-cache-only fp8 works (Ampere), no tensor-core acceleration |
| Cheapest | Tesla T4, Tesla V100 | Likely unsupported (Volta/Turing) - smoke test only |

Full rationale in
[`round-1/docs/VAST_TESTING_GUIDE.md`](../round-1/docs/VAST_TESTING_GUIDE.md)
section 0. Pick a Docker-enabled template, and request close to
**3 vCPU / 8GB RAM** in the offer if the marketplace lets you constrain it.
Launch it and copy the SSH command from the console.

## Step 2 — Connect and verify the environment

```bash
ssh -p <port> root@<host>

nvidia-smi                       # confirm the GPU model matches what you rented
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Step 3 — Get the code onto the instance

```bash
# If pushed to a git remote:
git clone <your-repo-url> && cd viettel-bai-3/round-1

# Otherwise, from your local machine:
rsync -avz -e "ssh -p <port>" round-1/ root@<host>:~/round-1/
```

## Step 4 — Download the model weights

BTC mounts the model automatically at `/model` in the graded environment;
on rented compute you do this yourself:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct --local-dir ~/models/LFM2.5-1.2B-Instruct
```

## Step 5 — Configure the local override and start the server

```bash
cd ~/round-1
cp docker-compose.override.example.yml docker-compose.override.yml
```

Edit `docker-compose.override.yml`:
```yaml
services:
  model:
    volumes:
      - /root/models/LFM2.5-1.2B-Instruct:/model:ro
    deploy:
      resources:
        limits:
          cpus: "3"
          memory: 8g
```

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
docker compose -f docker-compose.yml -f docker-compose.override.yml logs -f model
# wait for: Uvicorn running on http://0.0.0.0:8000
```

## Step 6 — Smoke test

```bash
curl -fsS http://localhost:8000/health

curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LFM2.5-1.2B-Instruct","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8,"temperature":0}'
```

If either fails, check `docker compose logs model` for OOM kills or CUDA
errors before going further - don't try to tune throughput on an unstable
base config.

## Step 7 — Run the benchmark (this is what produces the score)

```bash
python3 workload/replay_trace.py \
  --trace input/trace-descriptor.sample.jsonl \
  --url http://localhost:8000/v1/chat/completions \
  --output results/run.json
```

This replays all 70 conversations / 420 requests from
`docs/grading-workload-spec.json`'s published shape (see
[`docs/PLANNING.md`](PLANNING.md) section 5) and prints a JSON report to
stdout (also saved to `results/run.json`).

## Step 8 — Read the output

```json
{
  "requests": 420,
  "success": 420,
  "errors": {},
  "ers": 0.74,
  "score_without_accuracy": 74.0,
  "baseline_accuracy": 0.4,
  "accuracy_factor": null,
  "score": null,
  "ttft_ms": { "mean": 21.4, "p50": 21.3, "p95": 21.9 },
  "tpot_ms": { "mean": 3.4,  "p50": 3.4,  "p95": 3.5  }
}
```

What each field means and what to check:

| Field | What it tells you |
|---|---|
| `requests` / `success` | **Check these match first.** If `success < requests`, something failed - see `errors` before looking at anything else. |
| `errors` | Map of failure type → count (`TIMEOUT`, `HTTP_5xx`, `EMPTY_RESPONSE`, etc.). Should be `{}`. Any entry here means those requests scored 0. |
| `ers` | The Effective Request Score, 0–1. This is the number that matters - higher is better. Computed from `ttft_ms`/`tpot_ms` against the thresholds in `config/ers_config.py` (F_ttft=10ms, C_ttft=400ms, F_tpot=1ms, C_tpot=10ms). |
| `score_without_accuracy` | `100 × ers` - your latency-only score estimate, pre-Accuracy-Gate. |
| `accuracy_factor` / `score` | `null` until you pass `--accuracy <measured>` to `replay_trace.py` (see Step 10) - the Accuracy Gate only runs post-online-round on your real submission, so this is always an estimate locally. |
| `ttft_ms` / `tpot_ms` | Mean/p50/p95 in milliseconds. Compare against the floor/ceiling above: p95 well under the ceiling means that tail isn't costing you score; a mean near the floor means you're close to maxing out that component. |

Rule of thumb: if `ers` is well below 1.0, check whether `ttft_ms.mean` or
`tpot_ms.mean` is the bigger drag (whichever is proportionally closer to its
ceiling) - that tells you whether to focus tuning on prefill (TTFT,
`--max-num-batched-tokens`) or decode (TPOT, `--max-num-seqs`,
`--kv-cache-dtype`).

## Step 9 (optional) — Sweep parameters to improve the score

```bash
python3 sweep/sweep_params.py --max-num-seqs 16 32 48 64 --max-num-batched-tokens 4096 8192 16384 --repeats 3
cat results/ranking.json   # best candidate first, per the spec's tie-break order
```

## Step 10 (optional) — Sanity-check the Accuracy Gate

```bash
python3 accuracy/run_gpqa_local.py --url http://localhost:8000/v1/chat/completions --ers 0.74
```

This is a plumbing smoke test against 8 placeholder questions, not the real
GPQA Diamond set - see
[`round-1/accuracy/LM_EVAL_WIRING.md`](../round-1/accuracy/LM_EVAL_WIRING.md)
for a closer approximation.

## Step 11 — Shut down

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml down
```

Then stop/destroy the instance from the vast.ai console - it bills per hour
while running, whether or not the GPU is actively in use.

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits right after `up -d` | OOM on host RAM or VRAM | Lower `--gpu-memory-utilization`, confirm `--swap-space=0` is present, check `docker compose logs model` |
| `errors: {"TIMEOUT": N}` | Server overloaded for the configured concurrency | Lower `--max-num-seqs`, or raise `--timeout` in `replay_trace.py` if the box is just slow (lower-tier GPU, not a real fix) |
| `errors: {"HTTP_400: ..."}` | Context exceeds `--max-model-len` | Shouldn't happen at the current 6144 setting for the published trace shape; if you changed the trace, recompute via `workload/schema.py summarize_trace()` |
| `ers` much lower than expected on a cheaper-tier GPU | Expected - see `round-1/docs/VAST_TESTING_GUIDE.md` section 0 | Treat as directional only; re-validate on Ada/Hopper-class hardware before trusting absolute numbers |
