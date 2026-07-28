# Testing on Rented Vast.ai GPUs

Guideline for testing this solution on a rented Vast.ai instance as a
stand-in for the graded **MiG H200** slice (18GB VRAM, 3 CPU cores, 8GB RAM,
Ubuntu 24.04, driver 590.x, CUDA 13.x, Hopper architecture).

`LiquidAI/LFM2.5-1.2B-Instruct` is small (~2.4GB of BF16 weights) - almost
any GPU Vast.ai lists with ≥16GB VRAM can *run* it. The real question isn't
"can it run", it's **how trustworthy the numbers are**, which comes down to
compute capability (architecture generation), not raw VRAM. Pick a tier
below based on budget vs. how much you need to trust the result.

## 0. Compute tiers on Vast.ai, ranked by fidelity to the graded H200


| Tier                                          | Example Vast.ai listings                                     | Compute capability | FP8 support                                                                                                                                                                                                                                                                            | Relative cost              | Use for                                                                                                                                                                                           |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Best** - Hopper                             | H100 (80GB); H200 occasionally listed but rare/expensive     | 9.0                | Native FP8 tensor cores                                                                                                                                                                                                                                                                | $$$                        | Final pre-submission validation of any FP8/quantization decision - the only tier architecturally identical to the grading slice.                                                                  |
| **Good** - Ada Lovelace                       | L4 (24GB), L40/L40S (48GB), RTX 4090 (24GB)                  | 8.9                | Native FP8 tensor cores                                                                                                                                                                                                                                                                | $–$$ (L4/4090 often cheap) | Day-to-day tuning with FP8 confidence close to Hopper; L4 is inference-tuned and usually the best cost/fidelity balance.                                                                          |
| **Middle** - Ampere                           | A10/A10G (24GB), A100 (40/80GB), A40 (48GB), RTX 3090 (24GB) | 8.0 / 8.6          | **No FP8 tensor cores**, but meets vLLM's `--kv-cache-dtype=fp8` minimum (compute ≥8.0) - KV cache fp8 is a storage/quantize-dequantize format, not a tensor-core matmul, so this flag specifically should still exercise a real code path here, just without Ada/Hopper's throughput. | $–$$                       | A reasonable middle ground for *this* config, since `docker-compose.yml` only quantizes the KV cache (not model weights) - Ampere is architecturally capable of that path, unlike the tier below. |
| **Cheapest** - Volta/Turing ("Tesla"-branded) | Tesla T4 (16GB), Tesla V100 (16/32GB)                        | 7.0 / 7.5          | Below vLLM's fp8-kv-cache compute floor - likely rejected outright or unsupported                                                                                                                                                                                                      | $                          | Functional smoke tests only (does it start, does every request succeed) - least representative tier for anything latency- or FP8-related.                                                         |


Confirm the exact fp8 support boundary against the pinned
`vllm/vllm-openai:v0.22.1` image yourself (compute-capability requirements
can shift across vLLM releases) - don't take the table above as a
substitute for watching the startup log, see section 2.

**Bottom line:** any tier can validate that the serving stack works
end-to-end. Only Ada/Hopper validates the FP8 decision with real
confidence; Ampere is a reasonable proxy specifically because this
config's only quantization is the KV cache, not the weights; Volta/Turing
is cheapest but should be treated as a pure smoke test.

Two constants regardless of tier:

- **Memory bandwidth / VRAM won't match H200 exactly on any rented card** -
absolute TTFT/TPOT numbers won't transfer 1:1. Only *relative* comparisons
between two configs on the *same* rented card are meaningful.
- **CPU/RAM allocation** - Vast offers rarely give you exactly 3 CPU cores /
8GB RAM by default. Cap it explicitly (see section 2) or you won't catch
OOM/scheduling bottlenecks that would bite on the real grading slice.



## 1. Pick an instance

On the Vast.ai console, filter by GPU model for whichever tier fits your
budget/fidelity need above. Avoid picking a GPU with dramatically more VRAM
than 18GB "for headroom" - it hides KV-cache-capacity bottlenecks that
matter on the real slice, regardless of tier.

When configuring the instance:

- **CPU**: request as close to 3 vCPUs as the offer allows.
- **RAM**: request as close to 8GB as the offer allows (Vast often bundles
more by default - explicitly set a lower limit in the Docker options if
the marketplace UI allows it, and again via `docker-compose.override.yml`,
see section 2).
- **Disk**: at least 20GB free for the model weights + Docker image layers.
- **Template**: choose a Docker-enabled template ("provisioned with Docker
  - NVIDIA Container Toolkit") so you don't have to install the toolkit
  yourself.



## 2. Verify the environment and prepare the model

```bash
nvidia-smi                 # confirm the GPU model matches what you rented
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Download the model to local disk (BTC mounts it automatically at `/model`
in the graded environment; you must do this yourself here):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct --local-dir /workspace/models/LFM2.5-1.2B-Instruct
```

Set up the local override (mount + resource caps mimicking the graded slice):

```bash
cd warmup
cp docker-compose.override.example.yml docker-compose.override.yml
# edit docker-compose.override.yml:
#   volumes: - /workspace/models/LFM2.5-1.2B-Instruct:/model:ro
#   deploy.resources.limits: cpus: "3", memory: 8g
```

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
docker compose -f docker-compose.yml -f docker-compose.override.yml pull
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
docker compose -f docker-compose.yml -f docker-compose.override.yml logs -f model
```

Watch the startup log for two things before moving on:

1. `Uvicorn running on http://0.0.0.0:8000` (server is up).
2. Any warning/error mentioning `fp8`, `kv_cache_dtype`, or a kernel
  compatibility failure - see section 5 if you see one.



## 3. Smoke test

```bash
curl -fsS http://localhost:8000/health
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LFM2.5-1.2B-Instruct","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":8,"temperature":0}'
```

Check the container logs for OOM kills or "CUDA out of memory" - if either
appears, lower `--gpu-memory-utilization` or `--max-model-len` before
proceeding; don't try to tune throughput on top of an unstable base config.

## 4. Full local benchmark

```bash
python3 workload/generate_trace.py --output input/trace-descriptor.sample.jsonl   # or reuse the checked-in one
python3 workload/replay_trace.py --trace input/trace-descriptor.sample.jsonl \
  --output results/vast-baseline.json
```

You want `success == requests` (every conversation turn completed) before
trusting any ERS number from this box. A single timeout/error zeroes that
request's score, same as it would in the graded run.

## 5. Parameter sweep - and the FP8 caveat, concretely

```bash
python3 sweep/sweep_params.py --max-num-seqs 16 32 48 64 --max-num-batched-tokens 4096 8192 16384 --repeats 3
```

Use the resulting ranking (`results/ranking.json`) **directionally**: it
tells you which `max-num-seqs`/`max-num-batched-tokens` combination handles
concurrency best on this box, which is a reasonable prior for the graded
run. It does not tell you the graded ERS you'll get.

For `--kv-cache-dtype=fp8` specifically, what to expect depends on which
tier from section 0 you're on:

1. **Volta/Turing (Tesla T4/V100, compute <8.0)**: below vLLM's fp8-kv-cache
  floor - the server likely **refuses to start**, or the flag is silently
   rejected. Either way it's untestable on this tier; leave it in
   `docker-compose.yml` on the strength of it being spec-legal and
   effective on Hopper, but flag it as unverified in your own notes.
2. **Ampere (A10/A100/A40/RTX 3090, compute 8.0/8.6)**: meets vLLM's
  fp8-kv-cache compute floor, so the server should start and the
   quantize/dequantize path is real - but there's no FP8 tensor-core
   acceleration backing it, so don't over-trust the *magnitude* of any
   throughput gain you measure, only whether it helps or hurts directionally.
3. **Ada/Hopper (L4/L40S/RTX 4090/H100, compute ≥8.9)**: same tensor-core
  generation as the graded H200 - this is the tier whose FP8 numbers are
   actually worth trusting for a go/no-go decision.

If you can get any time on an Ada/Hopper-class rental before finalizing,
re-run your top 2-3 sweep candidates there and prefer those numbers for
anything FP8-related - everything measured on Ampere or below is a prior,
not a confirmation.

## 6. Accuracy-gate sanity check

Accuracy (unlike TTFT/TPOT) doesn't depend on GPU speed, so this is one of
the few checks whose result is meaningfully close to what the graded run
would show - run it here freely:

```bash
python3 accuracy/run_gpqa_local.py --url http://localhost:8000/v1/chat/completions --ers <measured-ers>
```

(See `accuracy/LM_EVAL_WIRING.md` for the closer `lm-evaluation-harness`
path.) Small numerical differences between GPU architectures' kernel
implementations (attention, etc.) can in principle shift a handful of
greedy-decoding outputs, but this effect should be negligible next to the
10-16 percentage-point accuracy-drop thresholds that matter here.

## 7. Before you stop the instance

- `docker compose down` to release the GPU cleanly.
- Vast.ai bills per-hour while the instance is running (even if you're not
actively using the GPU) - stop or destroy the instance once you've saved
`results/` locally.
- Do **not** commit `docker-compose.override.yml` or anything under
`results/` (both are git-ignored) - the submission artifact is the
unmodified `docker-compose.yml`.



## 8. Final pre-submission checklist

- [ ] `docker compose config` on `docker-compose.yml` alone (no override)
  ```
  resolves cleanly.
  ```
- [ ] `entrypoint`/`--model`/`--served-model-name`/`--host`/`--port` lines
  ```
  are untouched (still carry the `#Don't change this to vllm-server`
  comments).
  ```
- [ ] A full `workload/replay_trace.py` run against the unmodified
  ```
  `docker-compose.yml` (no override) reaches `success == requests`.
  ```
- [ ] Any FP8/quantization decision has either been validated on
  ```
  Ada/Hopper-class hardware, or you've accepted the risk knowingly
  given section 5's caveats.
  ```

