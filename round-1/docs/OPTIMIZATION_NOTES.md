# Optimization Notes - Round 1

Rationale behind the flags currently in `../docker-compose.yml`, and which
of the spec's allowed optimization directions (`docs/requirement.html`
section 3) are still unexplored. Supersedes the old `VAST_OPTIMIZATION_GUIDE.md`
and `Solution_Round_1.md`, which were written against the old spec (Qwen
model, old ERS thresholds).

## Hardware constraints driving every flag choice

MiG H200 slice: **18GB VRAM, 3 CPU cores, 8GB system RAM**. Every flag below
exists because of one of these three numbers:

| Constraint | Risk | Flag |
|---|---|---|
| 8GB host RAM | vLLM can spill KV cache to CPU RAM and OOM-kill the container | `--swap-space=0` |
| 3 CPU cores | Per-request log formatting and scheduling overhead competes with token generation | `--disable-log-requests` |
| 1 GPU (MiG slice) | Multi-GPU parallelism is pointless overhead | `--tensor-parallel-size=1` |
| 18GB VRAM | KV cache is the scarce resource, not compute | `--kv-cache-dtype=fp8`, `--enable-prefix-caching` |

## `--max-model-len`: 6144, not the BTC sample's 32768

`docs/grading-workload-spec.json` (public, gives the actual workload shape)
puts worst-case context at exactly 4700 tokens per conversation
(`shared_system_prefix_tokens` 1000 + `per_conversation_prefix_tokens` 1000 +
6 turns × (150 user + 300 output)). 6144 keeps ~30% headroom for
chat-template overhead without wasting the ~26k-token difference from the
BTC sample - that difference is VRAM that would otherwise sit idle instead
of holding more concurrent sequences' KV cache. `--max-num-seqs` (currently
32, carried over from the old 32768-context config) should be re-swept
upward given this - each sequence's KV footprint just shrank ~5x. If
`grading-workload-spec.json` ever changes, regenerate
`input/trace-descriptor.sample.jsonl` (`workload/generate_trace.py --from-spec`)
and recompute this number - don't leave it stale.

## Why FP8 KV cache, not weight quantization

The Accuracy Gate (`docs/requirement.html` section 2) penalizes accuracy
drop starting at Δ>0.10 and zeroes the entire submission's score at
Δ≥0.16. Weight quantization (AWQ/GPTQ INT4) on a 1.2B-parameter model risks
crossing that line, especially since the model here is smaller than the
old spec's 2-3B target - less redundancy to absorb quantization error.

**Current stance:** keep weights at native BF16, and only apply FP8 to the
**KV cache**, not the weights. This roughly doubles effective KV cache
capacity (more concurrent sequences fit in 18GB VRAM → higher ERS) without
touching the forward pass's numerical precision. Validate this assumption
with `accuracy/run_gpqa_local.py` (a rough smoke test) or
`accuracy/LM_EVAL_WIRING.md` (closer, still unofficial) before trusting it -
this hasn't been confirmed on the real model yet.

## Why prefix caching matters here specifically

The trace schema's `shared_system_prefix_tokens` field describes a system
prompt **identical across all conversations** (`docs/requirement.html`
section 1). `--enable-prefix-caching` lets vLLM reuse that prefix's KV cache
across every conversation instead of recomputing it, which should measurably
cut TTFT once the first conversation has primed the cache. `workload/replay_trace.py`
renders that shared prefix as literal identical text once per run (see
`workload/text_fill.py`), so this should be directly observable in a local
`replay_trace.py` run's TTFT distribution over time (later conversations
should show lower TTFT... check the raw skip/hit signal wLLM exposes in
its metrics if you want to confirm the cache is actually being hit, not just
assume it from an ERS improvement).

## Grid search: max-num-seqs / max-num-batched-tokens

`sweep/sweep_params.py` grid-searches both. Rules of thumb carried forward
from earlier exploration (re-validate on the real hardware, don't trust
these numbers blindly):

- Lower `--max-num-batched-tokens` favors decode (better TPOT for
  in-flight sequences) but lengthens prefill for new arrivals.
- Higher values can cut TTFT for bursty arrivals but degrade TPOT for
  sequences already decoding, if a big prefill batch monopolizes a step.
- Run each candidate ≥3 times (`--repeats 3`, the default) and rank by
  median - a single run's ERS is noisy enough to mislead a flag choice.

## Still unexplored (spec section 3 lists these as allowed)

- **Speculative decoding** (draft model or self-speculative) - could improve
  TPOT further; not yet wired into `docker-compose.yml`.
- **Custom CUDA/Triton kernels, FlashAttention/FlashInfer fused kernels,
  CUDA Graphs** - vLLM may already apply some of these by default depending
  on build flags; not independently verified here.
- **Semantic caching** - beyond prefix caching, no semantic-similarity cache
  layer is configured.
- **CPU/NVMe KV cache offloading** - not configured; with only 8GB host RAM,
  this trades VRAM pressure for host RAM pressure and needs careful sizing
  if attempted.

## Before submitting

Re-verify every flag in `docker-compose.yml` is still accepted by the
pinned `vllm/vllm-openai:v0.22.1` image - vLLM flags churn across releases
and none of the carried-forward flags (`--kv-cache-dtype`,
`--enable-chunked-prefill`, `--scheduler-policy`) have been re-validated
against this specific version since the spec updated the target model.
