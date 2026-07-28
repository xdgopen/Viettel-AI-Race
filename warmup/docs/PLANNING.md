# Refactor Plan — LLM Inference Optimization Challenge (Bài 3)

Status: **draft for review/brainstorming**. No code has been changed yet. This document exists to align on scope and architecture before implementation starts.

## 0. Why this document exists

`docs/requirement.html` (updated 18/07/2026) is the current, authoritative spec. It supersedes an older round-specific spec still sitting in `docs/1. Sơ loại.md` and the general phase overview in `docs/0. Đề bài.md`. The current `warmup/` codebase was built against the **old** spec and has not been updated. It also contains cleanup debt unrelated to the spec change. This plan covers both: closing the spec gap, and getting the code to a clean, non-duplicated structure.

## 1. Gap analysis — old spec vs. current spec vs. current code

| Aspect | Old spec (`docs/1. Sơ loại.md`) | Current spec (`docs/requirement.html`) | Current code (`warmup/`) |
|---|---|---|---|
| Model | `Qwen/Qwen3.5-2B` | `LiquidAI/LFM2.5-1.2B-Instruct` | `Qwen/Qwen3.5-2B` (compose, Solution doc, README) / `Qwen/Qwen2.5-3B-Instruct` (RUN_GUIDE.md) — inconsistent even with itself |
| OS / driver | Ubuntu 22.04 LTS, CUDA 12.x | Ubuntu 24.04 LTS, NVIDIA driver 590.x, CUDA 13.x | Docs still say Ubuntu 22.04 / CUDA 12.x |
| F_ttft / C_ttft | 100 ms / 1500 ms | **10 ms / 400 ms** | 100 ms / 1500 ms (`benchmark_local.py`) |
| F_tpot / C_tpot | 20 ms / 45 ms | **1 ms / 10 ms** | 20 ms / 45 ms (`benchmark_local.py`) |
| γ, w | 2, 0.5 | 2, 0.5 (unchanged) | 2, 0.5 — correct, coincidentally |
| Trace format | JSONL, one literal pre-built request body per row (`trace-round1.jsonl`, 120 rows) | **Multi-turn conversation descriptor**: `num_conversations`, `user_turns_per_conversation`, `total_request`, `shared_system_prefix_tokens`, `per_conversation_prefix_tokens`, `new_user_tokens_per_turn`, `output_tokens_per_turn_pinned`, `arrival` | Only understands the old literal-body format; no concept of a conversation, turn sequencing, or carrying forward assistant replies |
| Optimization framework | vLLM only | vLLM only (unchanged) | vLLM only — correct |
| Accuracy gate | GPQA Diamond, baseline 0.40, same f(Δ) piecewise curve | Same formula/baseline, run post-online-round on ≤5 chosen submissions via BTC's `bench-gpqa-diamond.sh`/`lm_eval` | **No local accuracy/GPQA tooling exists at all** |
| Submission compose baseline | `--max-model-len=262144`, no kv-cache/scheduling flags | `--max-model-len=32768`, `--gpu-memory-utilization=0.95`, `--tensor-parallel-size=1`, `--enable-prefix-caching` | `warmup/docker-compose.yml` already explores `--kv-cache-dtype=fp8`, `--enable-chunked-prefill`, `--scheduler-policy=priority` etc. — reasonable techniques, wrong model/thresholds, never validated against the new trace shape |

Additional repo debt, independent of the spec change:

- **`warmup/output/`** is a second, independently-forked, lower-quality copy of `benchmark_local.py` and `docker-compose.yml` (different scoring code, uses `numpy`, still hardcodes the old thresholds and `Qwen3.5-2B`, Vietnamese-only console output, no argparse). It directly conflicts with the root-level versions of the same filenames and should not be kept as-is.
- **`warmup/input/__MACOSX/`** — macOS zip artifacts (`._docker-compose-baseline.yml`, `._trace-round1.jsonl`), pure junk.
- **Three overlapping guides** (`RUN_GUIDE.md`, `COLAB_GUIDE.md`, `VAST_OPTIMIZATION_GUIDE.md`) repeat the same setup/benchmark steps with drifting details (different timeouts, different flag sets).
- **Top-level `README.md`** references `round-2/`/`round-3/` placeholders that don't exist on disk, and still states the model is Qwen.

## 2. Target architecture

Proposed clean layout for `warmup/` (kept as the folder name since it matches the competition's phase/round convention):

```
warmup/
├── README.md                              # single quickstart: setup → run → benchmark → sweep → accuracy check → submit
├── docs/
│   ├── OPTIMIZATION_NOTES.md              # merges VAST_OPTIMIZATION_GUIDE.md + Solution_Round_1.md strategy content
│   └── COLAB.md                           # trimmed Colab alt-path (from COLAB_GUIDE.md)
├── config/
│   └── ers_config.py                      # single source of truth for ALL scoring constants, each annotated with its requirement.html table row
├── trace/
│   ├── schema.py                          # typed multi-turn conversation descriptor + validator
│   ├── generate_trace.py                  # synthetic trace generator for the NEW schema (nothing like this exists today)
│   └── replay_trace.py                    # conversation-aware replay client: maintains per-conversation message history, sequences turns, computes ERS
├── benchmark/
│   ├── scoring.py                         # extracted: component_score / request_score / accuracy_factor / percentile
│   └── report.py                          # shared JSON report builder (ERS, TTFT/TPOT mean/p50/p95, error histogram)
├── accuracy/
│   ├── gpqa_subset.jsonl                  # small public smoke-test subset (NOT the secret BTC set)
│   ├── run_gpqa_local.py                  # best-effort local Δ / f(Δ) / score estimator
│   └── LM_EVAL_WIRING.md                  # documented lm_eval recipe for a closer (still unofficial) approximation
├── sweep/
│   └── sweep_params.py                    # Python successor to tune_vast.sh; ranks results by the spec's tie-break order
├── input/
│   └── trace-descriptor.sample.jsonl      # checked-in example in the new schema, produced by generate_trace.py
├── docker-compose.yml                     # retargeted to LFM2.5-1.2B-Instruct; mirrors the BTC sample verbatim (entrypoint/model/host/port comments untouched)
├── docker-compose.override.example.yml    # local-only resource/env overrides for sweeping; never submitted
└── Dockerfile                             # unchanged, revalidated against the new base image assumptions
```

**Proposed deletions** (folded into the structure above, not left as dead weight): `warmup/output/` (entire dir), `warmup/input/__MACOSX/` (entire dir), `warmup/input/trace-round1.jsonl`, `warmup/input/docker-compose-baseline.yml`, `warmup/benchmark_local.py`, `warmup/colab_run_benchmark.ipynb`, `warmup/RUN_GUIDE.md`, `warmup/VAST_OPTIMIZATION_GUIDE.md`, `warmup/Solution_Round_1.md`, `warmup/COLAB_GUIDE.md`, `warmup/requirements-benchmark.txt` (merged into a single `requirements.txt`).

## 3. Component purpose & what each replaces

| File | Purpose | Replaces / absorbs |
|---|---|---|
| `config/ers_config.py` | Named constants: `F_TTFT_MS=10, C_TTFT_MS=400, F_TPOT_MS=1, C_TPOT_MS=10, GAMMA=2, TTFT_WEIGHT=0.5`, plus accuracy-gate breakpoints (`0.10`, `0.16`, baseline `0.40`) and the tie-break order (accuracy drop → p95 TTFT → throughput → submission time). | The hardcoded, twice-drifted thresholds in `benchmark_local.py` and `output/benchmark_local.py`. Becomes the **only** place either script may hold a numeric threshold. |
| `benchmark/scoring.py` | Pure functions: `component_score`, `request_score`, `accuracy_factor`, `percentile`; imports only from `config/ers_config.py`. | De-duplicates the ERS math currently copy-pasted (with drift) between the two `benchmark_local.py` copies. |
| `benchmark/report.py` | Shared aggregate-report builder (ERS mean, TTFT/TPOT mean/p50/p95, error histogram, `score_without_accuracy`, `score`). | The report-building block at the bottom of `benchmark_local.py`'s `run()` — reused by both `replay_trace.py` and `sweep_params.py` instead of reimplemented per script. |
| `trace/schema.py` | Typed descriptor for the 8 new trace fields, with a non-fatal validator (e.g. checks `total_request` against `num_conversations × user_turns_per_conversation`). | Nothing existed — the old trace used a completely different, literal-body schema. |
| `trace/generate_trace.py` | Synthetic generator using the real `LFM2.5-1.2B-Instruct` tokenizer to hit exact target token counts per field; produces `input/trace-descriptor.sample.jsonl`; exposes knobs for conversation count, turn count, prefix sizes, arrival cadence. | Fills the gap that no sample file exists anywhere in the repo in the new format. |
| `trace/replay_trace.py` | The actual multi-turn-aware client: builds per-conversation `messages` history (system prompt from `shared_system_prefix_tokens`, `per_conversation_prefix_tokens` injected into turn 1), issues turns in sequence, appends assistant replies to history, enforces pinned output length, measures TTFT/TPOT via SSE token timestamps, scores via `benchmark/scoring.py`. | Wholesale replacement of `benchmark_local.py`'s `send_request`/`run()`, which fires pre-built bodies once with no conversation or turn concept. |
| `accuracy/run_gpqa_local.py` + `gpqa_subset.jsonl` | Sends bundled questions to the live endpoint, computes accuracy, feeds Δ/f(Δ) through `scoring.accuracy_factor`. | Nothing existed. Deliberately scoped as approximate — the real 100-question GPQA set is secret. |
| `accuracy/LM_EVAL_WIRING.md` | Documents pointing `lm_eval --model local-chat-completions` at the running server for `gpqa_diamond`, for a closer (still unofficial) approximation. | Documents rather than reimplements BTC's `bench-gpqa-diamond.sh`. |
| `docker-compose.yml` | BTC's exact sample structure (comments like `#Don't change this to vllm-server` preserved verbatim) + `--served-model-name=LFM2.5-1.2B-Instruct` + carried-forward optimization flags + a `--max-model-len` sized from real worst-case trace length instead of guessed. | Retargets the existing `warmup/docker-compose.yml`; `input/docker-compose-baseline.yml` is dropped since `requirement.html`'s embedded sample is now the single authoritative baseline reference. |
| `sweep/sweep_params.py` | Drives `docker compose` via subprocess across parameter grids, health-checks, invokes `replay_trace.py`, ranks by the spec's tie-break order (not raw ERS alone). | Structural rewrite of `tune_vast.sh` in Python so it can import `config/` and `benchmark/report.py` directly instead of an inline bash/heredoc JSON parser. |
| `warmup/README.md` | Single quickstart: build/run → generate or supply a trace → replay → sweep → accuracy sanity-check → submit. | Consolidates the operational steps of `RUN_GUIDE.md`. |
| `docs/OPTIMIZATION_NOTES.md` | Rationale for each flag choice (fp8 KV cache, chunked prefill, priority scheduling, prefix caching) and which spec-listed directions (speculative decoding, custom kernels, CPU/NVMe offload, semantic caching) remain unexplored. | Merges `VAST_OPTIMIZATION_GUIDE.md` + `Solution_Round_1.md`'s strategy sections. |

## 4. Phased roadmap

**Phase 0 — Spec-alignment & cleanup** (cheap, stops active drift, do first)
- Delete `warmup/output/`, `warmup/input/__MACOSX/`, `warmup/input/trace-round1.jsonl`, `warmup/input/docker-compose-baseline.yml`.
- Fix every stale `Qwen*` reference repo-wide (README.md, all warmup docs) to `LiquidAI/LFM2.5-1.2B-Instruct`.
- Fix stale OS/CUDA/driver claims (22.04/CUDA12 → 24.04/CUDA13/driver 590.x).
- Fix top-level `README.md`'s stale `round-2/round-3` placeholder claim.

**Phase 1 — Single source of truth for scoring**
- `config/ers_config.py` with all six thresholds + accuracy breakpoints, each citing its `requirement.html` source row.
- Extract `benchmark/scoring.py` + `benchmark/report.py`.

**Phase 2 — Trace tooling** (highest-risk, most novel — no prior code or sample exists)
- `trace/schema.py` → `trace/generate_trace.py` → `trace/replay_trace.py`.
- Only delete `benchmark_local.py`/`colab_run_benchmark.ipynb` once `replay_trace.py` is validated end-to-end.

**Phase 3 — Benchmarking validation** (a checkpoint, not new code)
- Stand up `docker-compose.yml` locally against a real or cached `/model` mount.
- Run `replay_trace.py`, sanity-check TTFT/TPOT/ERS are plausible and no silent truncation against `--max-model-len`.
- Size `--max-model-len` from the observed worst-case conversation length.

**Phase 4 — Optimization experiments**
- `sweep/sweep_params.py` over `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `--kv-cache-dtype`, `--scheduler-policy`, `--max-model-len`.
- Re-verify carried-forward flags are still accepted by the pinned `vllm/vllm-openai:v0.22.1` image (vLLM flags churn across releases and this was never re-verified after the spec update).
- Rank by the spec's tie-break order, not raw ERS.

**Phase 5 — Accuracy-gate helper** (independent of the ERS pipeline, lower urgency)
- `accuracy/gpqa_subset.jsonl`, `run_gpqa_local.py`, `LM_EVAL_WIRING.md`.

**Phase 6 — Docs/submission packaging** (last, since flags/layout are still moving through 1–5)
- Finalize `docker-compose.yml` with winning Phase 4 flags + Phase 3's `--max-model-len`.
- Consolidate guides per Section 2; delete the superseded ones.
- Update top-level `README.md` / `docs/Solution.md`.
- Stamp a "spec version aligned to requirement.html (18/07/2026)" marker in `ers_config.py` so a future spec revision is trivially diffable against what the code assumes.

## 5. Open questions — the trace schema is prose-only in `requirement.html`, no worked example exists

**Update:** `docs/grading-workload-spec.json` (a public data file, added after this plan was first written) resolved #2, #4, and #6 below with concrete numbers: `num_conversations=70`, `user_turns_per_conversation=6` (constant, not a range), `total_requests=420` (exactly `70×6`), `shared_system_prefix_tokens=1000`, `per_conversation_prefix_tokens=1000` (constant token *count* across conversations - only the filler *content* differs per conversation), `new_user_tokens_per_turn=150` and `output_tokens_per_turn_pinned=300` (constant scalars, not per-turn-varying lists), `arrival="Poisson, seed 42"`. `workload/schema.py`'s `expand_global_descriptor()` + `generate_trace.py --from-spec` now consume this format directly; the per-conversation-JSONL representation is kept internally as the materialized/replay format, populated either by expanding a real global-descriptor spec (preferred, when available) or by the original random-range sampling (for stress-testing shapes the spec doesn't cover). `input/trace-descriptor.sample.jsonl` is now generated from this real spec, not arbitrary defaults - worst-case context computes to exactly 4700 tokens, which is why `docker-compose.yml`'s `--max-model-len` dropped from the BTC sample's 32768 to 6144 (~30% headroom over 4700). Remaining open items:

1. **Turn sequencing** — does turn *N+1* fire only after turn *N*'s response completes ("closed-loop", like a real chat client), or at a fixed `arrival` offset regardless of turn *N*'s status? Still unconfirmed by the spec file (it only describes conversation-start arrivals, not intra-conversation timing). **Recommend closed-loop** — turn *N+1* structurally needs turn *N*'s assistant reply in its history to be a valid multi-turn request.
2. ~~Trace file granularity~~ — **resolved**: single global scalar descriptor, homogeneous across all conversations.
3. **Enforcing `output_tokens_per_turn_pinned`** — likely requires `max_tokens=min_tokens=<pinned>` + `ignore_eos: true` so TPOT reflects true per-token speed rather than variable-length generation stopping early. Still an assumption; the spec file doesn't state the enforcement mechanism.
4. ~~`total_request` reconciliation~~ — **resolved**: confirmed exact (`70×6=420`), not just a non-fatal-warning fallback case.
5. **No disclosed request timeout** — ERS treats error/timeout/0-token identically (score 0), so the exact value only affects local dev iteration speed, not scoring correctness. Keep as a documented CLI flag.
6. ~~`--max-model-len` sizing~~ — **resolved**: computed exactly from the spec file as 4700 tokens worst-case; `docker-compose.yml` uses 6144 (30% headroom). The Poisson arrival *rate* is still not given by the spec file (only "seed 42") and remains a CLI flag (`--arrival-rate-per-sec`, default 2.0/sec) - override it if a real rate is ever disclosed.

## 6. Verification plan (once implementation starts)

- Run `generate_trace.py` and hand-check token-count math for one synthetic conversation against the field definitions.
- Run `replay_trace.py` against a local vLLM instance actually serving `LFM2.5-1.2B-Instruct` and confirm TTFT/TPOT/ERS numbers look plausible before trusting any `sweep_params.py` results built on top.
- Diff the final `docker-compose.yml` against the literal BTC sample embedded in `requirement.html` to confirm only optimization flags were added, and the entrypoint/model/host/port lines (and their "don't change" comments) are untouched.
- Spot-check `accuracy/run_gpqa_local.py`'s Δ/f(Δ) output against a manually-computed example using the formulas in Section 1.

## 7. Critical files referenced

- `docs/requirement.html` — source of truth (read in full)
- `docs/1. Sơ loại.md`, `docs/0. Đề bài.md` — superseded specs, used for diffing (since removed from the repo by the user - no longer needed once this plan captured the diff)
- `docs/grading-workload-spec.json` — public real workload shape, added after this plan was first written; resolved several open questions in section 5
- `warmup/benchmark_local.py`, `warmup/output/benchmark_local.py` — drifted duplicate pair
- `warmup/docker-compose.yml`, `warmup/input/docker-compose-baseline.yml`
- `warmup/tune_vast.sh`
- `warmup/input/trace-round1.jsonl` — old-schema sample
- Top-level `README.md`
