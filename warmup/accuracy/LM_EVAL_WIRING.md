# Wiring `lm-evaluation-harness` for a closer (still unofficial) accuracy check

`accuracy/run_gpqa_local.py` is a smoke test against 8 self-authored
placeholder questions - useful for checking the Δ/f(Δ)/Score plumbing, not
for estimating real accuracy. For a closer approximation, point
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
at your running server and run the **public** `gpqa_diamond_zeroshot` task.

This is still not the official grader: BTC's post-online-round gate
(`bench-gpqa-diamond.sh` / `lm_eval`, per `docs/requirement.html`) uses a
secret, fixed 100-question set. A public-set run only tells you whether a
quantization/serving choice is in the right ballpark before you spend one of
your ≤5 final submission picks - it is not a substitute for the real number.

## Setup

```bash
pip install lm-eval
```

## Run against your local vLLM server

vLLM's OpenAI-compatible server can be scored directly via `lm_eval`'s
`local-chat-completions` model type - no need to load the model a second
time inside `lm_eval` itself:

```bash
lm_eval \
  --model local-chat-completions \
  --model_args model=LFM2.5-1.2B-Instruct,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,max_retries=3 \
  --tasks gpqa_diamond_zeroshot \
  --apply_chat_template \
  --output_path results/gpqa_local.json
```

Notes:

- `model` must match `--served-model-name` in `docker-compose.yml`
  (`LFM2.5-1.2B-Instruct`).
- Run this against a **separate** local instance or between online-round
  benchmark runs - don't run it concurrently with `workload/replay_trace.py`
  or `sweep/sweep_params.py`, since it will contend for the same GPU/serving
  capacity and skew both measurements.
- The reported `acc` for `gpqa_diamond_zeroshot` is the number to plug into
  `accuracy/run_gpqa_local.py --ers <measured_ers>` in place of the
  placeholder-set accuracy, or directly into
  `benchmark.scoring.accuracy_factor(measured_accuracy)`.
- If `lm_eval` reports on the full public GPQA Diamond set (~198 questions)
  rather than a 100-question subset, treat the accuracy number as directional,
  not identical in scale to BTC's own 100-question run.
