# Alternate dev path: Google Colab (free T4 GPU)

Colab doesn't support running Docker with GPU passthrough, so this runs
vLLM directly via pip instead of through `docker-compose.yml`. Useful for a
quick functional check of your code changes; **not** useful for tuning
serving parameters - a T4's memory bandwidth and VRAM don't match the graded
MiG H200 slice, so numbers here don't transfer. For parameter tuning, use a
rented GPU box instead (see `VAST_TESTING_GUIDE.md`).

## 1. Runtime

*Runtime → Change runtime type → T4 GPU → Save.*

## 2. Install & start vLLM in the background

```python
!pip install vllm aiohttp
```

```python
%%bash
export LD_LIBRARY_PATH=$(python3 -c 'import glob; print(":".join(glob.glob("/usr/local/lib/python3.*/dist-packages/nvidia/*/lib")))'):$LD_LIBRARY_PATH
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --served-model-name LFM2.5-1.2B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --max-num-seqs 8 > vllm.log 2>&1 &
```

Poll until ready:

```python
!tail -n 20 vllm.log   # look for "Uvicorn running on http://0.0.0.0:8000"
```

## 3. Upload the repo's `warmup/` folder

Upload (or `git clone`) at least `workload/`, `benchmark/`, `config/`, and
`input/trace-descriptor.sample.jsonl` into the Colab working directory, then:

```python
!pip install -r requirements.txt
```

## 4. Run the benchmark

```python
!python3 workload/replay_trace.py \
    --trace input/trace-descriptor.sample.jsonl \
    --url http://localhost:8000/v1/chat/completions \
    --output results/colab_run.json
```

This uses the same conversation-aware replay client and ERS thresholds as
the main workflow - no separate inline benchmark script to keep in sync.

## 5. Shut down

```python
!pkill -f api_server
```
