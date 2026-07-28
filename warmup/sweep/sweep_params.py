#!/usr/bin/env python3
"""Grid-search vLLM serving parameters against a fixed trace, ranking
candidates by the spec's tie-break order (requirement.html section 6):
ERS first, then p95 TTFT, then generation throughput.

Python successor to the old tune_vast.sh: same grid-search intent, but
drives docker compose via subprocess and imports benchmark/report.py and
config/ers_config.py directly instead of an inline bash/heredoc JSON parser.

Usage:
    # Quick default: 64/6144 vs. 64/8192, three runs each.
    python3 sweep/sweep_params.py

    # Optional broad exploration.
    python3 sweep/sweep_params.py --max-num-seqs 48 64 80 96 \
        --max-num-batched-tokens 4096 6144 8192 12288 --repeats 5

    # Exact candidates, avoiding a Cartesian-product sweep.
    python3 sweep/sweep_params.py \
        --candidates 64:8192:fp8 72:4096:fp8 80:4096:fp8
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workload.schema import load_trace, summarize_trace  # noqa: E402


def compose_files(explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return explicit
    files = [ROOT / "docker-compose.yml"]
    override = ROOT / "docker-compose.override.yml"
    if override.exists():
        files.append(override)
    return files


def compose_cmd(files: list[Path], *args: str) -> list[str]:
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", str(f)]
    return cmd + list(args)


def recreate_container(files: list[Path], env_overrides: dict[str, str]) -> None:
    import os

    env = {**os.environ, **env_overrides}
    subprocess.run(compose_cmd(files, "up", "-d", "--force-recreate"), cwd=ROOT, env=env, check=True)


def wait_healthy(health_url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(5)
    return False


def run_replay(trace: Path, url: str, output: Path, timeout: float,
               tokenizer: str) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "workload" / "replay_trace.py"),
         "--trace", str(trace), "--url", url, "--timeout", str(timeout),
         "--tokenizer", tokenizer, "--output", str(output)],
        cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return json.loads(output.read_text())


def rank_key(candidate: dict):
    """Online score first, followed by the published post-hoc tie-breaks.

    Do not bucket ERS by a noise band here: rounding can reverse two real
    ERS values at bucket boundaries. The raw medians remain the primary
    ordering; p95 TTFT and throughput only decide exact median ties.
    """
    return (-candidate["ers_median"], candidate["p95_ttft_ms_median"],
            -candidate["throughput_tokens_per_s"])


def parse_candidate(value: str) -> tuple[int, int, str]:
    try:
        seqs_text, tokens_text, kv_dtype = value.split(":")
        seqs, tokens = int(seqs_text), int(tokens_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be MAX_NUM_SEQS:MAX_NUM_BATCHED_TOKENS:KV_DTYPE"
        ) from exc
    if seqs <= 0 or tokens <= 0:
        raise argparse.ArgumentTypeError("candidate sequence and token limits must be positive")
    if kv_dtype not in {"fp8", "auto"}:
        raise argparse.ArgumentTypeError("candidate KV_DTYPE must be fp8 or auto")
    return seqs, tokens, kv_dtype


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, default=ROOT / "input" / "trace-descriptor.sample.jsonl")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--health-url", default="http://localhost:8000/health")
    p.add_argument("--max-num-seqs", type=int, nargs="+", default=[64],
                    help="Sequence limits to test (quick default: 64).")
    p.add_argument("--max-num-batched-tokens", type=int, nargs="+",
                    default=[6144, 8192],
                    help="Batch-token limits to test (quick default compares the "
                         "decode-friendly 6144 against the measured 8192 winner).")
    p.add_argument("--kv-cache-dtypes", nargs="+", choices=["fp8", "auto"],
                    default=["fp8"],
                    help="Use 'fp8 auto' to benchmark both the ERS-oriented setting "
                         "and a conservative default-precision fallback.")
    p.add_argument("--candidates", nargs="+", type=parse_candidate,
                    help="Exact SEQS:TOKENS:KV candidates. When provided, bypasses the "
                         "Cartesian product of the three grid arguments.")
    p.add_argument("--repeats", type=int, default=3,
                    help="Runs per candidate (quick default: 3); ranked by median ERS.")
    p.add_argument("--startup-timeout", type=float, default=300.0)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--tokenizer", default="",
                    help="Optional local HF tokenizer name/path for exact prompt token counts. "
                         "Empty uses the deterministic offline approximation.")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    files = compose_files(None)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    trace_records = load_trace(args.trace)
    total_output_tokens = sum(sum(r.output_tokens_per_turn_pinned) for r in trace_records)
    print(f"Trace summary: {json.dumps(summarize_trace(trace_records))}")

    candidate_grid = args.candidates or product(
        args.max_num_seqs, args.max_num_batched_tokens, args.kv_cache_dtypes
    )
    candidates = []
    for seqs, tokens, kv_dtype in candidate_grid:
        name = f"seqs-{seqs}_tokens-{tokens}_kv-{kv_dtype}"
        print(f"\n=== {name} ===")
        recreate_container(files, {
            "MAX_NUM_SEQS": str(seqs),
            "MAX_NUM_BATCHED_TOKENS": str(tokens),
            "KV_CACHE_DTYPE": kv_dtype,
        })

        if not wait_healthy(args.health_url, args.startup_timeout):
            print(f"  container never became healthy, skipping {name}")
            continue

        ers_values, p95_ttft_values, tpot_values, durations = [], [], [], []
        successful_requests, total_requests = [], []
        for run_idx in range(args.repeats):
            out_path = args.results_dir / f"{name}_run{run_idx}.json"
            started = time.monotonic()
            report = run_replay(
                args.trace, args.url, out_path, args.request_timeout, args.tokenizer
            )
            elapsed = time.monotonic() - started
            successful_requests.append(report["success"])
            total_requests.append(report["requests"])
            if report["success"] != report["requests"]:
                print(f"  run {run_idx}: WARNING "
                      f"success={report['success']}/{report['requests']} "
                      f"errors={report['errors']}")
            ers_values.append(report["ers"])
            p95 = report["ttft_ms"]["p95"]
            if p95 is not None:
                p95_ttft_values.append(p95)
            tpot = report["tpot_ms"]["mean"]
            if tpot is not None:
                tpot_values.append(tpot)
            durations.append(elapsed)
            print(f"  run {run_idx}: ers={report['ers']:.4f} success={report['success']}/{report['requests']} "
                  f"p95_ttft_ms={p95} mean_tpot_ms={tpot}")

        if not ers_values:
            continue

        median_duration = statistics.median(durations)
        candidates.append({
            "name": name,
            "max_num_seqs": seqs,
            "max_num_batched_tokens": tokens,
            "kv_cache_dtype": kv_dtype,
            "ers_median": statistics.median(ers_values),
            "p95_ttft_ms_median": statistics.median(p95_ttft_values) if p95_ttft_values else float("inf"),
            "mean_tpot_ms_median": statistics.median(tpot_values) if tpot_values else float("inf"),
            "throughput_tokens_per_s": total_output_tokens / median_duration if median_duration > 0 else 0.0,
            "all_requests_succeeded": all(
                ok == total for ok, total in zip(successful_requests, total_requests)
            ),
            "minimum_successful_requests": min(successful_requests),
            "requests_per_run": max(total_requests),
            "runs": len(ers_values),
        })

    candidates.sort(key=rank_key)

    ranking_path = args.results_dir / "ranking.json"
    ranking_path.write_text(json.dumps(candidates, indent=2))

    print("\n=== Ranking (best first) ===")
    for c in candidates:
        print(f"{c['ers_median']:.4f}\tp95_ttft={c['p95_ttft_ms_median']:.1f}ms\t"
              f"mean_tpot={c['mean_tpot_ms_median']:.3f}ms\t"
              f"stable={c['all_requests_succeeded']}\t"
              f"throughput={c['throughput_tokens_per_s']:.1f} tok/s\t{c['name']}")


if __name__ == "__main__":
    main()
