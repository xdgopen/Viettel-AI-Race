#!/usr/bin/env python3
"""Grid-search vLLM serving parameters against a fixed trace, ranking
candidates by the spec's tie-break order (requirement.html section 6):
ERS first, then p95 TTFT, then generation throughput.

Python successor to the old tune_vast.sh: same grid-search intent, but
drives docker compose via subprocess and imports benchmark/report.py and
config/ers_config.py directly instead of an inline bash/heredoc JSON parser.

Usage:
    python3 sweep/sweep_params.py --max-num-seqs 16 24 32 \
        --max-num-batched-tokens 4096 8192 16384 --repeats 3
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


def run_replay(trace: Path, url: str, output: Path, timeout: float) -> dict:
    subprocess.run(
        [sys.executable, str(ROOT / "workload" / "replay_trace.py"),
         "--trace", str(trace), "--url", url, "--timeout", str(timeout),
         "--tokenizer", "", "--output", str(output)],
        cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return json.loads(output.read_text())


def rank_key(candidate: dict, noise_band: float):
    """Sorts primarily by ERS (descending); within `noise_band` of the best
    ERS seen so far, falls back to the spec's tie-break order (p95 TTFT
    ascending, then throughput descending)."""
    return (-round(candidate["ers_median"] / noise_band), candidate["p95_ttft_ms_median"],
            -candidate["throughput_tokens_per_s"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, default=ROOT / "input" / "trace-descriptor.sample.jsonl")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--health-url", default="http://localhost:8000/health")
    p.add_argument("--max-num-seqs", type=int, nargs="+", default=[16, 32, 48, 64],
                    help="Widened vs. the old 16/24/32 range: --max-model-len dropped to 6144 "
                         "(from 32768), so each sequence's KV cache footprint shrank ~5x and "
                         "materially higher concurrency should now fit in 18GB VRAM.")
    p.add_argument("--max-num-batched-tokens", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--repeats", type=int, default=3, help="Runs per candidate; ranked by median ERS.")
    p.add_argument("--startup-timeout", type=float, default=300.0)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--noise-band", type=float, default=0.01,
                    help="ERS values within this of each other are treated as tied "
                         "for ranking purposes (requirement.html's noise-band tie-break).")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    files = compose_files(None)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    trace_records = load_trace(args.trace)
    total_output_tokens = sum(sum(r.output_tokens_per_turn_pinned) for r in trace_records)
    print(f"Trace summary: {json.dumps(summarize_trace(trace_records))}")

    candidates = []
    for seqs, tokens in product(args.max_num_seqs, args.max_num_batched_tokens):
        name = f"seqs-{seqs}_tokens-{tokens}"
        print(f"\n=== {name} ===")
        recreate_container(files, {"MAX_NUM_SEQS": str(seqs), "MAX_NUM_BATCHED_TOKENS": str(tokens)})

        if not wait_healthy(args.health_url, args.startup_timeout):
            print(f"  container never became healthy, skipping {name}")
            continue

        ers_values, p95_ttft_values, durations = [], [], []
        for run_idx in range(args.repeats):
            out_path = args.results_dir / f"{name}_run{run_idx}.json"
            started = time.monotonic()
            report = run_replay(args.trace, args.url, out_path, args.request_timeout)
            elapsed = time.monotonic() - started
            ers_values.append(report["ers"])
            p95 = report["ttft_ms"]["p95"]
            if p95 is not None:
                p95_ttft_values.append(p95)
            durations.append(elapsed)
            print(f"  run {run_idx}: ers={report['ers']:.4f} success={report['success']}/{report['requests']} "
                  f"p95_ttft_ms={p95}")

        if not ers_values:
            continue

        median_duration = statistics.median(durations)
        candidates.append({
            "name": name,
            "max_num_seqs": seqs,
            "max_num_batched_tokens": tokens,
            "ers_median": statistics.median(ers_values),
            "p95_ttft_ms_median": statistics.median(p95_ttft_values) if p95_ttft_values else float("inf"),
            "throughput_tokens_per_s": total_output_tokens / median_duration if median_duration > 0 else 0.0,
            "runs": len(ers_values),
        })

    candidates.sort(key=lambda c: rank_key(c, args.noise_band))

    ranking_path = args.results_dir / "ranking.json"
    ranking_path.write_text(json.dumps(candidates, indent=2))

    print("\n=== Ranking (best first) ===")
    for c in candidates:
        print(f"{c['ers_median']:.4f}\tp95_ttft={c['p95_ttft_ms_median']:.1f}ms\t"
              f"throughput={c['throughput_tokens_per_s']:.1f} tok/s\t{c['name']}")


if __name__ == "__main__":
    main()
