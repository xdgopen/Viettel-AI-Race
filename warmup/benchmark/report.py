"""Shared aggregate-report builder, used by workload/replay_trace.py and
sweep/sweep_params.py so the report shape never drifts between the two."""

from __future__ import annotations

import statistics
from typing import Optional

from benchmark.scoring import final_score, percentile


def build_report(results: list[dict], baseline_accuracy: Optional[float] = None,
                  accuracy_factor_value: Optional[float] = None) -> dict:
    """results: list of {"status": str, "score": float, "ttft_ms": float|None, "tpot_ms": float|None}."""
    ok = [r for r in results if r["status"] == "SUCCESS"]
    ers = statistics.fmean(r["score"] for r in results) if results else 0.0

    errors: dict[str, int] = {}
    for r in results:
        if r["status"] != "SUCCESS":
            errors[r["status"]] = errors.get(r["status"], 0) + 1

    return {
        "requests": len(results),
        "success": len(ok),
        "errors": errors,
        "ers": ers,
        "score_without_accuracy": 100.0 * ers,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_factor": accuracy_factor_value,
        "score": final_score(ers, accuracy_factor_value),
        "ttft_ms": _stats([r["ttft_ms"] for r in ok]),
        "tpot_ms": _stats([r["tpot_ms"] for r in ok]),
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }
