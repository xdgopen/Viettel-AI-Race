"""Pure ERS / Accuracy Gate scoring math - no I/O, no network.

All thresholds come from config.ers_config; nothing here may hardcode a
number that also appears in docs/requirement.html's scoring section.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from config import ers_config as cfg


def component_score(value_ms: float, floor_ms: float, ceiling_ms: float,
                     gamma: float = cfg.GAMMA) -> float:
    """clamp((ceiling - value) / (ceiling - floor), 0, 1) ** gamma"""
    x = (ceiling_ms - value_ms) / (ceiling_ms - floor_ms)
    return max(0.0, min(1.0, x)) ** gamma


def request_score(ttft_ms: Optional[float], tpot_ms: Optional[float]) -> float:
    """S_request: 0 on error/timeout/0-token, else w*s_ttft + (1-w)*s_tpot."""
    if ttft_ms is None or tpot_ms is None:
        return 0.0
    s_ttft = component_score(ttft_ms, cfg.F_TTFT_MS, cfg.C_TTFT_MS)
    s_tpot = component_score(tpot_ms, cfg.F_TPOT_MS, cfg.C_TPOT_MS)
    return cfg.TTFT_WEIGHT * s_ttft + (1 - cfg.TTFT_WEIGHT) * s_tpot


def accuracy_factor(measured_accuracy: float,
                     baseline_accuracy: float = cfg.BASELINE_ACCURACY_DEFAULT) -> float:
    """f(Delta): piecewise-linear accuracy-drop penalty."""
    drop = baseline_accuracy - measured_accuracy
    if drop <= cfg.ACCURACY_DROP_NO_PENALTY:
        return 1.0
    if drop >= cfg.ACCURACY_DROP_ZERO_SCORE:
        return 0.0
    span = cfg.ACCURACY_DROP_ZERO_SCORE - cfg.ACCURACY_DROP_NO_PENALTY
    return 1.0 - (drop - cfg.ACCURACY_DROP_NO_PENALTY) / span


def final_score(ers: float, accuracy_factor_value: Optional[float]) -> Optional[float]:
    """Score = 100 x ERS x f(Delta). None while f(Delta) is not yet known
    (Accuracy Gate only runs post-online-round, on the team's chosen submissions)."""
    if accuracy_factor_value is None:
        return None
    return 100.0 * ers * accuracy_factor_value


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
