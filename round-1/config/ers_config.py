"""Single source of truth for AI Race 2026 - Bai 3 scoring constants.

Every value here must trace back to a row in docs/requirement.html's
"Tham so cau hinh" table (section 2, "Cach tinh diem"). Nothing outside
this module may hardcode one of these numbers - import from here instead,
so a future spec revision only requires editing one file.

Spec version: docs/requirement.html, update ngay 18/07/2026.
"""

# --- ERS latency scoring (requirement.html S2, "Tham so cau hinh") ---
F_TTFT_MS: float = 10.0     # Floor cua TTFT
C_TTFT_MS: float = 400.0    # Ceiling cua TTFT
F_TPOT_MS: float = 1.0      # Floor cua TPOT
C_TPOT_MS: float = 10.0     # Ceiling cua TPOT
GAMMA: float = 2.0          # He so luy thua
TTFT_WEIGHT: float = 0.5    # Trong so cua TTFT (w)

# --- Accuracy Gate (requirement.html S2, "Accuracy Gate - sau vong online") ---
BASELINE_ACCURACY_DEFAULT: float = 0.40   # baseline BF16 GPQA Diamond accuracy
ACCURACY_DROP_NO_PENALTY: float = 0.10    # Delta <= this  -> f(Delta) = 1.0
ACCURACY_DROP_ZERO_SCORE: float = 0.16    # Delta >= this  -> f(Delta) = 0.0

# --- Tie-break order, applied post-hoc when scores are within noise (requirement.html S6) ---
TIE_BREAK_ORDER = (
    "accuracy_drop",    # 1. Muc do suy giam do chinh xac
    "p95_ttft",         # 2. Chi so p95 TTFT
    "throughput",       # 3. Toc do sinh van ban
    "submission_time",  # 4. Thoi diem nop bai (uu tien som hon)
)

# --- Fixed model facts (requirement.html S1) ---
MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
SERVED_MODEL_NAME = "LFM2.5-1.2B-Instruct"

SPEC_VERSION = "docs/requirement.html (update 18/07/2026)"
