"""Synthesizes literal prompt text of an exact (or approximate) token count.

The trace schema (workload/schema.py) only records token *counts* - actual
text is generated here at replay time. Token counts are best-effort: text is
trimmed to an exact tokenizer ID count, but the live server re-tokenizes the
full request (system prompt + chat template + this text), which can shift
the effective count by a few tokens. That's acceptable for local
stress-testing; it does not need to reproduce the organizers' opaque
official trace exactly.
"""

from __future__ import annotations

import random
import sys

_WORD_POOL = (
    "system model request latency throughput token cache batch schedule "
    "context window kernel attention decode prefill memory allocate stream "
    "response query answer summarize analyze compare explain describe list "
    "function dataset pipeline feature vector matrix optimize accuracy score "
    "server client network gpu cpu queue worker retry timeout concurrent"
).split()


class TokenCounter:
    """Fills text to an exact token count using a real HF tokenizer. Falls
    back to a word-count approximation (~0.72 words per token) if the
    tokenizer can't be loaded (offline, no `transformers`, no cached weights)."""

    def __init__(self, model_name: str | None):
        self._tokenizer = None
        if model_name:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as exc:  # noqa: BLE001 - any load failure -> fallback
                print(f"[text_fill] tokenizer load failed ({exc}); "
                      f"falling back to word-count approximation", file=sys.stderr)

    @property
    def is_exact(self) -> bool:
        return self._tokenizer is not None

    def fill(self, target_tokens: int, rng: random.Random) -> str:
        if target_tokens <= 0:
            return ""
        if self._tokenizer is None:
            n_words = max(1, round(target_tokens * 0.72))
            return " ".join(rng.choice(_WORD_POOL) for _ in range(n_words))

        pieces: list[str] = []
        ids: list[int] = []
        while len(ids) < target_tokens:
            pieces.append(rng.choice(_WORD_POOL))
            ids = self._tokenizer.encode(" ".join(pieces), add_special_tokens=False)
        ids = ids[:target_tokens]
        return self._tokenizer.decode(ids)
