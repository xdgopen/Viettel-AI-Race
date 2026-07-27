"""Typed schema for the multi-turn conversation workload trace described in
docs/requirement.html section 1 ("Mo ta cac gia tri trong file trace").

Design choices (see docs/PLANNING.md section 5, open questions #1/#2/#4):

- The trace is one JSONL row per conversation, not a single global
  descriptor object. Each row is self-contained; num_conversations and
  total_request from the spec are *derived* properties of the file (row
  count, sum of per-row turn counts) rather than literal fields - see
  summarize_trace().
- Turn sequencing is closed-loop: turn N+1 of a conversation is only sent
  after turn N's full response has been received, like a real chat client -
  it needs turn N's assistant reply to build turn N+1's message history.
  `arrival_ms` describes when the *conversation* becomes eligible to send
  its first (turn-1) request, not a per-turn schedule.
- `total_request` reconciliation against num_conversations x
  user_turns_per_conversation is a non-fatal warning (validate_trace), not
  a hard assertion, since turn counts may legitimately vary per conversation.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversationRecord:
    conversation_id: int
    arrival_ms: float
    shared_system_prefix_tokens: int
    per_conversation_prefix_tokens: int
    new_user_tokens_per_turn: list[int]
    output_tokens_per_turn_pinned: list[int]
    # Optional delay between receiving turn N's reply and sending turn N+1.
    # Not named in requirement.html; defaults to 0 (fire again immediately).
    think_time_ms: float = 0.0

    @property
    def user_turns(self) -> int:
        return len(self.new_user_tokens_per_turn)

    def to_json(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "arrival_ms": self.arrival_ms,
            "shared_system_prefix_tokens": self.shared_system_prefix_tokens,
            "per_conversation_prefix_tokens": self.per_conversation_prefix_tokens,
            "new_user_tokens_per_turn": self.new_user_tokens_per_turn,
            "output_tokens_per_turn_pinned": self.output_tokens_per_turn_pinned,
            "think_time_ms": self.think_time_ms,
        }

    @staticmethod
    def from_json(row: dict) -> "ConversationRecord":
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            arrival_ms=row["arrival_ms"],
            shared_system_prefix_tokens=row["shared_system_prefix_tokens"],
            per_conversation_prefix_tokens=row["per_conversation_prefix_tokens"],
            new_user_tokens_per_turn=list(row["new_user_tokens_per_turn"]),
            output_tokens_per_turn_pinned=list(row["output_tokens_per_turn_pinned"]),
            think_time_ms=row.get("think_time_ms", 0.0),
        )


def load_trace(path: Path) -> list[ConversationRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ConversationRecord.from_json(json.loads(line)))
    return records


def save_trace(records: list[ConversationRecord], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def validate_trace(records: list[ConversationRecord]) -> list[str]:
    """Non-fatal structural checks. Returns a list of warning strings;
    an empty list means the trace looks internally consistent."""
    if not records:
        return ["trace is empty"]

    warnings: list[str] = []
    prefixes = {r.shared_system_prefix_tokens for r in records}
    if len(prefixes) > 1:
        warnings.append(
            "shared_system_prefix_tokens should be identical across all conversations "
            f"(it's shared), found {len(prefixes)} distinct values: {sorted(prefixes)}"
        )

    for r in records:
        if len(r.output_tokens_per_turn_pinned) != r.user_turns:
            warnings.append(
                f"conversation {r.conversation_id}: new_user_tokens_per_turn has "
                f"{r.user_turns} entries but output_tokens_per_turn_pinned has "
                f"{len(r.output_tokens_per_turn_pinned)}"
            )
        if any(t <= 0 for t in r.new_user_tokens_per_turn):
            warnings.append(f"conversation {r.conversation_id}: non-positive user token count")
        if any(t <= 0 for t in r.output_tokens_per_turn_pinned):
            warnings.append(f"conversation {r.conversation_id}: non-positive pinned output length")

    return warnings


def summarize_trace(records: list[ConversationRecord]) -> dict:
    """Derives the aggregate fields named in requirement.html
    (num_conversations, total_request, user_turns_per_conversation) from the
    per-conversation rows, plus a worst-case context-length estimate that
    should drive --max-model-len sizing (see docs/PLANNING.md open question #6)."""
    if not records:
        return {"num_conversations": 0, "total_request": 0}

    turn_counts = [r.user_turns for r in records]
    total_request = sum(turn_counts)

    return {
        "num_conversations": len(records),
        "total_request": total_request,
        "user_turns_per_conversation": {
            "min": min(turn_counts),
            "max": max(turn_counts),
            "mean": total_request / len(records),
        },
        "shared_system_prefix_tokens": records[0].shared_system_prefix_tokens,
        "worst_case_context_tokens": max(_max_context_tokens(r) for r in records),
    }


def _max_context_tokens(r: ConversationRecord) -> int:
    """Running context length at the conversation's final turn: shared
    system prefix + per-conversation prefix + every prior turn's user and
    pinned-output tokens (closed-loop history keeps growing turn over turn)."""
    running = r.shared_system_prefix_tokens + r.per_conversation_prefix_tokens
    for user_tokens, output_tokens in zip(r.new_user_tokens_per_turn, r.output_tokens_per_turn_pinned):
        running += user_tokens + output_tokens
    return running


def expand_global_descriptor(spec: dict, arrival_rate_per_sec: float,
                              seed: int | None = None) -> list[ConversationRecord]:
    """Expands a single global scalar descriptor - the format the
    organizers actually publish (see docs/grading-workload-spec.json) - into
    one ConversationRecord per conversation.

    Unlike the random-range sampling in generate_trace.generate(), every
    conversation here is homogeneous: identical turn count, identical
    per-turn user/output token counts. Only arrival time differs between
    conversations (and the *content*, not the token count, of each
    conversation's per_conversation_prefix_tokens block - see
    workload/text_fill.py, which seeds its filler text by conversation_id).

    `seed` defaults to whatever's embedded in spec["arrival"] (e.g.
    "Poisson, seed 42"), falling back to 42 if unparseable. The arrival
    *rate* is not given by the spec file - it must be supplied explicitly.
    """
    num_conversations = spec["num_conversations"]
    turns = spec["user_turns_per_conversation"]
    new_user_tokens = spec["new_user_tokens_per_turn"]
    pinned_output = spec["output_tokens_per_turn_pinned"]

    if seed is None:
        match = re.search(r"seed\s+(\d+)", str(spec.get("arrival", "")), re.IGNORECASE)
        seed = int(match.group(1)) if match else 42

    rng = random.Random(seed)
    records = []
    arrival_ms = 0.0
    for cid in range(num_conversations):
        if cid > 0:
            arrival_ms += rng.expovariate(arrival_rate_per_sec) * 1000.0
        records.append(ConversationRecord(
            conversation_id=cid,
            arrival_ms=round(arrival_ms, 1),
            shared_system_prefix_tokens=spec["shared_system_prefix_tokens"],
            per_conversation_prefix_tokens=spec["per_conversation_prefix_tokens"],
            new_user_tokens_per_turn=[new_user_tokens] * turns,
            output_tokens_per_turn_pinned=[pinned_output] * turns,
        ))

    return records


def validate_global_descriptor(spec: dict) -> list[str]:
    """Non-fatal check that the spec's declared total_requests reconciles
    with num_conversations x user_turns_per_conversation."""
    total_declared = spec.get("total_requests", spec.get("total_request"))
    expected = spec["num_conversations"] * spec["user_turns_per_conversation"]
    if total_declared is not None and total_declared != expected:
        return [f"spec declares total_requests={total_declared} but "
                f"num_conversations x user_turns_per_conversation = {expected}"]
    return []


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/trace-descriptor.sample.jsonl")
    recs = load_trace(path)
    for w in validate_trace(recs):
        print(f"WARNING: {w}")
    print(json.dumps(summarize_trace(recs), indent=2))
