#!/usr/bin/env python3
"""Generate a multi-turn conversation trace matching the schema in
workload/schema.py, for local dev/tuning.

Two modes:
- `--from-spec <path>`: expand a published global scalar descriptor (e.g.
  docs/grading-workload-spec.json - num_conversations, user_turns_per_conversation,
  shared_system_prefix_tokens, per_conversation_prefix_tokens,
  new_user_tokens_per_turn, output_tokens_per_turn_pinned, arrival) into one
  homogeneous ConversationRecord per conversation. Use this whenever a real
  spec file is available - it's authoritative over the random-range mode.
- Default (no --from-spec): sample random per-conversation/per-turn
  variation from CLI-provided ranges, for stress-testing shapes the
  published spec doesn't cover.

The trace only stores token *counts*; literal prompt text is synthesized at
replay time by workload/text_fill.py, so this generator has no tokenizer
dependency and just samples/expands integers.

Usage:
    python3 workload/generate_trace.py --from-spec ../docs/grading-workload-spec.json \
        --arrival-rate-per-sec 2.0 --output input/trace-descriptor.sample.jsonl
    python3 workload/generate_trace.py --num-conversations 40 --output input/trace-descriptor.sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workload.schema import (
    ConversationRecord,
    expand_global_descriptor,
    save_trace,
    summarize_trace,
    validate_global_descriptor,
    validate_trace,
)


def generate(args: argparse.Namespace) -> list[ConversationRecord]:
    rng = random.Random(args.seed)
    records: list[ConversationRecord] = []

    # Poisson-process arrivals: exponential inter-arrival gaps between
    # conversation start times, at the requested mean rate.
    arrival_ms = 0.0
    for cid in range(args.num_conversations):
        if cid > 0:
            gap_s = rng.expovariate(args.arrival_rate_per_sec)
            arrival_ms += gap_s * 1000.0

        turns = rng.randint(args.min_turns, args.max_turns)
        records.append(ConversationRecord(
            conversation_id=cid,
            arrival_ms=round(arrival_ms, 1),
            shared_system_prefix_tokens=args.shared_system_prefix_tokens,
            per_conversation_prefix_tokens=rng.randint(*args.per_conversation_prefix_tokens_range),
            new_user_tokens_per_turn=[rng.randint(*args.new_user_tokens_range) for _ in range(turns)],
            output_tokens_per_turn_pinned=[args.output_tokens_pinned] * turns,
            think_time_ms=args.think_time_ms,
        ))

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-conversations", type=int, default=40)
    p.add_argument("--min-turns", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--shared-system-prefix-tokens", type=int, default=512)
    p.add_argument("--per-conversation-prefix-tokens-range", type=int, nargs=2, default=(100, 400),
                    metavar=("MIN", "MAX"))
    p.add_argument("--new-user-tokens-range", type=int, nargs=2, default=(60, 200), metavar=("MIN", "MAX"))
    p.add_argument("--output-tokens-pinned", type=int, default=200)
    p.add_argument("--arrival-rate-per-sec", type=float, default=2.0,
                    help="Mean rate of new conversations starting, per second (Poisson process).")
    p.add_argument("--think-time-ms", type=float, default=0.0,
                    help="Delay between receiving turn N's reply and sending turn N+1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-spec", type=Path,
                    help="Path to a published global scalar descriptor JSON "
                         "(e.g. docs/grading-workload-spec.json). Overrides all "
                         "random-range options above.")
    p.add_argument("--output", type=Path, default=Path("input/trace-descriptor.sample.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_spec:
        spec = json.loads(args.from_spec.read_text(encoding="utf-8"))
        for w in validate_global_descriptor(spec):
            print(f"WARNING: {w}", file=sys.stderr)
        records = expand_global_descriptor(spec, arrival_rate_per_sec=args.arrival_rate_per_sec)
    else:
        records = generate(args)

    for w in validate_trace(records):
        print(f"WARNING: {w}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trace(records, args.output)

    summary = summarize_trace(records)
    print(f"Wrote {len(records)} conversations to {args.output}")
    print(f"total_request={summary['total_request']}  "
          f"worst_case_context_tokens={summary['worst_case_context_tokens']}")


if __name__ == "__main__":
    main()
