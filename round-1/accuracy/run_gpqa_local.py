#!/usr/bin/env python3
"""Best-effort LOCAL accuracy-gate sanity check - NOT the official grader.

This does not reproduce BTC's GPQA Diamond post-online-round evaluation:
the real 100-question set is secret and this script never sees it. It only
exercises the accuracy_factor() math against a small, self-authored
placeholder question set (accuracy/gpqa_subset.jsonl - deliberately NOT real
GPQA content, to avoid touching a benchmark this competition explicitly
treats as a secret held-out eval), so you can sanity-check the
Delta/f(Delta)/Score plumbing and get a rough read on whether a
quantization choice is obviously safe or obviously too aggressive, before
spending one of your <=5 final submission picks.

For a closer (still unofficial) approximation against the public GPQA
Diamond set, see accuracy/LM_EVAL_WIRING.md.

Usage:
    python3 accuracy/run_gpqa_local.py --url http://localhost:8000/v1/chat/completions --ers 0.72
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp

from benchmark.scoring import accuracy_factor, final_score
from config import ers_config as cfg

_ANSWER_RE = re.compile(r"\b([ABCD])\b")


def build_prompt(item: dict) -> str:
    choices = "\n".join(f"{k}. {v}" for k, v in item["choices"].items())
    return (
        f"{item['question']}\n\n{choices}\n\n"
        "Answer with a single letter (A, B, C, or D) and nothing else."
    )


async def ask(session: aiohttp.ClientSession, url: str, model: str, item: dict,
               timeout: float) -> str | None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(item)}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            match = _ANSWER_RE.search(text.strip().upper())
            return match.group(1) if match else None
    except Exception:  # noqa: BLE001 - treat any failure as "no answer", not a crash
        return None


async def run(args: argparse.Namespace) -> None:
    items = [json.loads(line) for line in args.questions.open(encoding="utf-8") if line.strip()]
    async with aiohttp.ClientSession() as session:
        answers = await asyncio.gather(
            *(ask(session, args.url, args.model, item, args.timeout) for item in items)
        )

    answered = sum(1 for a in answers if a is not None)
    correct = sum(1 for item, ans in zip(items, answers) if ans == item["answer"])
    accuracy = correct / len(items) if items else 0.0
    factor = accuracy_factor(accuracy, args.baseline_accuracy)

    print(f"Answered {answered}/{len(items)}; correct {correct}/{len(items)} -> accuracy={accuracy:.3f}")
    print(f"Delta = {args.baseline_accuracy:.3f} - {accuracy:.3f} = {args.baseline_accuracy - accuracy:.3f}")
    print(f"f(Delta) = {factor:.4f}")
    if args.ers is not None:
        score = final_score(args.ers, factor)
        print(f"Estimated Score = 100 x {args.ers:.4f} x {factor:.4f} = {score:.2f}")

    print(
        "\nNOTE: this is a small self-authored placeholder set, NOT the real GPQA "
        "Diamond questions BTC grades with - treat this as a plumbing smoke test, "
        "not a real accuracy estimate. See accuracy/LM_EVAL_WIRING.md for a closer "
        "approximation against the public GPQA Diamond set."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=Path, default=Path(__file__).parent / "gpqa_subset.jsonl")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", default=cfg.SERVED_MODEL_NAME)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--baseline-accuracy", type=float, default=cfg.BASELINE_ACCURACY_DEFAULT)
    p.add_argument("--ers", type=float,
                    help="Optional measured ERS (from workload/replay_trace.py) to fold into an estimated Score.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
