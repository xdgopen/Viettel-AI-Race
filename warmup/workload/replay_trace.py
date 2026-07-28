#!/usr/bin/env python3
"""Conversation-aware trace replay client and local ERS estimator.

Unlike the old benchmark_local.py (fired one pre-built request body per
trace row, with no notion of a conversation), this replays the multi-turn
schema in workload/schema.py: each conversation's turns are issued in
sequence, carrying forward real message history (system prompt + prior
turns' user/assistant messages), closed-loop - turn N+1 is only sent after
turn N's response is fully received (see docs/PLANNING.md open question #1).

Usage:
    python3 workload/replay_trace.py --trace input/trace-descriptor.sample.jsonl \
        --url http://localhost:8000/v1/chat/completions --output results/run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp

from benchmark.report import build_report
from benchmark.scoring import accuracy_factor, request_score
from config import ers_config as cfg
from workload.schema import ConversationRecord, load_trace, validate_trace
from workload.text_fill import TokenCounter


async def _stream_completion(session: aiohttp.ClientSession, url: str, body: dict,
                              timeout: float) -> dict:
    """POSTs one chat-completion turn and measures TTFT/TPOT from SSE token
    events. Returns {"status", "ttft_ms", "tpot_ms", "reply_text"}."""
    started = time.perf_counter()
    first_token_at: float | None = None
    token_events: list[float] = []
    reply_parts: list[str] = []
    buffer = ""

    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                detail = (await response.text())[:300].replace("\n", " ")
                return {"status": f"HTTP_{response.status}: {detail}", "ttft_ms": None,
                        "tpot_ms": None, "reply_text": ""}

            async for chunk in response.content.iter_any():
                # Network chunks do not necessarily align with SSE lines.
                buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        data = json.loads(raw)
                        choices = data.get("choices") or []
                        content = choices[0].get("delta", {}).get("content") if choices else None
                        if content:
                            now = time.perf_counter()
                            first_token_at = first_token_at or now
                            token_events.append(now)
                            reply_parts.append(content)

        if not token_events:
            return {"status": "EMPTY_RESPONSE", "ttft_ms": None, "tpot_ms": None, "reply_text": ""}

        ttft_ms = (first_token_at - started) * 1000.0
        tpot_ms = (
            (token_events[-1] - token_events[0]) / (len(token_events) - 1) * 1000.0
            if len(token_events) > 1 else 0.0
        )
        return {"status": "SUCCESS", "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
                "reply_text": "".join(reply_parts)}
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
        return {"status": "TIMEOUT", "ttft_ms": None, "tpot_ms": None, "reply_text": ""}
    except Exception as exc:  # noqa: BLE001 - report as a scored failure, don't crash the run
        return {"status": f"EXCEPTION_{type(exc).__name__}: {exc}", "ttft_ms": None,
                "tpot_ms": None, "reply_text": ""}


async def _replay_conversation(session: aiohttp.ClientSession, url: str, model: str,
                                record: ConversationRecord, system_text: str,
                                counter: TokenCounter, benchmark_start: float,
                                timeout: float) -> list[dict]:
    due = benchmark_start + record.arrival_ms / 1000.0
    await asyncio.sleep(max(0.0, due - time.perf_counter()))

    rng = random.Random(record.conversation_id)
    messages = [{"role": "system", "content": system_text}] if system_text else []
    results: list[dict] = []

    for turn_idx, (user_tokens, pinned_output) in enumerate(
        zip(record.new_user_tokens_per_turn, record.output_tokens_per_turn_pinned)
    ):
        user_text = counter.fill(user_tokens, rng)
        if turn_idx == 0 and record.per_conversation_prefix_tokens > 0:
            prefix_text = counter.fill(record.per_conversation_prefix_tokens, rng)
            user_text = f"{prefix_text}\n\n{user_text}"
        messages.append({"role": "user", "content": user_text})

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": pinned_output,
            "min_tokens": pinned_output,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
        }
        outcome = await _stream_completion(session, url, body, timeout)
        results.append({
            "conversation_id": record.conversation_id,
            "turn": turn_idx,
            "status": outcome["status"],
            "ttft_ms": outcome["ttft_ms"],
            "tpot_ms": outcome["tpot_ms"],
            "score": request_score(outcome["ttft_ms"], outcome["tpot_ms"]),
        })

        # Placeholder keeps history well-formed if a turn fails, so later
        # turns in the same conversation can still be attempted and scored.
        messages.append({"role": "assistant", "content": outcome["reply_text"] or ""})

        if record.think_time_ms > 0:
            await asyncio.sleep(record.think_time_ms / 1000.0)

    return results


async def run(args: argparse.Namespace) -> int:
    records = load_trace(args.trace)
    if not records:
        raise SystemExit("Trace is empty")
    for w in validate_trace(records):
        print(f"WARNING: {w}", file=sys.stderr)

    counter = TokenCounter(args.tokenizer)
    rng = random.Random("shared-system-prefix")
    system_text = counter.fill(records[0].shared_system_prefix_tokens, rng) if records[0].shared_system_prefix_tokens else ""

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    benchmark_start = time.perf_counter() + args.warmup_delay
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(_replay_conversation(
                session, args.url, args.model, record, system_text, counter,
                benchmark_start, args.timeout,
            ))
            for record in records
        ]
        per_conversation_results = await asyncio.gather(*tasks)

    results = [r for conv_results in per_conversation_results for r in conv_results]

    factor = accuracy_factor(args.accuracy, args.baseline_accuracy) if args.accuracy is not None else None
    report = build_report(results, baseline_accuracy=args.baseline_accuracy, accuracy_factor_value=factor)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**report, "results": results}, indent=2), encoding="utf-8")

    return 0 if report["success"] == report["requests"] else 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, default=Path("input/trace-descriptor.sample.jsonl"))
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", default=cfg.SERVED_MODEL_NAME)
    p.add_argument("--tokenizer", default=cfg.MODEL_NAME,
                    help="HF tokenizer used to render prompt text; falls back to an approximation if unavailable.")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-turn request timeout (seconds).")
    p.add_argument("--warmup-delay", type=float, default=0.25)
    p.add_argument("--baseline-accuracy", type=float, default=cfg.BASELINE_ACCURACY_DEFAULT)
    p.add_argument("--accuracy", type=float, help="Optional measured GPQA accuracy in [0,1]")
    p.add_argument("--output", type=Path)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
