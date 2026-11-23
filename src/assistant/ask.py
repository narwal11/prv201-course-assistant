"""Ask one question from the terminal and print the full execution trace.

    python -m assistant.ask "How does k in k-NN affect overfitting?"
    python -m assistant.ask "Run the Lecture 6 k-NN example with k = 3" --save traces/knn_k3.json

The saved JSON is the inspectable trace we submit as evidence: retrieved chunks
with similarities, every LLM/tool step, sandbox output, flags and latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings, load_dotenv
from .orchestrator import build_orchestrator
from .types import AnswerResult


def print_trace(res: AnswerResult) -> None:
    print(f"\nQ: {res.question}")
    print(f"\n-- retrieval (top {len(res.retrieved)}) --")
    for r in res.retrieved:
        print(f"  {r['tag']}  sim={r['similarity']:.3f}  {r['label']:<24} {r['id']}")
    for i, t in enumerate(res.tool_calls, 1):
        r = t.result
        print(f"\n-- run_python #{i}: exit={r.exit_code} timed_out={r.timed_out} {r.duration_ms} ms --")
        print("  " + t.code.replace("\n", "\n  "))
        if r.stdout:
            print("  stdout: " + r.stdout.strip().replace("\n", "\n          "))
        if r.stderr:
            print("  stderr: " + r.stderr.strip()[-600:].replace("\n", "\n          "))
    print(f"\n-- answer ({'DECLINED' if res.declined else 'answered'}) --\n{res.answer}")
    print(f"\n-- meta -- latency={res.latency_ms} ms  llm_calls={res.llm_calls}  "
          f"tokens in/out={res.input_tokens}/{res.output_tokens}  backend={res.backend}  flags={res.flags}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--lecture", type=int)
    ap.add_argument("--save", type=Path, help="write the full trace as JSON")
    args = ap.parse_args(argv)
    res = build_orchestrator(Settings()).answer(args.question, args.lecture, use_cache=False)
    print_trace(res)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
        print(f"\nTrace saved to {args.save}")


if __name__ == "__main__":
    main()
