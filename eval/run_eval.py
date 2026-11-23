"""Run the test set through the real system and compute every metric that does not
need a human: retrieval recall@5, out-of-scope handling, false declines, code-tool
run rate, latency, tokens. Also writes two blind grading sheets for the human
metrics (task success, code explained correctly), scored later by eval/score.py.

    python eval/run_eval.py --testset eval/testset.jsonl

Test set format (one JSON object per line):
    {"id": "A01", "category": "answerable" | "code" | "out_of_scope",
     "question": "...", "gold": ["L06_knn.pdf#p12"], "notes": "..."}
`gold` lists the page(s) that contain the answer (any chunk on that page counts).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.config import Settings, load_dotenv  # noqa: E402
from assistant.orchestrator import build_orchestrator  # noqa: E402

CATEGORIES = {"answerable", "code", "out_of_scope"}


def page_key(chunk_id: str) -> str:
    """'L06_knn.pdf#p12#c0' -> 'L06_knn.pdf#p12'"""
    return chunk_id.rsplit("#c", 1)[0]


def load_testset(path: Path) -> list[dict]:
    items, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        it = json.loads(line)
        if it.get("category") not in CATEGORIES:
            raise ValueError(f"line {n}: category must be one of {CATEGORIES}")
        if it["id"] in seen:
            raise ValueError(f"line {n}: duplicate id {it['id']}")
        if it["category"] == "answerable" and not it.get("gold"):
            raise ValueError(f"line {n}: answerable items need a gold page")
        seen.add(it["id"])
        items.append(it)
    return items


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    return {"min": round(xs[0], 3), "p25": round(xs[len(xs) // 4], 3), "median": round(statistics.median(xs), 3),
            "max": round(xs[-1], 3)}


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=ROOT / "eval" / "testset.jsonl")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    items = load_testset(args.testset)[: args.limit]
    out = args.out or ROOT / "eval" / "results" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    orch = build_orchestrator(settings)

    rows = []
    for i, it in enumerate(items, 1):
        res = orch.answer(it["question"], it.get("lecture"), use_cache=False)
        retrieved_pages = [page_key(r["id"]) for r in res.retrieved]
        gold = set(it.get("gold") or [])
        row = {
            **it,
            "answer": res.answer, "raw_answer": res.raw_answer, "declined": res.declined, "flags": res.flags,
            "latency_ms": res.latency_ms, "llm_calls": res.llm_calls,
            "tokens_in": res.input_tokens, "tokens_out": res.output_tokens,
            "top_sim": res.retrieved[0]["similarity"] if res.retrieved else None,
            "retrieved": [(r["id"], r["similarity"]) for r in res.retrieved],
            "recall_hit": bool(gold & set(retrieved_pages)) if gold else None,
            "cited": [c["label"] + f" ({c['source']})" for c in res.citations],
            "cited_text": [r["text"] for r in res.retrieved if r["id"] in {c["id"] for c in res.citations}],
            "tool_calls": len(res.tool_calls),
            "tool_ran_ok": any(t.result.ok for t in res.tool_calls),
            "tool_detail": [{"code": t.code, "ok": t.result.ok, "stdout": t.result.stdout[-500:],
                             "stderr": t.result.stderr[-300:]} for t in res.tool_calls],
        }
        rows.append(row)
        print(f"[{i}/{len(items)}] {it['id']:<5} {it['category']:<12} hit={row['recall_hit']!s:<5} "
              f"declined={res.declined!s:<5} tools={row['tool_calls']} {res.latency_ms} ms {res.flags}")

    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # ---------------- automatic metrics
    ans = [r for r in rows if r["category"] == "answerable"]
    code = [r for r in rows if r["category"] == "code"]
    oos = [r for r in rows if r["category"] == "out_of_scope"]
    with_gold = [r for r in rows if r["recall_hit"] is not None]
    lat_plain = [r["latency_ms"] for r in rows if r["tool_calls"] == 0 and "below_cutoff" not in r["flags"]]
    lat_code = [r["latency_ms"] for r in rows if r["tool_calls"] > 0]
    summary = {
        "n_items": len(rows),
        "config": {"model": settings.model, "embed_model": settings.embed_model, "top_k": settings.top_k,
                   "sim_cutoff": settings.sim_cutoff, "chunk_words": settings.chunk_words,
                   "chunks_in_store": orch.store.count()},
        "recall_at_5": pct(sum(r["recall_hit"] for r in with_gold), len(with_gold)),
        "out_of_scope_declined": pct(sum(r["declined"] for r in oos), len(oos)),
        "answerable_falsely_declined": pct(sum(r["declined"] for r in ans), len(ans)),
        "code_questions_tool_called": pct(sum(r["tool_calls"] > 0 for r in code), len(code)),
        "code_questions_tool_ran_ok": pct(sum(r["tool_ran_ok"] for r in code), len(code)),
        "uncited_answers": pct(sum("uncited" in r["flags"] for r in rows if not r["declined"]),
                               sum(not r["declined"] for r in rows)),
        "invalid_citations": sum("invalid_citation" in r["flags"] for r in rows),
        "median_latency_ms_no_code": statistics.median(lat_plain) if lat_plain else None,
        "median_latency_ms_with_code": statistics.median(lat_code) if lat_code else None,
        "tokens_total": {"in": sum(r["tokens_in"] for r in rows), "out": sum(r["tokens_out"] for r in rows)},
        # use these two distributions to pick SIM_CUTOFF
        "top_sim_answerable": quantiles([r["top_sim"] for r in ans if r["top_sim"] is not None]),
        "top_sim_out_of_scope": quantiles([r["top_sim"] for r in oos if r["top_sim"] is not None]),
        "recall_misses": [r["id"] for r in with_gold if not r["recall_hit"]],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- blind grading sheets (same shuffled order, no system metadata)
    order = list(range(len(rows)))
    random.Random(201).shuffle(order)
    key = {}
    header = ["item", "question", "answer", "cited_sources", "cited_text", "gold_source", "code_run",
              "correct (Y/N)", "citation_supports (Y/N/NA)", "code_explained_correctly (Y/N/NA)", "comment"]
    for grader in ("A", "B"):
        with (out / f"grading_{grader}.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for n, idx in enumerate(order, 1):
                r = rows[idx]
                key[f"Q{n:02d}"] = r["id"]
                code_txt = "\n---\n".join(f"{t['code']}\n=> {t['stdout'] or t['stderr']}" for t in r["tool_detail"])
                w.writerow([f"Q{n:02d}", r["question"], r["answer"], "; ".join(r["cited"]),
                            "\n---\n".join(r["cited_text"]), "; ".join(r.get("gold") or []), code_txt,
                            "", "", "", ""])
    (out / "grading_key.json").write_text(json.dumps(key, indent=2))

    print("\n" + json.dumps({k: v for k, v in summary.items() if not k.startswith("top_sim")}, indent=2))
    print(f"\nSaved to {out}. Next: each grader fills in grading_A.csv / grading_B.csv separately, then\n"
          f"  python eval/score.py --run {out}")


if __name__ == "__main__":
    main()
