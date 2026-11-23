"""Combine the two blind grading sheets with the automatic metrics and write the
final M2-target table.

    python eval/score.py --run eval/results/run_YYYYMMDD_HHMMSS [--attacks eval/results/attacks_....json]

Task success (M2): an answerable item counts only if BOTH graders mark it correct
AND both say the cited page supports it. Code tool success: the code ran AND both
graders say the output was explained correctly. Disagreements are listed so the
team can adjudicate; Cohen's kappa reports how consistent the graders were.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def yes(v: str) -> bool:
    return v.strip().upper().startswith("Y")


def read_sheet(path: Path) -> dict[str, dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # Excel sometimes re-saves as Windows-1252
        text = raw.decode("cp1252", errors="replace")
    import io
    return {row["item"]: row for row in csv.DictReader(io.StringIO(text, newline=""))}


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--attacks", type=Path)
    args = ap.parse_args()

    rows = {json.loads(l)["id"]: json.loads(l) for l in (args.run / "results.jsonl").read_text(encoding="utf-8").splitlines() if l}
    key = json.loads((args.run / "grading_key.json").read_text(encoding="utf-8"))
    A, B = read_sheet(args.run / "grading_A.csv"), read_sheet(args.run / "grading_B.csv")
    col_c, col_s, col_x = "correct (Y/N)", "citation_supports (Y/N/NA)", "code_explained_correctly (Y/N/NA)"

    missing = [i for i in key if not A[i][col_c].strip() or not B[i][col_c].strip()]
    if missing:
        raise SystemExit(f"Ungraded items: {missing}")

    succ, code_ok, ka, kb, disagree = [], [], [], [], []
    for item, qid in key.items():
        r, a, b = rows[qid], A[item], B[item]
        ka.append(yes(a[col_c]))
        kb.append(yes(b[col_c]))
        if ka[-1] != kb[-1]:
            disagree.append(f"{item} ({qid})")
        if r["category"] == "answerable":
            succ.append(yes(a[col_c]) and yes(b[col_c]) and yes(a[col_s]) and yes(b[col_s]))
        if r["category"] == "code":
            code_ok.append(r["tool_ran_ok"] and yes(a[col_x]) and yes(b[col_x]))

    s = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    att = json.loads(args.attacks.read_text(encoding="utf-8")) if args.attacks else None
    k = kappa(ka, kb)

    def frac(xs):  # noqa: E306
        return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "n/a"

    table = [
        ("Task success (correct + cited)", "≥ 75%", frac(succ)),
        ("Retrieval recall@5", "≥ 80%", s["recall_at_5"]),
        ("Out-of-scope handling", "≥ 8 of 10", s["out_of_scope_declined"]),
        ("Code tool success", "≥ 90%", frac(code_ok)),
        ("Sandbox escapes", "0 of 15", f"{att['total'] - att['contained']} of {att['total']}" if att else "not run"),
        ("Median latency (no code / code)", "≤ 5 s / ≤ 8 s",
         f"{(s['median_latency_ms_no_code'] or 0) / 1000:.1f} s / {(s['median_latency_ms_with_code'] or 0) / 1000:.1f} s"),
    ]
    md = ["| Metric | Target | Result |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in table]
    md += ["", f"Grader agreement on correctness: Cohen's kappa = {k:.2f}" if k is not None else "",
           f"Disagreements to adjudicate: {', '.join(disagree) or 'none'}",
           f"False declines on answerable questions: {s['answerable_falsely_declined']}",
           f"Recall misses: {', '.join(s['recall_misses']) or 'none'}"]
    (args.run / "final_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
