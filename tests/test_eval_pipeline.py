"""End-to-end check of eval/run_eval.py -> grading sheets -> eval/score.py using
test doubles, so the harness itself is known to work before the real run."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402
import score  # noqa: E402
from assistant.orchestrator import Orchestrator  # noqa: E402
from conftest import FakeSandbox, text, tool  # noqa: E402
from assistant.llm import ScriptedLLM  # noqa: E402


def test_eval_and_score(store, tmp_path, monkeypatch):
    ts = tmp_path / "ts.jsonl"
    ts.write_text(encoding="utf-8", data="\n".join(json.dumps(x) for x in [
        {"id": "A1", "category": "answerable", "question": "How does the choice of k affect overfitting in k-NN?",
         "gold": ["L06_knn_sample.pdf#p2"]},
        {"id": "C1", "category": "code", "question": "Run the k-NN lab example with k = 3",
         "gold": ["L06_knn_lab_sample.ipynb#p3"]},
        {"id": "O1", "category": "out_of_scope", "question": "Who won the 2026 World Cup?"},
    ]))
    llm = ScriptedLLM([text("Small k overfits [S1]."), tool("print(0.98)"), text("Accuracy 0.98 [S1].")])
    orch = Orchestrator(store, llm, FakeSandbox(), sim_cutoff=0.1)  # hashing-embedder scale; OOS item scores 0.07
    monkeypatch.setattr(run_eval, "build_orchestrator", lambda s: orch)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_eval", "--testset", str(ts), "--out", str(out)])
    run_eval.main()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["recall_at_5"].startswith("2/2")
    assert summary["out_of_scope_declined"].startswith("1/1")       # declined by the cutoff
    assert summary["code_questions_tool_ran_ok"].startswith("1/1")

    for g in ("A", "B"):  # graders mark everything correct
        p = out / f"grading_{g}.csv"
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        for r in rows:
            r["correct (Y/N)"], r["citation_supports (Y/N/NA)"], r["code_explained_correctly (Y/N/NA)"] = "Y", "Y", "Y"
        with p.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    monkeypatch.setattr(sys, "argv", ["score", "--run", str(out)])
    score.main()
    md = (out / "final_metrics.md").read_text(encoding="utf-8")
    assert "| Task success (correct + cited) | ≥ 75% | 1/1 (100%) |" in md
    assert "| Code tool success | ≥ 90% | 1/1 (100%) |" in md
