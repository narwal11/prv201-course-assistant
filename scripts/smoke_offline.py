"""Offline smoke run: exercises the real ingestion, retrieval, orchestrator loop,
citation checking, logging and a real Python execution, with NO API key and NO
Docker. The LLM is scripted (its two replies are fixed below) and the code runs in
the unsafe local runner, so this proves the plumbing, not answer quality.
For real traces use:  python -m assistant.ask "..." --save traces/x.json

    ALLOW_UNSAFE_LOCAL_SANDBOX=1 python scripts/smoke_offline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant.ask import print_trace  # noqa: E402
from assistant.embeddings import HashingEmbedder  # noqa: E402
from assistant.ingest import collect  # noqa: E402
from assistant.llm import ScriptedLLM  # noqa: E402
from assistant.orchestrator import Orchestrator  # noqa: E402
from assistant.sandbox import LocalSubprocessSandbox  # noqa: E402
from assistant.store import InMemoryStore  # noqa: E402
from assistant.types import LLMReply, ToolCall  # noqa: E402

KNN_K3 = """from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
acc = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train).score(X_test, y_test)
print("k=3 test accuracy:", round(acc, 3))"""

chunks, problems = collect(ROOT / "sample_data", 180, 30)
store = InMemoryStore(HashingEmbedder())
store.add(chunks)
print(f"Ingested {len(chunks)} chunks; problems: {problems or 'none'}")

llm = ScriptedLLM([
    LLMReply("I'll run the lab example with k = 3.", [ToolCall("toolu_smoke1", "run_python", {"code": KNN_K3})]),
    LLMReply("Using the Lecture 6 lab setup [S1] and scikit-learn's KNeighborsClassifier [S2] with "
             "n_neighbors=3, the sandbox printed the k = 3 test accuracy shown above.", []),
])
orch = Orchestrator(store, llm, LocalSubprocessSandbox(timeout_s=10), sim_cutoff=0.1,
                    log_path=ROOT / "logs" / "smoke_requests.jsonl")
res = orch.answer("Run the Lecture 6 k-NN lab example with k = 3 and report the test accuracy.")
print_trace(res)
out = ROOT / "traces" / "smoke_offline_trace.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"NOTE": "offline smoke run: scripted LLM + unsafe local runner; not an evaluation result",
                           **res.to_dict()}, indent=2))
print(f"\nSaved {out.relative_to(ROOT)}")
