from pathlib import Path

import pytest

from assistant.embeddings import HashingEmbedder
from assistant.ingest import collect
from assistant.llm import ScriptedLLM
from assistant.orchestrator import Orchestrator
from assistant.store import InMemoryStore
from assistant.types import LLMReply, SandboxResult, ToolCall

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"


class FakeSandbox:
    """Returns queued results and records the code it was given."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.codes = []

    def run(self, code):
        self.codes.append(code)
        if self.results:
            return self.results.pop(0)
        return SandboxResult(True, 0, "ok\n", "", False, 5)


def text(t):
    return LLMReply(text=t, tool_calls=[])


def tool(code, cid="t1", t=""):
    return LLMReply(text=t, tool_calls=[ToolCall(cid, "run_python", {"code": code})])


@pytest.fixture(scope="session")
def store():
    s = InMemoryStore(HashingEmbedder())
    chunks, problems = collect(SAMPLE, size=180, overlap=30)
    assert not problems
    s.add(chunks)
    return s


@pytest.fixture
def make_orch(store, tmp_path):
    def _make(replies, sandbox=None, **kw):
        llm = ScriptedLLM(replies)
        sbx = sandbox or FakeSandbox()
        kw.setdefault("sim_cutoff", 0.05)
        orch = Orchestrator(store, llm, sbx, log_path=tmp_path / "log.jsonl", **kw)
        return orch, llm, sbx
    return _make
