import os
import sys
from pathlib import Path

import pytest

from assistant.config import Settings, load_dotenv
from assistant.llm import AnthropicLLM, FallbackLLM, LLMError, OllamaLLM, ScriptedLLM
from assistant.sandbox import DockerSandbox, LocalSubprocessSandbox
from assistant.types import LLMReply, SandboxResult, ToolCall

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))


# ------------------------------------------------------------------ LLM adapters
HISTORY = [
    {"role": "user", "text": "Q"},
    {"role": "assistant", "text": "Let me run it.", "tool_calls": [ToolCall("tu_1", "run_python", {"code": "print(2)"})]},
    {"role": "tool", "call_id": "tu_1", "content": "exit_code: 0\nstdout:\n2", "is_error": False},
    {"role": "user", "text": "Tool-call limit reached."},
]


def test_anthropic_message_conversion():
    msgs = AnthropicLLM.to_messages(HISTORY)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][1] == {"type": "tool_use", "id": "tu_1", "name": "run_python", "input": {"code": "print(2)"}}
    # tool_result must come first and share one user message with the follow-up note
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "text"]
    assert msgs[2]["content"][0]["tool_use_id"] == "tu_1"


def test_ollama_message_conversion():
    msgs = OllamaLLM.to_messages("SYS", HISTORY)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"code": "print(2)"}


def test_fallback_switches_on_error():
    class Down:
        name = "down"
        def complete(self, *a, **k):
            raise LLMError("outage")
    fb = FallbackLLM(Down(), ScriptedLLM([LLMReply("from fallback", [])]))
    assert fb.complete("s", [], [], True).text == "from fallback"


# ------------------------------------------------------------------ sandbox
def test_docker_command_has_every_limit():
    cmd = " ".join(DockerSandbox("img", 10, 512).command("n"))
    for flag in ["--rm", "--network none", "--read-only", "--memory 512m", "--memory-swap 512m",
                 "--pids-limit 64", "--cap-drop ALL", "no-new-privileges", "--user 65534:65534",
                 "noexec", "timeout -s KILL 10", "python -I -"]:
        assert flag in cmd, flag


def test_local_sandbox_refuses_without_opt_in(monkeypatch):
    monkeypatch.delenv("ALLOW_UNSAFE_LOCAL_SANDBOX", raising=False)
    with pytest.raises(RuntimeError):
        LocalSubprocessSandbox()


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("ALLOW_UNSAFE_LOCAL_SANDBOX", "1")
    return LocalSubprocessSandbox(timeout_s=2, memory_mb=512, max_output=200)


def test_local_sandbox_runs_and_reports_errors(local):
    ok = local.run("import numpy as np\nprint(np.arange(4).sum())")
    assert ok.ok and ok.stdout.strip() == "6"
    bad = local.run("1/0")
    assert not bad.ok and "ZeroDivisionError" in bad.stderr


def test_local_sandbox_timeout_and_truncation(local):
    assert local.run("while True: pass").timed_out
    big = local.run("print('x' * 5000)")
    assert "truncated" in big.stdout and len(big.stdout) < 300


def test_sandbox_result_for_model():
    r = SandboxResult(False, None, "", "", True, 10_000)
    assert "timed out" in r.for_model()
    assert "truncated" in SandboxResult(True, 0, "y" * 50, "", False, 1).for_model(max_chars=20)


# ------------------------------------------------------------------ config + eval helpers
def test_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    for k in ("LLM_BACKEND", "TOP_K", "OLLAMA_MODEL"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text('LLM_BACKEND=ollama   # anthropic | ollama\nTOP_K=7\nOLLAMA_MODEL="llama 3 # x"\n')
    load_dotenv(tmp_path / ".env")
    s = Settings()
    assert (s.llm_backend, s.top_k, s.ollama_model) == ("ollama", 7, "llama 3 # x")


def test_eval_helpers(tmp_path):
    from run_eval import load_testset, page_key
    from score import kappa
    assert page_key("L06_knn.pdf#p12#c1") == "L06_knn.pdf#p12"
    assert len(load_testset(ROOT / "eval" / "testset.example.jsonl")) == 4
    bad = tmp_path / "t.jsonl"
    bad.write_text('{"id": "A1", "category": "answerable", "question": "q"}\n')
    with pytest.raises(ValueError):
        load_testset(bad)
    assert kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert abs(kappa([True, True, False, False], [True, False, True, False])) < 1e-9


def test_api_contract():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from assistant import api
    from assistant.orchestrator import Orchestrator
    from assistant.store import InMemoryStore
    from assistant.embeddings import HashingEmbedder
    from conftest import FakeSandbox
    api.get_orchestrator.cache_clear()
    orch = Orchestrator(InMemoryStore(HashingEmbedder()), ScriptedLLM([]), FakeSandbox(), sim_cutoff=0.3)
    api.get_orchestrator = lambda: orch
    c = TestClient(api.app)
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "anything"}).json()["declined"] is True
