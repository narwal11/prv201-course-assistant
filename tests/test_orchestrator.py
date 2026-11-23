import json

from assistant import prompts
from assistant.llm import LLMError
from assistant.orchestrator import looks_like_code
from assistant.types import SandboxResult
from conftest import FakeSandbox, text, tool

Q = "How does the choice of k affect overfitting in k-NN?"
FAIL = SandboxResult(False, 1, "", "NameError: name 'x' is not defined", False, 5)


def test_plain_answer_with_citation(make_orch):
    orch, llm, sbx = make_orch([text("Small k overfits; large k underfits [S1].")])
    res = orch.answer(Q)
    assert not res.declined and res.flags == [] and res.llm_calls == 1 and sbx.codes == []
    assert res.citations[0]["label"] == res.retrieved[0]["label"]
    assert "[S1]" not in res.answer and f"[{res.retrieved[0]['label']}]" in res.answer
    user_msg = llm.calls[0]["history"][0]["text"]
    assert "[S1] (" in user_msg and Q in user_msg


def test_below_cutoff_declines_without_llm(make_orch):
    orch, llm, _ = make_orch([], sim_cutoff=0.99)
    res = orch.answer("Who won the 2026 World Cup?")
    assert res.declined and "below_cutoff" in res.flags and llm.calls == []


def test_code_question_below_cutoff_still_reaches_llm(make_orch):
    orch, llm, _ = make_orch([text("Not covered: not in the materials.")], sim_cutoff=0.99)
    res = orch.answer("What does this code print?\n```python\nprint(sum(range(10)))\n```")
    assert len(llm.calls) == 1 and "code_question_below_cutoff" in res.flags and res.declined


def test_model_not_covered_is_declined(make_orch):
    orch, _, _ = make_orch([text("Not covered: the PRV201 materials don't discuss this.")])
    res = orch.answer(Q)
    assert res.declined and "uncited" not in res.flags


def test_tool_call_round_trip(make_orch):
    sbx = FakeSandbox([SandboxResult(True, 0, "1 0.956\n5 0.978\n", "", False, 900)])
    orch, llm, _ = make_orch([tool("print(1)"), text("k=5 scored 0.978 [S2].")], sandbox=sbx)
    res = orch.answer("Run the Lecture 6 k-NN example for k = 1 and 5")
    assert sbx.codes == ["print(1)"] and len(res.tool_calls) == 1 and res.llm_calls == 2
    hist = llm.calls[1]["history"]
    assert hist[1]["role"] == "assistant" and hist[2]["role"] == "tool" and "0.978" in hist[2]["content"]
    assert not hist[2]["is_error"]


def test_tool_cap_enforced(make_orch):
    orch, llm, sbx = make_orch([tool("a=1", "t1"), tool("a=2", "t2"), tool("a=3", "t3", t="Answer [S1].")])
    res = orch.answer(Q)
    assert len(sbx.codes) == 2                         # third call never ran
    assert [c["allow_tools"] for c in llm.calls] == [True, True, False]
    assert llm.calls[2]["history"][-1]["text"].startswith(prompts.TOOL_LIMIT_NOTE)
    assert res.llm_calls == 3


def test_parallel_calls_beyond_cap_get_error_results(make_orch):
    from assistant.types import LLMReply, ToolCall
    three = LLMReply("", [ToolCall(f"t{i}", "run_python", {"code": f"print({i})"}) for i in range(3)])
    orch, llm, sbx = make_orch([three, text("Done [S1].")])
    orch.answer(Q)
    tool_msgs = [h for h in llm.calls[1]["history"] if h["role"] == "tool"]
    assert len(sbx.codes) == 2 and len(tool_msgs) == 3 and tool_msgs[2]["is_error"]


def test_two_failures_answer_without_code(make_orch):
    sbx = FakeSandbox([FAIL, FAIL])
    orch, llm, _ = make_orch([tool("x"), tool("x+1"), text("I couldn't run the code, but [S1].")], sandbox=sbx)
    res = orch.answer(Q)
    assert "code_failed" in res.flags and all(h["is_error"] for h in llm.calls[1]["history"] if h["role"] == "tool")
    assert prompts.CODE_FAILED_NOTE in llm.calls[2]["history"][-1]["text"]


def test_malformed_tool_call(make_orch):
    from assistant.types import LLMReply, ToolCall
    bad = LLMReply("", [ToolCall("t1", "run_python", {"source": "print(1)"})])
    orch, llm, sbx = make_orch([bad, text("Answer [S1].")])
    res = orch.answer(Q)
    assert sbx.codes == [] and "malformed_tool_call" in res.flags


def test_citation_checks(make_orch):
    orch, _, _ = make_orch([text("Made-up source [S9].")])
    assert {"invalid_citation", "uncited"} <= set(orch.answer(Q).flags)
    orch, _, _ = make_orch([text("No citations at all.")])
    assert orch.answer(Q).flags == ["uncited"]


def test_llm_error_is_graceful_and_not_cached(make_orch):
    class Down:
        name = "down"
        def complete(self, *a, **k):
            raise LLMError("503")
    orch, _, _ = make_orch([])
    orch.llm = Down()
    res = orch.answer(Q)
    assert "llm_error" in res.flags and res.answer == prompts.LLM_UNAVAILABLE
    assert orch._cache == {}


def test_cache_hit_skips_llm(make_orch):
    orch, llm, _ = make_orch([text("Answer [S1].")])
    first = orch.answer(Q)
    second = orch.answer("  how does the CHOICE of k affect overfitting in k-NN? ")
    assert len(llm.calls) == 1 and second.cached and second.answer == first.answer


def test_request_log_written(make_orch, tmp_path):
    orch, _, _ = make_orch([text("Answer [S1].")])
    orch.answer(Q)
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["question"] == Q and rec["llm_calls"] == 1 and len(rec["retrieved"]) == 5


def test_looks_like_code():
    assert looks_like_code("```python\nprint(1)\n```")
    assert looks_like_code("Run the Lecture 6 k-NN example with k = 3")
    assert looks_like_code("what does np.mean([1,2]) return")
    assert not looks_like_code("What is the bias-variance trade-off?")
