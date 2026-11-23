"""The online loop: retrieve -> prompt -> (run_python, at most N times) -> answer.

A fixed, bounded loop instead of an open-ended agent: every request makes at
most MAX_TOOL_CALLS sandbox runs and MAX_TOOL_CALLS + 1 LLM calls, so latency
and cost have a hard ceiling and every trace is easy to read.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from . import prompts
from .llm import LLM, RUN_PYTHON_TOOL, LLMError
from .sandbox import Sandbox
from .store import VectorStore
from .types import AnswerResult, Retrieved, ToolTrace

log = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[S(\d+)\]")
_CODE_RE = re.compile(
    r"```|^\s*(def |import |from \w+ import|for .+ in .+:|print\()|\b(np|pd|plt)\.\w+|"
    r"\b(run|execute)\b.*\b(code|snippet|example)\b|\bwhat does (this|the following) (code|snippet)",
    re.I | re.M)


def looks_like_code(question: str) -> bool:
    return bool(_CODE_RE.search(question))


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


class Orchestrator:
    def __init__(self, store: VectorStore, llm: LLM, sandbox: Sandbox, *, top_k: int = 5,
                 sim_cutoff: float = 0.30, max_tool_calls: int = 2, log_path: Path | None = None,
                 cache_size: int = 256, max_output_chars: int = 4000):
        self.store, self.llm, self.sandbox = store, llm, sandbox
        self.top_k, self.sim_cutoff, self.max_tool_calls = top_k, sim_cutoff, max_tool_calls
        self.log_path, self.cache_size, self.max_output_chars = log_path, cache_size, max_output_chars
        self._cache: OrderedDict[tuple, AnswerResult] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def answer(self, question: str, lecture: int | None = None, use_cache: bool = True) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        t0 = time.monotonic()
        key = (_norm(question), lecture)
        if use_cache:
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    hit = copy.deepcopy(hit)
            if hit is not None:  # log outside the lock (_log takes it too)
                hit.cached, hit.latency_ms = True, int((time.monotonic() - t0) * 1000)
                self._log(hit, lecture)
                return hit

        res = self._answer_uncached(question, lecture)
        res.latency_ms = int((time.monotonic() - t0) * 1000)
        if use_cache and "llm_error" not in res.flags:
            with self._lock:
                self._cache[key] = copy.deepcopy(res)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        self._log(res, lecture)
        return res

    # ------------------------------------------------------------------ core
    def _answer_uncached(self, question: str, lecture: int | None) -> AnswerResult:
        retrieved = self.store.query(question, self.top_k, lecture)
        res = AnswerResult(question=question, answer="", declined=False,
                           retrieved=[self._ret_dict(i, r) for i, r in enumerate(retrieved, 1)])
        top = max((r.similarity for r in retrieved), default=0.0)
        code_q = looks_like_code(question)

        if top < self.sim_cutoff and not code_q:
            res.answer = res.raw_answer = prompts.DECLINE_BELOW_CUTOFF
            res.declined = True
            res.flags.append("below_cutoff")
            return res
        if top < self.sim_cutoff:
            res.flags.append("code_question_below_cutoff")

        excerpts = [(f"S{i}", r.chunk.label(), r.chunk.text) for i, r in enumerate(retrieved, 1)]
        history: list[dict] = [{"role": "user", "text": prompts.build_user_message(question, excerpts)}]
        tools = [RUN_PYTHON_TOOL]
        used = failed = 0
        final: str | None = None

        try:
            for _ in range(self.max_tool_calls + 1):
                reply = self.llm.complete(prompts.SYSTEM_PROMPT, history, tools,
                                          allow_tools=used < self.max_tool_calls)
                res.llm_calls += 1
                res.input_tokens += reply.input_tokens
                res.output_tokens += reply.output_tokens
                res.backend = reply.backend
                if not reply.tool_calls:
                    final = reply.text
                    break
                history.append({"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls})
                for tc in reply.tool_calls:  # every tool_use needs a matching tool_result
                    code = tc.args.get("code") if isinstance(tc.args, dict) else None
                    if tc.name != "run_python" or not isinstance(code, str) or not code.strip():
                        history.append({"role": "tool", "call_id": tc.id, "is_error": True,
                                        "content": "ERROR: unknown tool or missing 'code' argument."})
                        res.flags.append("malformed_tool_call")
                        continue
                    if used >= self.max_tool_calls:
                        history.append({"role": "tool", "call_id": tc.id, "is_error": True,
                                        "content": "ERROR: tool-call limit reached; code was not run."})
                        continue
                    result = self.sandbox.run(code)
                    used += 1
                    failed += 0 if result.ok else 1
                    res.tool_calls.append(ToolTrace(code=code, result=result))
                    history.append({"role": "tool", "call_id": tc.id, "is_error": not result.ok,
                                    "content": result.for_model(self.max_output_chars)})
                if used >= self.max_tool_calls:
                    note = prompts.TOOL_LIMIT_NOTE
                    if failed == used:
                        note += " " + prompts.CODE_FAILED_NOTE
                        res.flags.append("code_failed")
                    history.append({"role": "user", "text": note})
        except LLMError as e:
            log.error("LLM failure: %s", e)
            res.answer = res.raw_answer = prompts.LLM_UNAVAILABLE
            res.flags.append("llm_error")
            return res

        if final is None:  # only reachable if a back end ignores tool_choice=none
            final = "Sorry, I couldn't finish answering that. Please try rephrasing."
            res.flags.append("loop_exhausted")

        res.raw_answer = final.strip()
        res.declined = res.raw_answer.lower().startswith("not covered")
        self._attach_citations(res, retrieved)
        return res

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ret_dict(i: int, r: Retrieved) -> dict:
        return {"tag": f"S{i}", "id": r.chunk.id, "label": r.chunk.label(), "source": r.chunk.source,
                "page": r.chunk.page, "lecture": r.chunk.lecture, "similarity": round(r.similarity, 4),
                "text": r.chunk.text}

    def _attach_citations(self, res: AnswerResult, retrieved: list[Retrieved]) -> None:
        nums = [int(n) for n in _CITE_RE.findall(res.raw_answer)]
        valid = sorted({n for n in nums if 1 <= n <= len(retrieved)})
        if any(n not in valid for n in nums):
            res.flags.append("invalid_citation")
        if not res.declined and not valid:
            res.flags.append("uncited")
        res.citations = [res.retrieved[n - 1] | {"text": None} for n in valid]

        def sub(m: re.Match) -> str:
            n = int(m.group(1))
            return f"[{retrieved[n - 1].chunk.label()}]" if 1 <= n <= len(retrieved) else "[?]"

        res.answer = _CITE_RE.sub(sub, res.raw_answer)

    def _log(self, res: AnswerResult, lecture: int | None) -> None:
        if not self.log_path:
            return
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": res.question, "lecture_filter": lecture, "declined": res.declined,
            "flags": res.flags, "latency_ms": res.latency_ms, "cached": res.cached,
            "llm_calls": res.llm_calls, "backend": res.backend,
            "tokens": {"in": res.input_tokens, "out": res.output_tokens},
            "retrieved": [(r["id"], r["similarity"]) for r in res.retrieved],
            "citations": [c["id"] for c in res.citations],
            "tool_calls": [{"ok": t.result.ok, "exit": t.result.exit_code, "timed_out": t.result.timed_out,
                            "ms": t.result.duration_ms} for t in res.tool_calls],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def build_orchestrator(settings, store=None, llm=None, sandbox=None) -> Orchestrator:
    """Wire the real components from Settings (tests pass doubles instead)."""
    if store is None:
        from .embeddings import get_embedder
        from .store import ChromaStore

        store = ChromaStore(settings.chroma_dir, settings.collection, get_embedder(settings.embed_model))
    if llm is None:
        from .llm import build_llm

        llm = build_llm(settings)
    if sandbox is None:
        from .sandbox import build_sandbox

        sandbox = build_sandbox(settings)
    return Orchestrator(store, llm, sandbox, top_k=settings.top_k, sim_cutoff=settings.sim_cutoff,
                        max_tool_calls=settings.max_tool_calls, log_path=settings.log_path,
                        cache_size=settings.cache_size, max_output_chars=settings.sandbox_max_output)
