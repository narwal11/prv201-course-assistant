"""LLM back ends behind one interface.

The orchestrator keeps a provider-neutral transcript:
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [ToolCall, ...]}
    {"role": "tool", "call_id": ..., "content": ..., "is_error": bool}
Each adapter converts that transcript to its provider's wire format.
"""
from __future__ import annotations

import json
import logging
import random
import time
import uuid
from typing import Any, Protocol

import requests

from .types import LLMReply, ToolCall

log = logging.getLogger(__name__)

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute a short, self-contained Python 3 snippet in an isolated sandbox and return "
        "stdout/stderr. Use ONLY when the question needs code to be run (e.g. computing a result, "
        "running a course example, checking what a snippet prints). numpy, pandas, scikit-learn and "
        "matplotlib (Agg backend) are installed. No network, no files persist, 10 second limit. "
        "Print every value you need to see."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Complete Python source to execute."}},
        "required": ["code"],
    },
}


class LLMError(RuntimeError):
    """Raised when a back end cannot produce a reply after its retries."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, history: list[dict], tools: list[dict], allow_tools: bool) -> LLMReply: ...


def _backoff(attempt: int) -> None:
    time.sleep(min(8.0, 0.5 * 2 ** attempt) + random.random() * 0.25)


# --------------------------------------------------------------------------- Anthropic
class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 800, retries: int = 3):
        import anthropic

        self._anthropic = anthropic
        # The SDK already retries some errors; we turn that off and do our own
        # exponential backoff so retries are visible in our logs.
        self._client = anthropic.Anthropic(max_retries=0, timeout=30.0)
        self.model, self.max_tokens, self.retries = model, max_tokens, retries
        self.name = f"anthropic:{model}"

    @staticmethod
    def to_messages(history: list[dict]) -> list[dict]:
        msgs: list[dict] = []

        def user_blocks() -> list:
            # tool results and follow-up notes must share ONE user message
            if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                return msgs[-1]["content"]
            msgs.append({"role": "user", "content": []})
            return msgs[-1]["content"]

        for h in history:
            if h["role"] == "user":
                user_blocks().append({"type": "text", "text": h["text"]})
            elif h["role"] == "tool":
                user_blocks().append({"type": "tool_result", "tool_use_id": h["call_id"],
                                      "content": h["content"], "is_error": h.get("is_error", False)})
            elif h["role"] == "assistant":
                blocks: list[dict[str, Any]] = []
                if h.get("text"):
                    blocks.append({"type": "text", "text": h["text"]})
                for tc in h.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
                msgs.append({"role": "assistant", "content": blocks})
        return msgs

    def complete(self, system: str, history: list[dict], tools: list[dict], allow_tools: bool) -> LLMReply:
        a = self._anthropic
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=self.to_messages(history))
        if tools:
            # Tools stay declared (earlier tool_use blocks reference them); "none" blocks new calls.
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto"} if allow_tools else {"type": "none"}
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.messages.create(**kwargs)
                break
            except (a.APIConnectionError, a.APIStatusError) as e:  # APITimeoutError is a connection error
                status = getattr(e, "status_code", None)
                if status is not None and status != 429 and status < 500:
                    # 400/401/403/404: our bug or a bad key; retrying won't help
                    raise LLMError(f"Anthropic rejected the request ({status}): {e}") from e
                last = e
                log.warning("Anthropic transient error %s (attempt %d): %s", status, attempt + 1, e)
                if attempt < self.retries:
                    _backoff(attempt)
        else:
            raise LLMError(f"Anthropic unavailable after {self.retries + 1} attempts: {last}")

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMReply(text=text, tool_calls=calls, stop_reason=resp.stop_reason or "",
                        input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                        backend=self.name)


# --------------------------------------------------------------------------- Ollama
class OllamaLLM:
    """Free local fallback (Llama 3.1 8B). Weaker, but keeps the demo alive in an outage."""

    def __init__(self, url: str, model: str, retries: int = 2):
        self.url, self.model, self.retries = url.rstrip("/"), model, retries
        self.name = f"ollama:{model}"

    @staticmethod
    def to_messages(system: str, history: list[dict]) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        for h in history:
            if h["role"] == "user":
                msgs.append({"role": "user", "content": h["text"]})
            elif h["role"] == "tool":
                msgs.append({"role": "tool", "content": h["content"]})
            else:
                m: dict[str, Any] = {"role": "assistant", "content": h.get("text", "")}
                if h.get("tool_calls"):
                    m["tool_calls"] = [{"function": {"name": t.name, "arguments": t.args}} for t in h["tool_calls"]]
                msgs.append(m)
        return msgs

    def complete(self, system: str, history: list[dict], tools: list[dict], allow_tools: bool) -> LLMReply:
        body: dict[str, Any] = {"model": self.model, "stream": False, "messages": self.to_messages(system, history)}
        if tools and allow_tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["input_schema"]}} for t in tools]
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.url}/api/chat", json=body, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                last = e
                if attempt < self.retries:
                    _backoff(attempt)
        else:
            raise LLMError(f"Ollama unavailable: {last}")
        msg = data.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            calls.append(ToolCall(f"ollama_{uuid.uuid4().hex[:8]}", fn.get("name", ""), args))
        return LLMReply(text=msg.get("content", ""), tool_calls=calls, stop_reason=data.get("done_reason", ""),
                        input_tokens=data.get("prompt_eval_count", 0), output_tokens=data.get("eval_count", 0),
                        backend=self.name)


# --------------------------------------------------------------------------- fallback + test double
class FallbackLLM:
    """Try the primary back end; on LLMError switch to the fallback for that request."""

    def __init__(self, primary: LLM, fallback: LLM):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}->{fallback.name}"

    def complete(self, system, history, tools, allow_tools):
        try:
            return self.primary.complete(system, history, tools, allow_tools)
        except LLMError as e:
            log.error("Primary LLM failed, using fallback: %s", e)
            return self.fallback.complete(system, history, tools, allow_tools)


class ScriptedLLM:
    """Returns pre-written replies in order. Used by tests and the offline smoke run
    to exercise the orchestrator loop without an API key."""

    def __init__(self, replies: list[LLMReply]):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self.name = "scripted"

    def complete(self, system, history, tools, allow_tools):
        self.calls.append({"history": [dict(h) for h in history], "allow_tools": allow_tools})
        if not self._replies:
            raise LLMError("ScriptedLLM ran out of replies")
        r = self._replies.pop(0)
        if r.tool_calls and not allow_tools:  # mimic tool_choice=none
            return LLMReply(text=r.text or "(answered without tools)", tool_calls=[], backend=self.name)
        r.backend = self.name
        return r


def build_llm(settings) -> LLM:
    if settings.llm_backend == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    primary = AnthropicLLM(settings.model, settings.max_tokens, settings.api_retries)
    if settings.fallback_backend == "ollama":
        return FallbackLLM(primary, OllamaLLM(settings.ollama_url, settings.ollama_model))
    return primary
