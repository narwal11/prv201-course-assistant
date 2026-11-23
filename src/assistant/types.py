"""Plain data types shared by every component. Keeping these provider-neutral is
what lets us swap Claude for Ollama, or Docker for a test runner, without
touching the orchestrator."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    source: str            # file name, e.g. "L06_knn.pdf"
    page: int              # 1-based page (PDF) or cell number (notebook)
    lecture: int | None    # parsed from the file name when possible
    kind: str              # slides | notebook | reading | other

    def label(self) -> str:
        lec = f"Lecture {self.lecture}" if self.lecture is not None else self.source
        unit = "cell" if self.kind == "notebook" else "p."
        return f"{lec}, {unit} {self.page}"

    def meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        d.pop("id")
        d["lecture"] = -1 if self.lecture is None else self.lecture  # Chroma metadata can't be None
        return d


@dataclass
class Retrieved:
    chunk: Chunk
    similarity: float


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = ""


@dataclass
class SandboxResult:
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int

    def for_model(self, max_chars: int = 4000) -> str:
        if self.timed_out:
            body = "ERROR: execution timed out and the process was killed."
        else:
            body = f"exit_code: {self.exit_code}\nstdout:\n{self.stdout}\nstderr:\n{self.stderr}"
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n...[truncated {len(body) - max_chars} chars]"
        return body


@dataclass
class ToolTrace:
    code: str
    result: SandboxResult


@dataclass
class AnswerResult:
    question: str
    answer: str
    declined: bool
    raw_answer: str = ""   # model text with [S#] tags; `answer` has them replaced by labels
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolTrace] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    latency_ms: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
