"""Central configuration. Every setting can be overridden with an environment
variable (or a .env file loaded by the entry points), so the same code runs on a
laptop, in CI, and in the demo without edits."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader (avoids a python-dotenv dependency). Existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if val[:1] in ("'", '"') and val[:1] in val[1:]:
            val = val[1:val.index(val[0], 1)]          # quoted: keep everything inside
        else:
            val = val.split(" #", 1)[0].strip()        # unquoted: drop inline comment
        os.environ.setdefault(key.strip(), val)


@dataclass
class Settings:
    # --- ingestion / retrieval -------------------------------------------
    raw_dir: Path = field(default_factory=lambda: Path(_env("RAW_DIR", "data/raw")))
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", "data/chroma")))
    collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "prv201"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    # MiniLM truncates input at 256 word pieces, so chunks are ~180 words (~235 word pieces).
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "180")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "30")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    # Cosine similarity below which we say "not covered" without calling the LLM.
    # Tune on the eval set (eval/run_eval.py prints the similarity distribution).
    sim_cutoff: float = field(default_factory=lambda: float(_env("SIM_CUTOFF", "0.30")))

    # --- LLM -----------------------------------------------------------------
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "anthropic"))  # anthropic | ollama
    model: str = field(default_factory=lambda: _env("ASSISTANT_MODEL", "claude-haiku-5-5"))
    fallback_backend: str = field(default_factory=lambda: _env("LLM_FALLBACK", "none"))  # none | ollama
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1:8b"))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "800")))
    max_tool_calls: int = field(default_factory=lambda: int(_env("MAX_TOOL_CALLS", "2")))
    api_retries: int = field(default_factory=lambda: int(_env("API_RETRIES", "3")))

    # --- sandbox -------------------------------------------------------------
    sandbox_backend: str = field(default_factory=lambda: _env("SANDBOX_BACKEND", "docker"))  # docker | local
    sandbox_image: str = field(default_factory=lambda: _env("SANDBOX_IMAGE", "prv201-sandbox:latest"))
    sandbox_timeout_s: int = field(default_factory=lambda: int(_env("SANDBOX_TIMEOUT_S", "10")))
    sandbox_memory_mb: int = field(default_factory=lambda: int(_env("SANDBOX_MEMORY_MB", "512")))
    sandbox_max_output: int = field(default_factory=lambda: int(_env("SANDBOX_MAX_OUTPUT", "4000")))

    # --- service -------------------------------------------------------------
    log_path: Path = field(default_factory=lambda: Path(_env("LOG_PATH", "logs/requests.jsonl")))
    cache_size: int = field(default_factory=lambda: int(_env("CACHE_SIZE", "256")))
    api_url: str = field(default_factory=lambda: _env("API_URL", "http://localhost:8000"))
