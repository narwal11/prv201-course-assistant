"""Offline pipeline: parse course files -> chunk -> embed -> store.

Run:  python -m assistant.ingest --src data/raw --reset
Writes the vectors to ChromaDB and a human-readable copy of every chunk to
data/chunks.jsonl so we can inspect what each lecture actually contributes.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

from .config import Settings, load_dotenv
from .types import Chunk

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".ipynb", ".md", ".txt"}
_LECTURE_RE = re.compile(r"(?:^|[^a-z])(?:l|lec|lecture|week|wk)[ _-]?0*(\d{1,2})(?!\d)", re.I)


# --------------------------------------------------------------------------- parsing
def lecture_from_name(name: str) -> int | None:
    m = _LECTURE_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


def kind_from_name(path: Path) -> str:
    n = path.name.lower()
    if path.suffix.lower() == ".ipynb":
        return "notebook"
    if "reading" in n or "paper" in n:
        return "reading"
    if path.suffix.lower() == ".pdf" and ("slide" in n or lecture_from_name(n) is not None):
        return "slides"
    return "other"


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def pdf_pages(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (1-based page number, text). PyMuPDF keeps layout order best; pypdf is
    a pure-Python fallback so ingestion still works where PyMuPDF won't install."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                yield i, page.get_text()
        return
    except ImportError:
        log.warning("PyMuPDF not installed; falling back to pypdf for %s", path.name)
    from pypdf import PdfReader

    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        yield i, page.extract_text() or ""


def notebook_cells(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (1-based cell number, text). Code cells are fenced so the LLM can tell
    code from prose; outputs are skipped (they are often huge or binary)."""
    try:
        import nbformat

        nb = nbformat.read(str(path), as_version=4)
        cells = nb.cells
    except ImportError:
        cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    for i, cell in enumerate(cells, start=1):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if cell.get("cell_type") == "code" and src.strip():
            src = f"```python\n{src}\n```"
        yield i, src


def file_units(path: Path) -> Iterator[tuple[int, str]]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        yield from pdf_pages(path)
    elif suf == ".ipynb":
        yield from notebook_cells(path)
    else:
        yield 1, path.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- chunking
def split_words(text: str, size: int, overlap: int) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_file(path: Path, size: int, overlap: int, min_words: int = 5) -> list[Chunk]:
    """Chunks never cross a page boundary, so every chunk cites exactly one page."""
    lecture, kind = lecture_from_name(path.name), kind_from_name(path)
    chunks: list[Chunk] = []
    for page, raw in file_units(path):
        text = raw if kind == "notebook" else clean(raw)
        if len(text.split()) < min_words:
            continue
        for j, piece in enumerate(split_words(text, size, overlap)):
            chunks.append(Chunk(id=f"{path.name}#p{page}#c{j}", text=piece, source=path.name,
                                page=page, lecture=lecture, kind=kind))
    return chunks


def collect(src: Path, size: int, overlap: int) -> tuple[list[Chunk], list[str]]:
    chunks, problems = [], []
    files = sorted(p for p in src.rglob("*")
                   if p.suffix.lower() in SUPPORTED and not p.name.lower().startswith("readme"))
    if not files:
        raise FileNotFoundError(f"No supported files ({', '.join(sorted(SUPPORTED))}) under {src}")
    for f in files:
        try:
            got = chunk_file(f, size, overlap)
        except Exception as e:  # one bad file must not kill the whole ingest
            problems.append(f"{f.name}: {type(e).__name__}: {e}")
            continue
        if not got:
            problems.append(f"{f.name}: no extractable text (scanned PDF?)")
        chunks.extend(got)
    return chunks, problems


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    s = Settings()
    ap = argparse.ArgumentParser(description="Ingest PRV201 materials into ChromaDB")
    ap.add_argument("--src", type=Path, default=s.raw_dir)
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    ap.add_argument("--dump", type=Path, default=Path("data/chunks.jsonl"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from .embeddings import get_embedder
    from .store import ChromaStore

    chunks, problems = collect(args.src, s.chunk_words, s.chunk_overlap)
    store = ChromaStore(s.chroma_dir, s.collection, get_embedder(s.embed_model))
    if args.reset:
        store.reset()
    store.add(chunks)

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    with args.dump.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({"id": c.id, "label": c.label(), "words": len(c.text.split()), "text": c.text}) + "\n")

    per = Counter(c.lecture if c.lecture is not None else "unlabeled" for c in chunks)
    print(f"Ingested {len(chunks)} chunks from {len({c.source for c in chunks})} files into '{s.collection}'.")
    for lec, n in sorted(per.items(), key=lambda kv: str(kv[0])):
        print(f"  lecture {lec}: {n} chunks")
    for p in problems:
        print(f"  WARNING {p}")


if __name__ == "__main__":
    main()
