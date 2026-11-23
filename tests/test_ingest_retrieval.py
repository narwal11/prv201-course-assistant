from pathlib import Path

import pytest

from assistant.ingest import chunk_file, collect, lecture_from_name, split_words
from conftest import SAMPLE


@pytest.mark.parametrize("name,lec", [
    ("L06_knn.pdf", 6), ("Lecture 12 - Trees.pdf", 12), ("lec03.ipynb", 3), ("week_4_slides.pdf", 4),
    ("syllabus.pdf", None), ("lab_results_2026.pdf", None), ("L101.pdf", None),
])
def test_lecture_from_name(name, lec):
    assert lecture_from_name(name) == lec


def test_split_words_overlap_and_coverage():
    words = [f"w{i}" for i in range(500)]
    chunks = split_words(" ".join(words), size=180, overlap=30)
    assert all(len(c.split()) <= 180 for c in chunks)
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]          # overlap
    assert set(" ".join(chunks).split()) == set(words)                # nothing dropped
    with pytest.raises(ValueError):
        split_words("a b c", size=10, overlap=10)


def test_pdf_chunks_keep_page_numbers():
    chunks = chunk_file(SAMPLE / "L06_knn_sample.pdf", 180, 30)
    assert [c.page for c in chunks] == [1, 2, 3]
    assert all(c.lecture == 6 and c.kind == "slides" for c in chunks)
    assert "overfitting" in chunks[1].text
    assert chunks[1].label() == "Lecture 6, p. 2"


def test_notebook_cells_are_fenced_and_numbered():
    chunks = chunk_file(SAMPLE / "L06_knn_lab_sample.ipynb", 180, 30)
    assert [c.page for c in chunks] == [1, 2, 3]
    assert chunks[2].text.startswith("```python") and "KNeighborsClassifier" in chunks[2].text
    assert chunks[2].label() == "Lecture 6, cell 3"


def test_collect_ids_unique_and_readme_skipped():
    chunks, problems = collect(SAMPLE, 180, 30)
    assert not problems
    assert len({c.id for c in chunks}) == len(chunks)
    assert not any(c.source.lower().startswith("readme") for c in chunks)


def test_collect_reports_bad_files(tmp_path: Path):
    (tmp_path / "L01_empty.pdf").write_bytes(b"not a pdf")
    (tmp_path / "L02_notes.md").write_text("Gradient descent updates weights against the gradient of the loss.")
    chunks, problems = collect(tmp_path, 180, 30)
    assert len(chunks) == 1 and len(problems) == 1 and "L01_empty.pdf" in problems[0]


def test_retrieval_finds_right_page(store):
    top = store.query("how does choosing k cause overfitting", k=5)
    assert top[0].chunk.source == "L06_knn_sample.pdf" and top[0].chunk.page == 2
    assert top[0].similarity >= top[-1].similarity


def test_lecture_filter(store):
    hits = store.query("test set cross-validation", k=5, lecture=6)
    assert hits and all(h.chunk.lecture == 6 for h in hits)
