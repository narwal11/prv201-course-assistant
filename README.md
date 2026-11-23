# PRV201 Course Assistant (M3 prototype)

A retrieval-augmented assistant that answers PRV201 questions **only from the course
materials**, cites lecture and page, and can run short Python snippets in a locked-down
Docker sandbox (`run_python`). Architecture and rationale: see the M2 and M3 design documents.

```
Offline:  data/raw/*.pdf|*.ipynb|*.md ──parse──► chunk (≤180 words, 1 page each) ──MiniLM──► ChromaDB
Online:   Streamlit ─► FastAPI /ask ─► retrieve top-5 ─► (below cutoff? decline) ─► Claude Haiku 5.5
                                                         ▲          │ run_python (≤2 calls)
                                                         └── output ◄┘ Docker sandbox
          ─► citation check ─► answer + sources + trace ─► logs/requests.jsonl
```

## Repository map

| Path | What it is |
|---|---|
| `src/assistant/ingest.py` | Parse PDFs (PyMuPDF, pypdf fallback) and notebooks, chunk per page, embed, store. CLI. |
| `src/assistant/embeddings.py` | `MiniLMEmbedder` (production); `HashingEmbedder` (tests only). |
| `src/assistant/store.py` | `ChromaStore` (cosine, lecture filter); `InMemoryStore` (tests). |
| `src/assistant/orchestrator.py` | The bounded loop: retrieve → prompt → ≤2 tool calls → citation check → log. |
| `src/assistant/llm.py` | `AnthropicLLM` (retries + backoff), `OllamaLLM` fallback, `FallbackLLM`, `ScriptedLLM`. |
| `src/assistant/sandbox.py` | `DockerSandbox` (all limits) and the unsafe `LocalSubprocessSandbox` for tests. |
| `src/assistant/prompts.py` | System prompt and fixed messages. |
| `src/assistant/api.py` | FastAPI app: `POST /ask`, `GET /health`. |
| `src/assistant/ask.py` | CLI that prints/saves a full execution trace. |
| `ui/app.py` | Streamlit chat UI (answers, sources, code + output, optional debug view). |
| `sandbox/Dockerfile` | Image for `run_python` (numpy, pandas, scikit-learn, matplotlib). |
| `eval/run_eval.py`, `eval/score.py` | Test-set runner (automatic metrics + blind grading sheets) and scorer. |
| `eval/attacks/`, `eval/run_attacks.py` | 15 sandbox attack snippets and the containment checker. |
| `tests/` | 40 pytest tests (no API key or Docker needed). |
| `scripts/smoke_offline.py` | Offline end-to-end run (scripted LLM) for a quick plumbing check. |

## Setup (≈10 minutes)

Requires Python 3.10+ and Docker.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # then paste your ANTHROPIC_API_KEY
docker build -t prv201-sandbox:latest sandbox/
```

## Run it

```bash
# 1. put the course files in data/raw/ (name them like L06_knn.pdf so lecture numbers are detected)
python -m assistant.ingest --src data/raw --reset       # prints chunks per lecture + any bad files

# 2. one question from the terminal, with the full trace
python -m assistant.ask "How does k affect overfitting in k-NN?"
python -m assistant.ask "Run the Lecture 6 k-NN example with k = 3" --save traces/knn_k3.json

# 3. the app
uvicorn assistant.api:app --port 8000                  # terminal 1
streamlit run ui/app.py                                # terminal 2 -> http://localhost:8501
```

`curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"What is cross-validation?"}'`
returns the answer, citations, retrieved chunks with similarities, every tool call with its
output, flags, token counts and latency.

## Test and evaluate

```bash
pytest                                                  # unit + pipeline tests, ~3 s, offline
ALLOW_UNSAFE_LOCAL_SANDBOX=1 python scripts/smoke_offline.py

python eval/run_attacks.py                              # 15 attacks against the Docker sandbox
cp eval/testset.example.jsonl eval/testset.jsonl        # then write the 60 real questions
python eval/run_eval.py --testset eval/testset.jsonl    # auto metrics + grading_A.csv / grading_B.csv
#   two members grade their sheet separately (blind), then:
python eval/score.py --run eval/results/run_<timestamp> --attacks eval/results/attacks_<timestamp>.json
```

`run_eval.py` also prints the top-similarity distribution for answerable vs. out-of-scope
questions; use it to set `SIM_CUTOFF` in `.env`.

## Safety notes

* `SANDBOX_BACKEND=local` is **not** isolated and refuses to start without
  `ALLOW_UNSAFE_LOCAL_SANDBOX=1`. It exists only for tests. `run_attacks.py` refuses to use it.
* Never commit `.env`. Set a monthly spending cap on the Anthropic key.
* Homework policy: the system prompt gives hints and explanations, not full solutions to graded work.
