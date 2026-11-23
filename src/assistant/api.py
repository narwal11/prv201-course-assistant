"""FastAPI backend.   Run:  uvicorn assistant.api:app --port 8000"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, load_dotenv
from .orchestrator import Orchestrator, build_orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="PRV201 Course Assistant", version="0.3.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    lecture: int | None = Field(None, ge=0, le=99, description="optional: restrict retrieval to one lecture")


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    load_dotenv()
    return build_orchestrator(Settings())


@app.get("/health")
def health() -> dict:
    orch = get_orchestrator()
    return {"status": "ok", "chunks": orch.store.count(), "llm": orch.llm.name}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    # sync def: FastAPI runs it in a worker thread, so blocking API/Docker calls are fine
    try:
        return get_orchestrator().answer(req.question, req.lecture).to_dict()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
