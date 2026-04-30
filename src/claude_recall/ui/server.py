"""Local FastAPI + HTMX UI for claudegrep.

Binds 127.0.0.1 by default. No auth — assumes single-user local machine.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from claude_recall.config import load_config
from claude_recall.embed.local import LocalEmbedder
from claude_recall.search import bm25, hybrid, vector
from claude_recall.search.filters import parse as parse_filters
from claude_recall.search.rerank import Reranker
from claude_recall.store import db
from claude_recall.store import session as session_store

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

_HERE = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))


class _State:
    """Lazy-loaded singletons shared across requests."""

    def __init__(self) -> None:
        self._cfg = load_config()
        self._embedder: LocalEmbedder | None = None
        self._reranker: Reranker | None = None

    @property
    def cfg(self) -> Any:
        return self._cfg

    def conn(self, *, with_vec: bool = True) -> sqlite3.Connection:
        if with_vec:
            return db.connect(self._cfg.db_path, vector_dim=self.embedder.dim)
        return db.connect(self._cfg.db_path)

    @property
    def embedder(self) -> LocalEmbedder:
        if self._embedder is None:
            self._embedder = LocalEmbedder(DEFAULT_MODEL)
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(DEFAULT_RERANKER)
        return self._reranker


def create_app() -> FastAPI:
    app = FastAPI(title="claudegrep", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
    state = _State()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        return _TEMPLATES.TemplateResponse(request, "index.html", {"request": request})

    @app.post("/search", response_class=HTMLResponse)
    def search(
        request: Request,
        q: str = Form(...),
        mode: str = Form("hybrid"),
        rerank: str = Form(""),
        limit: int = Form(10),
    ) -> Any:
        cleaned, filters = parse_filters(q)
        do_rerank = rerank == "on"
        fetch = limit * 3 if do_rerank else limit

        if mode == "bm25":
            conn = state.conn(with_vec=False)
            hits = bm25.search(conn, cleaned, limit=fetch, filters=filters)
        elif mode == "vector":
            conn = state.conn(with_vec=True)
            hits = vector.search(conn, state.embedder, cleaned, limit=fetch, filters=filters)
        else:
            conn = state.conn(with_vec=True)
            hits = hybrid.search(conn, state.embedder, cleaned, limit=fetch, filters=filters)

        if do_rerank and hits:
            hits = state.reranker.rerank(cleaned, hits, top_k=limit)
        else:
            hits = hits[:limit]

        return _TEMPLATES.TemplateResponse(
            request, "_hits.html", {"request": request, "hits": hits, "query": q}
        )

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    def show_session(request: Request, session_id: str) -> Any:
        conn = state.conn(with_vec=False)
        try:
            chunks = session_store.list_session_chunks(conn, session_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        if not chunks:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        ctx = {"request": request, "chunks": chunks, "session_id": session_id}
        return _TEMPLATES.TemplateResponse(request, "session.html", ctx)

    @app.get("/chunk/{chunk_id}/text", response_class=PlainTextResponse)
    def chunk_text(chunk_id: str) -> str:
        conn = state.conn(with_vec=False)
        try:
            c = session_store.get_chunk(conn, chunk_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        if c is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        return c.text

    return app


def serve(host: str = "127.0.0.1", port: int = 7777) -> None:
    """Run uvicorn with the local UI."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
