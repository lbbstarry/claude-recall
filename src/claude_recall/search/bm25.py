"""BM25 search via SQLite FTS5."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    session_id: str
    project: str
    started_at: str
    role_mix: str
    score: float
    text: str


_FTS_SAFE = re.compile(r"[^\w\u4e00-\u9fff]+")


def _to_match_query(q: str) -> str:
    """Convert free-text into a safe FTS5 MATCH expression.

    Splits on non-word characters, quotes each token, and ANDs them.
    """
    tokens = [t for t in _FTS_SAFE.split(q) if t]
    if not tokens:
        return '""'
    return " AND ".join(f'"{t}"' for t in tokens)


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
) -> list[SearchHit]:
    match = _to_match_query(query)
    sql = """
        SELECT c.id, c.session_id, c.project, c.started_at, c.role_mix, c.text,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
    """
    params: list[object] = [match]
    if project:
        sql += " AND c.project = ?"
        params.append(project)
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        SearchHit(
            chunk_id=r["id"],
            session_id=r["session_id"],
            project=r["project"],
            started_at=r["started_at"],
            role_mix=r["role_mix"],
            score=float(r["score"]),
            text=r["text"],
        )
        for r in rows
    ]
