"""BM25 search via SQLite FTS5."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from claude_recall.search.filters import Filters

_FTS_SAFE = re.compile(r"[^\w\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    session_id: str
    project: str
    started_at: str
    role_mix: str
    score: float
    text: str


def _to_match_query(q: str) -> str:
    tokens = [t for t in _FTS_SAFE.split(q) if t]
    if not tokens:
        return '""'
    return " AND ".join(f'"{t}"' for t in tokens)


def _apply_filters(filters: Filters | None) -> tuple[str, list[object]]:
    if filters is None:
        return "", []
    sql_parts: list[str] = []
    params: list[object] = []
    if filters.project:
        sql_parts.append("c.project = ?")
        params.append(filters.project)
    if filters.since_iso:
        sql_parts.append("c.started_at >= ?")
        params.append(filters.since_iso)
    if filters.role:
        sql_parts.append("c.role_mix LIKE ?")
        params.append(f"%{filters.role}%")
    if filters.tool:
        sql_parts.append("c.tool_names LIKE ?")
        params.append(f"%{filters.tool}%")
    if not sql_parts:
        return "", []
    return " AND " + " AND ".join(sql_parts), params


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    filters: Filters | None = None,
    project: str | None = None,
) -> list[SearchHit]:
    if project and (filters is None or filters.project is None):
        filters = Filters(
            project=project,
            since_iso=filters.since_iso if filters else None,
            role=filters.role if filters else None,
            tool=filters.tool if filters else None,
        )
    match = _to_match_query(query)
    where_extra, extra_params = _apply_filters(filters)
    sql = f"""
        SELECT c.id, c.session_id, c.project, c.started_at, c.role_mix, c.text,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?{where_extra}
        ORDER BY score LIMIT ?
    """
    params: list[object] = [match, *extra_params, limit]
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
