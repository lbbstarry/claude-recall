"""Vector KNN search via sqlite-vec."""

from __future__ import annotations

import sqlite3

from claude_recall.embed.base import Embedder
from claude_recall.search.bm25 import SearchHit
from claude_recall.search.filters import Filters


def _matches(row: sqlite3.Row, f: Filters | None) -> bool:
    if f is None:
        return True
    if f.project and row["project"] != f.project:
        return False
    if f.since_iso and row["started_at"] < f.since_iso:
        return False
    if f.role and f.role not in row["role_mix"]:
        return False
    return not (f.tool and f.tool not in (row["tool_names"] or ""))


def search(
    conn: sqlite3.Connection,
    embedder: Embedder,
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
    qvec = embedder.embed([query])[0].tobytes()
    rows = conn.execute(
        """
        SELECT c.id, c.session_id, c.project, c.started_at, c.role_mix,
               c.tool_names, c.text, v.distance AS score
        FROM chunks_vec v
        JOIN chunks c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        """,
        (qvec, max(limit * 5, 50)),
    ).fetchall()
    out: list[SearchHit] = []
    for r in rows:
        if not _matches(r, filters):
            continue
        out.append(
            SearchHit(
                chunk_id=r["id"],
                session_id=r["session_id"],
                project=r["project"],
                started_at=r["started_at"],
                role_mix=r["role_mix"],
                score=float(r["score"]),
                text=r["text"],
            )
        )
        if len(out) >= limit:
            break
    return out
