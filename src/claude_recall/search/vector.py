"""Vector KNN search via sqlite-vec."""

from __future__ import annotations

import sqlite3

from claude_recall.embed.base import Embedder
from claude_recall.search.bm25 import SearchHit


def search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
) -> list[SearchHit]:
    qvec = embedder.embed([query])[0].tobytes()
    sql = """
        SELECT c.id, c.session_id, c.project, c.started_at, c.role_mix, c.text,
               v.distance AS score
        FROM chunks_vec v
        JOIN chunks c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
    """
    params: list[object] = [qvec, max(limit * 5, 50)]
    rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        if project and r["project"] != project:
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
