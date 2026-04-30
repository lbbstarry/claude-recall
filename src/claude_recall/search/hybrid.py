"""Hybrid retrieval: BM25 + vector, fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import sqlite3

from claude_recall.embed.base import Embedder
from claude_recall.search import bm25, vector
from claude_recall.search.bm25 import SearchHit

RRF_K = 60


def search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
    candidates: int = 50,
) -> list[SearchHit]:
    bm = bm25.search(conn, query, limit=candidates, project=project)
    vc = vector.search(conn, embedder, query, limit=candidates, project=project)

    bm_rank = {h.chunk_id: i + 1 for i, h in enumerate(bm)}
    vc_rank = {h.chunk_id: i + 1 for i, h in enumerate(vc)}
    by_id: dict[str, SearchHit] = {h.chunk_id: h for h in bm}
    for h in vc:
        by_id.setdefault(h.chunk_id, h)

    fused: list[tuple[float, SearchHit]] = []
    for cid, hit in by_id.items():
        score = 0.0
        if cid in bm_rank:
            score += 1.0 / (RRF_K + bm_rank[cid])
        if cid in vc_rank:
            score += 1.0 / (RRF_K + vc_rank[cid])
        fused.append((score, hit))

    fused.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, h in fused[:limit]:
        out.append(
            SearchHit(
                chunk_id=h.chunk_id,
                session_id=h.session_id,
                project=h.project,
                started_at=h.started_at,
                role_mix=h.role_mix,
                score=s,
                text=h.text,
            )
        )
    return out
