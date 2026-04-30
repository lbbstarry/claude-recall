"""End-to-end ingest: scan JSONL → chunk → embed → upsert (incremental)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from claude_recall.embed.base import Embedder
from claude_recall.embed.cache import EmbedCache
from claude_recall.ingest.chunker import chunk_messages
from claude_recall.parsers.claude_code import file_sha256, iter_messages
from claude_recall.store import repo

BATCH_SIZE = 64


@dataclass
class IngestStats:
    files_seen: int = 0
    files_skipped: int = 0
    files_indexed: int = 0
    chunks_written: int = 0
    chunks_embedded: int = 0


def ingest_all(
    conn: sqlite3.Connection,
    projects_dir: Path,
    *,
    embedder: Embedder | None = None,
    cache: EmbedCache | None = None,
) -> IngestStats:
    stats = IngestStats()
    if not projects_dir.exists():
        return stats

    pending_ids: list[str] = []
    pending_texts: list[str] = []

    def flush_embeddings() -> None:
        if not pending_ids or embedder is None or cache is None:
            return
        vecs = cache.embed_with_cache(embedder, pending_texts)
        repo.upsert_embeddings(conn, pending_ids, [v.tobytes() for v in vecs])
        stats.chunks_embedded += len(pending_ids)
        pending_ids.clear()
        pending_texts.clear()
        conn.commit()

    for jsonl in sorted(projects_dir.rglob("*.jsonl")):
        stats.files_seen += 1
        try:
            mtime = jsonl.stat().st_mtime
        except OSError:
            continue
        sha = file_sha256(jsonl)
        if repo.file_unchanged(conn, str(jsonl), mtime, sha):
            stats.files_skipped += 1
            continue
        session_id = jsonl.stem
        project = jsonl.parent.name
        repo.delete_session_vectors(conn, session_id)
        repo.delete_session_chunks(conn, session_id)
        chunks = list(chunk_messages(iter_messages(jsonl)))
        n = repo.insert_chunks(conn, chunks)
        repo.upsert_file(
            conn,
            path=str(jsonl),
            project=project,
            session_id=session_id,
            mtime=mtime,
            sha256=sha,
        )
        stats.files_indexed += 1
        stats.chunks_written += n
        if embedder is not None and cache is not None:
            for c in chunks:
                pending_ids.append(c.id)
                pending_texts.append(c.text)
                if len(pending_ids) >= BATCH_SIZE:
                    flush_embeddings()
        conn.commit()

    flush_embeddings()
    return stats
