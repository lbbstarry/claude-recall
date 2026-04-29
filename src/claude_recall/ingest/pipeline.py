"""End-to-end ingest: scan JSONL → chunk → upsert (incremental via mtime+sha)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from claude_recall.ingest.chunker import chunk_messages
from claude_recall.parsers.claude_code import file_sha256, iter_messages
from claude_recall.store import repo


@dataclass
class IngestStats:
    files_seen: int = 0
    files_skipped: int = 0
    files_indexed: int = 0
    chunks_written: int = 0


def ingest_all(conn: sqlite3.Connection, projects_dir: Path) -> IngestStats:
    stats = IngestStats()
    if not projects_dir.exists():
        return stats
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
        conn.commit()
    return stats
