"""Data access for files + chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone

from claude_recall.ingest.chunker import Chunk


def upsert_file(
    conn: sqlite3.Connection,
    *,
    path: str,
    project: str,
    session_id: str,
    mtime: float,
    sha256: str,
) -> None:
    conn.execute(
        """
        INSERT INTO files(path, project, session_id, mtime, sha256, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime=excluded.mtime,
            sha256=excluded.sha256,
            indexed_at=excluded.indexed_at
        """,
        (path, project, session_id, mtime, sha256, datetime.now(timezone.utc).isoformat()),
    )


def file_unchanged(conn: sqlite3.Connection, path: str, mtime: float, sha256: str) -> bool:
    row = conn.execute(
        "SELECT mtime, sha256 FROM files WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return False
    return abs(row["mtime"] - mtime) < 1e-6 and row["sha256"] == sha256


def delete_session_chunks(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))


def insert_chunks(conn: sqlite3.Connection, chunks: Iterable[Chunk]) -> int:
    rows = [
        (
            c.id,
            c.session_id,
            c.project,
            c.turn_index,
            c.start_uuid,
            c.end_uuid,
            c.started_at,
            c.ended_at,
            c.role_mix,
            ",".join(c.tool_names),
            int(c.has_tool_use),
            c.text,
            c.token_count,
        )
        for c in chunks
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks
        (id, session_id, project, turn_index, start_uuid, end_uuid,
         started_at, ended_at, role_mix, tool_names, has_tool_use, text, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def upsert_embeddings(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
    vectors_bytes: list[bytes],
) -> None:
    """Upsert embeddings into chunks_vec virtual table keyed by chunks.rowid."""
    if not chunk_ids:
        return
    placeholders = ",".join("?" * len(chunk_ids))
    rowmap = dict(
        conn.execute(
            f"SELECT id, rowid FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    )
    rows = []
    for cid, vec in zip(chunk_ids, vectors_bytes, strict=True):
        rid = rowmap.get(cid)
        if rid is None:
            continue
        rows.append((rid, vec))
    if not rows:
        return
    # vec0 supports REPLACE via DELETE + INSERT
    rids = [r[0] for r in rows]
    rid_placeholders = ",".join("?" * len(rids))
    conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({rid_placeholders})", rids)
    conn.executemany(
        "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
        rows,
    )


def delete_session_vectors(conn: sqlite3.Connection, session_id: str) -> None:
    rows = conn.execute(
        "SELECT rowid FROM chunks WHERE session_id = ?", (session_id,)
    ).fetchall()
    if not rows:
        return
    rids = [r["rowid"] for r in rows]
    placeholders = ",".join("?" * len(rids))
    conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", rids)


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    n_files = conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]
    n_chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    n_projects = conn.execute("SELECT COUNT(DISTINCT project) AS c FROM chunks").fetchone()["c"]
    return {"files": n_files, "chunks": n_chunks, "projects": n_projects}
