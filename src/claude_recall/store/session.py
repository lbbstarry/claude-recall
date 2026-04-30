"""Read access for sessions and chunks (rendering / export / inject)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredChunk:
    id: str
    session_id: str
    project: str
    turn_index: int
    started_at: str
    role_mix: str
    text: str


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> StoredChunk | None:
    """Lookup a chunk by exact id or by id-prefix (>=8 chars)."""
    row = conn.execute(
        """SELECT id, session_id, project, turn_index, started_at, role_mix, text
           FROM chunks WHERE id = ?""",
        (chunk_id,),
    ).fetchone()
    if row is None and len(chunk_id) >= 8:
        row = conn.execute(
            """SELECT id, session_id, project, turn_index, started_at, role_mix, text
               FROM chunks WHERE id LIKE ? LIMIT 2""",
            (chunk_id + "%",),
        ).fetchall()
        if not row:
            return None
        if len(row) > 1:
            raise ValueError(f"chunk prefix '{chunk_id}' is ambiguous")
        row = row[0]
    if row is None:
        return None
    return StoredChunk(
        id=row["id"],
        session_id=row["session_id"],
        project=row["project"],
        turn_index=row["turn_index"],
        started_at=row["started_at"],
        role_mix=row["role_mix"],
        text=row["text"],
    )


def list_session_chunks(
    conn: sqlite3.Connection, session_id: str, *, turn: int | None = None
) -> list[StoredChunk]:
    sql = """SELECT id, session_id, project, turn_index, started_at, role_mix, text
             FROM chunks WHERE session_id = ?"""
    params: list[object] = [session_id]
    if turn is not None:
        sql += " AND turn_index = ?"
        params.append(turn)
    sql += " ORDER BY turn_index ASC"
    rows = conn.execute(sql, params).fetchall()
    if not rows and len(session_id) >= 8:
        # try prefix match on session_id
        prefix_rows = conn.execute(
            "SELECT DISTINCT session_id FROM chunks WHERE session_id LIKE ? LIMIT 2",
            (session_id + "%",),
        ).fetchall()
        if len(prefix_rows) == 1:
            return list_session_chunks(conn, prefix_rows[0]["session_id"], turn=turn)
        if len(prefix_rows) > 1:
            raise ValueError(f"session prefix '{session_id}' is ambiguous")
    return [
        StoredChunk(
            id=r["id"],
            session_id=r["session_id"],
            project=r["project"],
            turn_index=r["turn_index"],
            started_at=r["started_at"],
            role_mix=r["role_mix"],
            text=r["text"],
        )
        for r in rows
    ]


def render_session_markdown(chunks: list[StoredChunk]) -> str:
    if not chunks:
        return ""
    head = chunks[0]
    lines = [
        f"# Session `{head.session_id}` — {head.project}",
        f"_started_ {head.started_at[:19]}  ·  _turns_ {len(chunks)}",
        "",
    ]
    for c in chunks:
        lines.append(f"## Turn {c.turn_index}  ·  {c.started_at[:19]}  ·  {c.role_mix}")
        lines.append("")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)
