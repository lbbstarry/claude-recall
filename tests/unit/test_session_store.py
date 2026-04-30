import pytest

from claude_recall.ingest.chunker import Chunk
from claude_recall.store import db, repo
from claude_recall.store import session as session_store


def _chunk(session_id: str, turn: int, text: str) -> Chunk:
    cid = f"{session_id}-{turn:04d}"
    return Chunk(
        id=cid,
        session_id=session_id,
        project="p1",
        turn_index=turn,
        start_uuid=f"u{turn}",
        end_uuid=f"u{turn}",
        started_at="2026-04-29T00:00:00Z",
        ended_at="2026-04-29T00:00:00Z",
        role_mix="user+assistant",
        tool_names=(),
        has_tool_use=False,
        text=text,
        token_count=len(text) // 4,
    )


def test_get_chunk_exact_and_prefix(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    repo.insert_chunks(conn, [_chunk("session-aaaaaaaa", 0, "hello")])
    conn.commit()
    assert session_store.get_chunk(conn, "session-aaaaaaaa-0000") is not None
    by_prefix = session_store.get_chunk(conn, "session-")
    assert by_prefix is not None and by_prefix.text == "hello"


def test_get_chunk_ambiguous_prefix(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    repo.insert_chunks(
        conn,
        [_chunk("aaaaaaaa-1", 0, "x"), _chunk("aaaaaaaa-2", 0, "y")],
    )
    conn.commit()
    with pytest.raises(ValueError):
        session_store.get_chunk(conn, "aaaaaaaa")


def test_list_session_chunks_orders_by_turn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    repo.insert_chunks(
        conn,
        [
            _chunk("sxxxxxxxxxxx", 2, "third"),
            _chunk("sxxxxxxxxxxx", 0, "first"),
            _chunk("sxxxxxxxxxxx", 1, "second"),
        ],
    )
    conn.commit()
    out = session_store.list_session_chunks(conn, "sxxxxxxxxxxx")
    assert [c.text for c in out] == ["first", "second", "third"]


def test_render_markdown(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    repo.insert_chunks(conn, [_chunk("sxxxxxxxxxxx", 0, "hi"), _chunk("sxxxxxxxxxxx", 1, "bye")])
    conn.commit()
    md = session_store.render_session_markdown(
        session_store.list_session_chunks(conn, "sxxxxxxxxxxx")
    )
    assert "# Session" in md
    assert "Turn 0" in md and "Turn 1" in md
    assert "hi" in md and "bye" in md
