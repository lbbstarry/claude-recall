"""Smoke tests for the FastAPI UI. Skips cleanly if [serve] extras absent."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient  # noqa: E402

from claude_recall.config import Config  # noqa: E402
from claude_recall.ingest.chunker import Chunk  # noqa: E402
from claude_recall.store import db, repo  # noqa: E402


def _chunk(session_id: str, turn: int, text: str) -> Chunk:
    return Chunk(
        id=f"{session_id}-{turn:04d}",
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    conn = db.connect(db_path)
    repo.insert_chunks(conn, [_chunk("sxxxxxxxxxxx", 0, "hello world")])
    conn.commit()
    conn.close()

    fake_cfg = Config(
        claude_projects_dir=tmp_path / "projects",
        data_dir=tmp_path,
        db_path=db_path,
    )
    monkeypatch.setattr("claude_recall.ui.server.load_config", lambda: fake_cfg)

    from claude_recall.ui.server import create_app

    return TestClient(create_app())


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "claudegrep" in r.text
    assert "search" in r.text.lower()


def test_search_bm25_returns_hit(client):
    r = client.post("/search", data={"q": "hello", "mode": "bm25", "limit": "10"})
    assert r.status_code == 200
    assert "hello world" in r.text


def test_search_no_match_renders_empty_state(client):
    r = client.post("/search", data={"q": "zzzzzzzz", "mode": "bm25", "limit": "10"})
    assert r.status_code == 200
    assert "No hits" in r.text


def test_session_view(client):
    r = client.get("/session/sxxxxxxxxxxx")
    assert r.status_code == 200
    assert "hello world" in r.text
    assert "turn 0" in r.text


def test_session_404(client):
    r = client.get("/session/missing000000")
    assert r.status_code == 404


def test_chunk_text_endpoint(client):
    r = client.get("/chunk/sxxxxxxxxxxx-0000/text")
    assert r.status_code == 200
    assert r.text == "hello world"


def test_chunk_text_404(client):
    r = client.get("/chunk/notarealchunk/text")
    assert r.status_code == 404
