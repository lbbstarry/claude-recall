from claude_recall.ingest.chunker import Chunk
from claude_recall.search import bm25
from claude_recall.store import db, repo


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{i}",
        session_id="s1",
        project="p1",
        turn_index=i,
        start_uuid=f"u{i}",
        end_uuid=f"u{i}",
        started_at="2026-04-29T00:00:00Z",
        ended_at="2026-04-29T00:00:00Z",
        role_mix="user+assistant",
        tool_names=(),
        has_tool_use=False,
        text=text,
        token_count=len(text) // 4,
    )


def test_bm25_search_finds_chunks(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    repo.insert_chunks(
        conn,
        [
            _chunk(0, "how to use sqlite vec for hybrid search"),
            _chunk(1, "deploy a fastapi app on railway"),
            _chunk(2, "vector index with sqlite-vec and FTS5"),
        ],
    )
    conn.commit()
    hits = bm25.search(conn, "sqlite vec", limit=5)
    ids = [h.chunk_id for h in hits]
    assert "c0" in ids
    assert "c2" in ids
    assert "c1" not in ids


def test_bm25_search_no_match(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    repo.insert_chunks(conn, [_chunk(0, "hello world")])
    conn.commit()
    assert bm25.search(conn, "completely unrelated zzz") == []
