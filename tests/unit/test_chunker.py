from claude_recall.ingest.chunker import chunk_messages
from claude_recall.parsers.claude_code import Message


def _msg(uuid: str, role: str, text: str, ts: str = "2026-04-29T00:00:00Z") -> Message:
    return Message(
        uuid=uuid,
        session_id="s1",
        project="p1",
        parent_uuid=None,
        role=role,
        timestamp=ts,
        text=text,
        tool_names=(),
    )


def test_chunker_groups_turns():
    msgs = [
        _msg("u1", "user", "question 1"),
        _msg("a1", "assistant", "answer 1"),
        _msg("a2", "assistant", "more"),
        _msg("u2", "user", "question 2"),
        _msg("a3", "assistant", "answer 2"),
    ]
    chunks = list(chunk_messages(msgs))
    assert len(chunks) == 2
    assert chunks[0].turn_index == 0
    assert chunks[0].start_uuid == "u1"
    assert chunks[0].end_uuid == "a2"
    assert "question 1" in chunks[0].text
    assert "answer 1" in chunks[0].text
    assert chunks[1].turn_index == 1
    assert chunks[1].start_uuid == "u2"
    assert chunks[1].role_mix == "assistant+user"


def test_chunker_handles_empty():
    assert list(chunk_messages([])) == []
