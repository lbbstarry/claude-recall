"""Group messages into per-turn chunks (one user msg + following assistant msgs)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from claude_recall.parsers.claude_code import Message

MAX_CHARS = 8000  # ~2k tokens, comfortable for embedding models


@dataclass(frozen=True)
class Chunk:
    id: str
    session_id: str
    project: str
    turn_index: int
    start_uuid: str
    end_uuid: str
    started_at: str
    ended_at: str
    role_mix: str
    tool_names: tuple[str, ...]
    has_tool_use: bool
    text: str
    token_count: int


def _chunk_id(session_id: str, turn_index: int) -> str:
    return hashlib.sha256(f"{session_id}:{turn_index}".encode()).hexdigest()[:32]


def _flush(buf: list[Message], session_id: str, project: str, turn_index: int) -> Chunk | None:
    if not buf:
        return None
    text = "\n\n".join(f"[{m.role}] {m.text}" for m in buf)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "…"
    roles = sorted({m.role for m in buf})
    tools = tuple(t for m in buf for t in m.tool_names)
    return Chunk(
        id=_chunk_id(session_id, turn_index),
        session_id=session_id,
        project=project,
        turn_index=turn_index,
        start_uuid=buf[0].uuid,
        end_uuid=buf[-1].uuid,
        started_at=buf[0].timestamp,
        ended_at=buf[-1].timestamp,
        role_mix="+".join(roles),
        tool_names=tools,
        has_tool_use=bool(tools),
        text=text,
        token_count=len(text) // 4,  # cheap heuristic; replace with tiktoken later
    )


def chunk_messages(messages: Iterable[Message]) -> Iterator[Chunk]:
    """Group a session's messages into turns.

    A turn starts at a user message and extends through following assistant
    messages until the next user message.
    """
    buf: list[Message] = []
    turn_index = 0
    session_id = ""
    project = ""
    for m in messages:
        session_id = m.session_id
        project = m.project
        if m.role == "user" and buf:
            chunk = _flush(buf, session_id, project, turn_index)
            if chunk is not None:
                yield chunk
                turn_index += 1
            buf = []
        buf.append(m)
    chunk = _flush(buf, session_id, project, turn_index)
    if chunk is not None:
        yield chunk
