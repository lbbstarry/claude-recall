"""Parse Claude Code JSONL session files into typed Message records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Message:
    uuid: str
    session_id: str
    project: str
    parent_uuid: str | None
    role: str
    timestamp: str
    text: str
    tool_names: tuple[str, ...]

    @property
    def has_tool_use(self) -> bool:
        return bool(self.tool_names)


def extract_text(content: Any) -> tuple[str, tuple[str, ...]]:
    """Extract flat text and tool names from a message content payload."""
    if isinstance(content, str):
        return content, ()
    if not isinstance(content, list):
        return "", ()
    parts: list[str] = []
    tools: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "")
            if t:
                parts.append(t)
        elif btype == "tool_use":
            name = block.get("name", "?")
            tools.append(name)
            parts.append(f"[tool_use:{name}]")
        elif btype == "tool_result":
            inner_text, _ = extract_text(block.get("content", ""))
            if inner_text:
                parts.append(f"[tool_result] {inner_text}")
    return "\n".join(parts), tuple(tools)


def iter_messages(jsonl_path: Path) -> Iterator[Message]:
    """Yield Message objects from a single Claude Code JSONL file."""
    project = jsonl_path.parent.name
    session_id = jsonl_path.stem
    try:
        f = jsonl_path.open("r", encoding="utf-8")
    except OSError:
        return
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") not in ("user", "assistant"):
                continue
            msg = rec.get("message") or {}
            text, tools = extract_text(msg.get("content", ""))
            if not text.strip():
                continue
            yield Message(
                uuid=rec.get("uuid", ""),
                session_id=session_id,
                project=project,
                parent_uuid=rec.get("parentUuid"),
                role=msg.get("role", rec.get("type", "?")),
                timestamp=rec.get("timestamp", ""),
                text=text,
                tool_names=tools,
            )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
