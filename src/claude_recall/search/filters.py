"""Parse a free-text query for filter prefixes (project:foo since:7d role:user tool:Bash).

Returns (cleaned_query, filters_dict). Unknown prefixes pass through into the query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

KNOWN = {"project", "since", "role", "tool"}
_TOKEN = re.compile(r'(\w+):("(?:[^"\\]|\\.)*"|\S+)')


@dataclass(frozen=True)
class Filters:
    project: str | None = None
    since_iso: str | None = None
    role: str | None = None
    tool: str | None = None


def _parse_since(raw: str) -> str | None:
    """Accepts e.g. 7d, 24h, 30m, or ISO datetime."""
    m = re.fullmatch(r"(\d+)([dhm])", raw)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
        return (datetime.now(UTC) - delta).isoformat()
    try:
        return datetime.fromisoformat(raw).astimezone(UTC).isoformat()
    except ValueError:
        return None


def parse(query: str) -> tuple[str, Filters]:
    matches = list(_TOKEN.finditer(query))
    keep: dict[str, str] = {}
    spans: list[tuple[int, int]] = []
    for m in matches:
        key = m.group(1).lower()
        if key not in KNOWN:
            continue
        val = m.group(2)
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        keep[key] = val
        spans.append(m.span())
    cleaned = query
    for start, end in reversed(spans):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, Filters(
        project=keep.get("project"),
        since_iso=_parse_since(keep["since"]) if keep.get("since") else None,
        role=keep.get("role"),
        tool=keep.get("tool"),
    )
