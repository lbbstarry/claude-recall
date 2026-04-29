"""Paths and config defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir


@dataclass(frozen=True)
class Config:
    claude_projects_dir: Path
    data_dir: Path
    db_path: Path


def load_config() -> Config:
    data_dir = Path(user_data_dir("claude-recall", appauthor=False))
    data_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        claude_projects_dir=Path.home() / ".claude" / "projects",
        data_dir=data_dir,
        db_path=data_dir / "recall.db",
    )
