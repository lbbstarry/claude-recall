"""On-disk embedding cache keyed by sha256(model_name|text)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from claude_recall.embed.base import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emb_cache (
    key  TEXT PRIMARY KEY,
    vec  BLOB NOT NULL
);
"""


def _key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbedCache:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def embed_with_cache(self, embedder: Embedder, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, embedder.dim), dtype=np.float32)
        keys = [_key(embedder.name, t) for t in texts]
        # Look up existing
        placeholders = ",".join("?" * len(keys))
        rows = self.conn.execute(
            f"SELECT key, vec FROM emb_cache WHERE key IN ({placeholders})",
            keys,
        ).fetchall()
        cached = {k: v for k, v in rows}

        # Compute missing
        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        if missing_idx:
            new_vecs = embedder.embed([texts[i] for i in missing_idx])
            inserts = []
            for j, i in enumerate(missing_idx):
                blob = new_vecs[j].tobytes()
                cached[keys[i]] = blob
                inserts.append((keys[i], blob))
            self.conn.executemany(
                "INSERT OR REPLACE INTO emb_cache(key, vec) VALUES (?, ?)",
                inserts,
            )
            self.conn.commit()

        # Reassemble in original order
        out = np.empty((len(texts), embedder.dim), dtype=np.float32)
        for i, k in enumerate(keys):
            out[i] = np.frombuffer(cached[k], dtype=np.float32)
        return out
