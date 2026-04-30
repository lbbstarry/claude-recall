"""Embedder protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dim) float32 array, L2-normalized."""
        ...
