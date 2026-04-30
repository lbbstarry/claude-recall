"""Local embedder via sentence-transformers (default: bge-small-zh-v1.5)."""

from __future__ import annotations

import numpy as np


class LocalEmbedder:
    """Wraps a sentence-transformers model. Lazy-loads on first call."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32, copy=False)
