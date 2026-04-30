"""Cross-encoder reranker (bge-reranker). Lazy-loaded, applied to top-N candidates."""

from __future__ import annotations

from claude_recall.search.bm25 import SearchHit


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.name)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        self._ensure_loaded()
        assert self._model is not None
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, hits, strict=True), key=lambda x: x[0], reverse=True)
        out = []
        for s, h in ranked[:top_k]:
            out.append(
                SearchHit(
                    chunk_id=h.chunk_id,
                    session_id=h.session_id,
                    project=h.project,
                    started_at=h.started_at,
                    role_mix=h.role_mix,
                    score=float(s),
                    text=h.text,
                )
            )
        return out
