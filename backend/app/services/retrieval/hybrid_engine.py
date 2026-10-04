from dataclasses import replace
from typing import Protocol

from backend.app.services.retrieval.base import SearchResult


class Searchable(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        ...


class HybridSearchEngine:
    """
    Hybrid search dung Reciprocal Rank Fusion de gop keyword va semantic.
    """

    def __init__(
        self,
        keyword_engine: Searchable,
        semantic_engine: Searchable,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 5,
    ):
        self.keyword_engine = keyword_engine
        self.semantic_engine = semantic_engine
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.documents = list(getattr(keyword_engine, "documents", []))

    def build(self, documents) -> None:
        self.documents = documents

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0 or not query.strip():
            return []

        candidate_k = max(top_k * self.candidate_multiplier, top_k)
        keyword_results = self.keyword_engine.search(query, top_k=candidate_k)
        semantic_results = self.semantic_engine.search(query, top_k=candidate_k)

        fused: dict[str, dict] = {}
        self._add_results(fused, keyword_results, mode="keyword")
        self._add_results(fused, semantic_results, mode="semantic")

        ranked = sorted(
            fused.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        return [
            self._to_hybrid_result(item)
            for item in ranked[:top_k]
        ]

    def _add_results(
        self,
        fused: dict[str, dict],
        results: list[SearchResult],
        *,
        mode: str,
    ) -> None:
        for rank, result in enumerate(results, start=1):
            key = str(result.chunk_id or result.doc_id)
            if not key:
                continue

            item = fused.setdefault(
                key,
                {
                    "result": result,
                    "hybrid_score": 0.0,
                    "keyword_score": None,
                    "semantic_score": None,
                    "keyword_rank": None,
                    "semantic_rank": None,
                    "matched_modes": [],
                },
            )

            item["hybrid_score"] += 1 / (self.rrf_k + rank)
            item[f"{mode}_score"] = result.score
            item[f"{mode}_rank"] = rank
            if mode not in item["matched_modes"]:
                item["matched_modes"].append(mode)

            if mode == "semantic" or item["result"].score < result.score:
                item["result"] = result

    def _to_hybrid_result(self, item: dict) -> SearchResult:
        result = item["result"]
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "keyword_score": item["keyword_score"],
                "semantic_score": item["semantic_score"],
                "keyword_rank": item["keyword_rank"],
                "semantic_rank": item["semantic_rank"],
                "matched_modes": item["matched_modes"],
            }
        )
        return replace(result, score=float(item["hybrid_score"]), metadata=metadata)
