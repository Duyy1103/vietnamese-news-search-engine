from dataclasses import replace
from typing import Any

from backend.app.services.retrieval.base import SearchResult


class Reranker:
    """
    Cross-encoder reranker cho SearchResult candidates.

    Reranker khong retrieve toan bo corpus. No chi cham lai cac candidates
    da lay tu keyword, semantic hoac hybrid.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        max_length: int = 512,
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing reranker dependencies. Install torch and transformers "
                "to use reranking."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        top_k: int = 10,
        batch_size: int = 8,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query or top_k <= 0 or not candidates:
            return []

        scores = self._score_pairs(
            query=query,
            documents=[self._build_document_text(candidate) for candidate in candidates],
            batch_size=batch_size,
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked_results = []
        for rank, (result, rerank_score) in enumerate(ranked[:top_k], start=1):
            metadata = dict(result.metadata or {})
            metadata.update(
                {
                    "retrieval_score": result.score,
                    "rerank_score": float(rerank_score),
                    "rerank_rank": rank,
                    "reranked": True,
                }
            )
            reranked_results.append(
                replace(
                    result,
                    score=float(rerank_score),
                    metadata=metadata,
                )
            )

        return reranked_results

    def _score_pairs(
        self,
        *,
        query: str,
        documents: list[str],
        batch_size: int,
    ) -> list[float]:
        scores: list[float] = []

        with self.torch.no_grad():
            for start in range(0, len(documents), batch_size):
                batch_documents = documents[start: start + batch_size]
                pairs = [(query, document) for document in batch_documents]

                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with self.torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    outputs = self.model(**inputs)

                batch_scores = outputs.logits.view(-1).float().cpu().numpy()
                scores.extend(batch_scores.tolist())

                del inputs, outputs
                if self.device == "cuda":
                    self.torch.cuda.empty_cache()

        return scores

    def _build_document_text(self, result: SearchResult, max_chars: int = 1200) -> str:
        parts = []
        if result.title:
            parts.append(result.title)
        if result.topic:
            parts.append("Chuyen muc: " + result.topic)
        if result.snippet:
            parts.append(result.snippet)

        return "\n".join(parts)[:max_chars]


class NoopReranker:
    """
    Test helper / fallback-friendly reranker implementation.
    """

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        top_k: int = 10,
        batch_size: int = 8,
    ) -> list[SearchResult]:
        return candidates[:top_k]
