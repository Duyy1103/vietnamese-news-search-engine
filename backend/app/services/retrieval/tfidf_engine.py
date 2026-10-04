import heapq
import math

from backend.app.services.preprocessing.text_normalizer import (
    clean_display_text,
    normalize_text,
)
from backend.app.services.retrieval.base import SearchDocument, SearchResult


class TfidfSearchEngine:
    """
    Search engine lexical dung TF-IDF cosine similarity.

    Engine nay dung de lam baseline va so sanh voi BM25. Production lexical
    search nen uu tien BM25 vi ranking thuong on dinh hon cho search tin tuc.
    """

    def __init__(
        self,
        title_weight: int = 3,
        min_df: int = 1,
    ):
        self.title_weight = title_weight
        self.min_df = min_df

        self.documents: list[SearchDocument] = []
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.doc_vectors: list[dict[int, float]] = []

    def _build_document_text(self, document: SearchDocument) -> str:
        title_parts = [document.title,
                       document.title_processed] * self.title_weight
        body_parts = [
            document.content_processed,
            document.combined_processed,
            document.combined_unaccented,
            document.content,
        ]

        return " ".join(part for part in title_parts + body_parts if part)

    def build(self, documents: list[SearchDocument]) -> None:
        self.documents = documents
        self.vocabulary = {}
        self.idf = []
        self.doc_vectors = []

        tokenized_docs = [
            normalize_text(self._build_document_text(document)).split()
            for document in documents
        ]

        doc_freq: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        terms = sorted(
            term for term, freq in doc_freq.items() if freq >= self.min_df
        )
        self.vocabulary = {term: index for index, term in enumerate(terms)}

        total_docs = len(tokenized_docs)
        self.idf = [
            math.log((1 + total_docs) / (1 + doc_freq[term])) + 1
            for term in terms
        ]

        self.doc_vectors = [
            self._vectorize_tokens(tokens)
            for tokens in tokenized_docs
        ]

    def _vectorize_tokens(self, tokens: list[str]) -> dict[int, float]:
        term_counts: dict[int, int] = {}
        for token in tokens:
            term_index = self.vocabulary.get(token)
            if term_index is None:
                continue
            term_counts[term_index] = term_counts.get(term_index, 0) + 1

        if not term_counts:
            return {}

        max_tf = max(term_counts.values())
        vector = {
            term_index: (count / max_tf) * self.idf[term_index]
            for term_index, count in term_counts.items()
        }

        norm = sum(value * value for value in vector.values()) ** 0.5
        if norm == 0:
            return {}

        return {
            term_index: value / norm
            for term_index, value in vector.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            return []

        query_tokens = normalize_text(query).split()
        query_vector = self._vectorize_tokens(query_tokens)
        if not query_vector or not self.documents:
            return []

        scores: list[tuple[int, float]] = []
        query_items = query_vector.items()

        for doc_index, doc_vector in enumerate(self.doc_vectors):
            score = sum(
                query_weight * doc_vector.get(term_index, 0.0)
                for term_index, query_weight in query_items
            )
            if score > 0:
                scores.append((doc_index, score))

        top_items = heapq.nlargest(top_k, scores, key=lambda item: item[1])

        results = []
        for doc_index, score in top_items:
            document = self.documents[doc_index]
            result_doc_id = document.metadata.get("doc_id", document.id)
            chunk_id = document.metadata.get("chunk_id")
            results.append(
                SearchResult(
                    doc_id=result_doc_id,
                    score=score,
                    title=clean_display_text(document.title),
                    snippet=clean_display_text(document.content, max_chars=300),
                    url=document.url,
                    chunk_id=chunk_id,
                    source=document.source,
                    topic=document.topic,
                    author=document.author,
                    crawled_at=document.crawled_at,
                    metadata=document.metadata,
                )
            )

        return results
