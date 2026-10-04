from backend.app.services.retrieval.base import SearchResult
from backend.app.services.retrieval.reranker import Reranker


def make_result(doc_id: str, score: float) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_id=f"{doc_id}_0",
        score=score,
        title=f"Title {doc_id}",
        snippet=f"Snippet {doc_id}",
        url=f"https://example.com/{doc_id}",
    )


def test_reranker_orders_candidates_and_preserves_retrieval_metadata():
    reranker = Reranker.__new__(Reranker)
    reranker._score_pairs = lambda **kwargs: [0.2, 0.9]

    results = reranker.rerank(
        query="test",
        candidates=[make_result("1", 3.0), make_result("2", 1.0)],
        top_k=2,
    )

    assert [result.doc_id for result in results] == ["2", "1"]
    assert results[0].score == 0.9
    assert results[0].metadata["retrieval_score"] == 1.0
    assert results[0].metadata["rerank_score"] == 0.9
    assert results[0].metadata["rerank_rank"] == 1
    assert results[0].metadata["reranked"] is True
