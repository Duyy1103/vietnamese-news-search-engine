from backend.app.services.retrieval.base import SearchResult
from backend.app.services.retrieval.hybrid_engine import HybridSearchEngine


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.documents = []

    def search(self, query, top_k=10):
        return self.results[:top_k]


def result(doc_id, score):
    return SearchResult(
        doc_id=doc_id,
        chunk_id=f"{doc_id}_0",
        score=score,
        title=f"Doc {doc_id}",
        snippet=f"Snippet {doc_id}",
        url=f"https://example.com/{doc_id}",
    )


def test_hybrid_engine_merges_keyword_and_semantic_with_rrf_metadata():
    keyword = FakeEngine([result("1", 8.0), result("2", 4.0)])
    semantic = FakeEngine([result("2", 0.9), result("3", 0.8)])
    engine = HybridSearchEngine(keyword, semantic, rrf_k=60)

    results = engine.search("test query", top_k=3)

    assert [item.doc_id for item in results] == ["2", "1", "3"]
    assert results[0].metadata["matched_modes"] == ["keyword", "semantic"]
    assert results[0].metadata["keyword_rank"] == 2
    assert results[0].metadata["semantic_rank"] == 1
    assert results[0].metadata["keyword_score"] == 4.0
    assert results[0].metadata["semantic_score"] == 0.9


def test_hybrid_engine_returns_empty_list_for_empty_query():
    engine = HybridSearchEngine(FakeEngine([]), FakeEngine([]))

    assert engine.search("", top_k=5) == []
