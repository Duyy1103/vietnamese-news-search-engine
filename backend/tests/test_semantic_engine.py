from backend.app.services.retrieval.semantic_engine import SemanticSearchEngine


class FakeIndex:
    ntotal = 2

    def search(self, query_embedding, top_k):
        return [[0.91, 0.72]], [[1, 0]]


def test_semantic_engine_returns_results_from_metadata():
    metadata = [
        {
            "doc_id": "1",
            "chunk_id": "1_0",
            "text": "Gia vang trong nuoc tang manh.",
            "url": "https://example.com/gold",
            "topic": "Kinh te",
        },
        {
            "doc_id": "2",
            "chunk_id": "2_0",
            "text": "Cong an bat nghi pham cuop tiem vang.",
            "url": "https://example.com/crime",
            "topic": "Phap luat",
        },
    ]
    engine = SemanticSearchEngine(
        index=FakeIndex(),
        metadata=metadata,
        config={"query_prefix": "query: "},
        query_encoder=lambda texts: [[0.0]],
    )

    assert engine.documents == []

    results = engine.search("cuop tiem vang", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "2"
    assert results[0].chunk_id == "2_0"
    assert results[0].score == 0.91
    assert results[0].topic == "Phap luat"


def test_semantic_engine_returns_empty_list_for_empty_query():
    engine = SemanticSearchEngine(
        index=FakeIndex(),
        metadata=[{"doc_id": "1", "text": "Noi dung"}],
        config={"query_prefix": "query: "},
        query_encoder=lambda texts: [[0.0]],
    )

    assert engine.search("", top_k=5) == []
