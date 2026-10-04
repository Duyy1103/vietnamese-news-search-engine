from backend.app.services.retrieval.base import SearchDocument
from backend.app.services.retrieval.tfidf_engine import TfidfSearchEngine


def test_tfidf_returns_relevant_document_first():
    documents = [
        SearchDocument(
            id=1,
            title="Liverpool thang Strasbourg",
            content="Du doan bong da Liverpool vs Strasbourg giao huu CLB.",
            url="https://example.com/liverpool",
        ),
        SearchDocument(
            id=2,
            title="Gia vang tang",
            content="Thi truong vang trong nuoc tang manh.",
            url="https://example.com/gold",
        ),
    ]
    engine = TfidfSearchEngine()
    engine.build(documents)

    results = engine.search("Liverpool Strasbourg bong da", top_k=2)

    assert len(results) > 0
    assert results[0].doc_id == 1
    assert results[0].score > 0


def test_tfidf_uses_chunk_metadata_for_chunk_results():
    documents = [
        SearchDocument(
            id="22648_4",
            title="Minh Hang",
            content="Doanh nhan Nguyen Quoc Bao hon Minh Hang 10 tuoi.",
            url="https://example.com/minh-hang",
            combined_unaccented="doanh_nhan nguyen_quoc_bao minh_hang",
            metadata={"doc_id": "22648", "chunk_id": "22648_4"},
        )
    ]
    engine = TfidfSearchEngine()
    engine.build(documents)

    results = engine.search("Nguyen Quoc Bao Minh Hang", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "22648"
    assert results[0].chunk_id == "22648_4"


def test_tfidf_returns_empty_list_for_empty_query():
    engine = TfidfSearchEngine()

    assert engine.search("", top_k=5) == []
