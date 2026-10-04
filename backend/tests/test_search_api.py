import json

from fastapi.testclient import TestClient

from backend.app.api.routes import search as search_route
from backend.app.main import app
from backend.app.services.retrieval.base import SearchResult


def test_search_api_returns_lexical_results(tmp_path, monkeypatch):
    data_path = tmp_path / "chunks.json"
    index_path = tmp_path / "lexical.pkl"
    records = [
        {
            "doc_id": "22648",
            "chunk_id": "22648_4",
            "url": "https://example.com/minh-hang",
            "topic": "showbiz",
            "raw_text": "Doanh nhan Nguyen Quoc Bao hon Minh Hang 10 tuoi.",
            "chunk_unaccented": "doanh_nhan nguyen_quoc_bao minh_hang",
        }
    ]
    data_path.write_text(json.dumps(records), encoding="utf-8")

    monkeypatch.setattr(search_route.settings, "lexical_data_path", data_path)
    monkeypatch.setattr(search_route.settings, "lexical_index_path", index_path)
    monkeypatch.setattr(search_route.settings, "lexical_build_limit", 10)
    monkeypatch.setattr(search_route.settings, "search_backend", "lexical")
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get("/search", params={"q": "Nguyen Quoc Bao", "page_size": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "keyword"
    assert payload["total"] == 1
    assert payload["results"][0]["doc_id"] == "22648"
    assert payload["results"][0]["chunk_id"] == "22648_4"
    assert payload["results"][0]["score"] > 0
    assert "<em>Nguyen</em>" in payload["results"][0]["highlight"]["summary"][0]
    assert "<em>Quoc</em>" in payload["results"][0]["highlight"]["summary"][0]
    assert "<em>Bao</em>" in payload["results"][0]["highlight"]["summary"][0]


def test_diagnostics_api_reports_index_state(tmp_path, monkeypatch):
    data_path = tmp_path / "chunks.json"
    data_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(search_route.settings, "lexical_data_path", data_path)
    monkeypatch.setattr(search_route.settings, "lexical_index_path", tmp_path / "missing.pkl")
    monkeypatch.setattr(search_route.settings, "search_backend", "lexical")
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"]["reachable"] is True
    assert payload["indexes"]["article_index_ready"] is True


def test_search_api_can_use_semantic_backend(monkeypatch):
    class FakeSemanticEngine:
        documents = []

        def search(self, query, top_k=10):
            return [
                SearchResult(
                    doc_id="2",
                    chunk_id="2_0",
                    score=0.91,
                    title="Cuop tiem vang",
                    snippet="Cong an bat nghi pham cuop tiem vang.",
                    url="https://example.com/crime",
                    topic="Phap luat",
                )
            ]

    monkeypatch.setattr(search_route.settings, "search_backend", "semantic")
    monkeypatch.setattr(search_route, "get_semantic_engine", lambda: FakeSemanticEngine())
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get(
        "/search",
        params={"q": "cuop tiem vang", "mode": "semantic", "page_size": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "semantic"
    assert payload["total"] == 1
    assert payload["results"][0]["doc_id"] == "2"

    monkeypatch.setattr(search_route.settings, "search_backend", "lexical")
    search_route.clear_engine_caches()


def test_search_api_deduplicates_chunk_results_by_document(monkeypatch):
    class FakeEngine:
        documents = []

        def search(self, query, top_k=10):
            return [
                SearchResult(
                    doc_id="2",
                    chunk_id="2_0",
                    score=0.91,
                    title="Cuop tiem vang",
                    snippet="Chunk dau tien cua cung mot bai.",
                    url="https://example.com/crime",
                    topic="Phap luat",
                ),
                SearchResult(
                    doc_id="2",
                    chunk_id="2_1",
                    score=0.87,
                    title="Cuop tiem vang",
                    snippet="Chunk thu hai cua cung mot bai.",
                    url="https://example.com/crime",
                    topic="Phap luat",
                ),
                SearchResult(
                    doc_id="3",
                    chunk_id="3_0",
                    score=0.71,
                    title="Gia vang",
                    snippet="Mot bai khac.",
                    url="https://example.com/gold",
                    topic="Kinh te",
                ),
            ]

    monkeypatch.setattr(search_route.settings, "search_backend", "keyword")
    monkeypatch.setattr(search_route, "get_search_engine", lambda mode="": FakeEngine())
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get("/search", params={"q": "vang", "page_size": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [result["doc_id"] for result in payload["results"]] == ["2", "3"]
    assert payload["results"][0]["chunk_id"] == "2_0"


def test_search_api_can_use_hybrid_backend(monkeypatch):
    class FakeEngine:
        documents = []

        def __init__(self, result):
            self.result = result

        def search(self, query, top_k=10):
            return [self.result]

    keyword_result = SearchResult(
        doc_id="1",
        chunk_id="1_0",
        score=3.2,
        title="Gia vang",
        snippet="Gia vang tang.",
        url="https://example.com/gold",
        topic="Kinh te",
    )
    semantic_result = SearchResult(
        doc_id="2",
        chunk_id="2_0",
        score=0.91,
        title="Cuop tiem vang",
        snippet="Cong an bat nghi pham cuop tiem vang.",
        url="https://example.com/crime",
        topic="Phap luat",
    )

    monkeypatch.setattr(search_route, "get_keyword_engine", lambda: FakeEngine(keyword_result))
    monkeypatch.setattr(search_route, "get_semantic_engine", lambda: FakeEngine(semantic_result))
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get(
        "/search",
        params={"q": "cuop tiem vang", "mode": "hybrid", "page_size": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "hybrid"
    assert payload["total"] == 2
    assert payload["results"][0]["metadata"]["matched_modes"]


def test_search_api_can_rerank_keyword_results(tmp_path, monkeypatch):
    class FakeReranker:
        def rerank(self, query, candidates, top_k=10, batch_size=8):
            reranked = list(reversed(candidates[:top_k]))
            for rank, result in enumerate(reranked, start=1):
                result.metadata["reranked"] = True
                result.metadata["rerank_rank"] = rank
                result.metadata["rerank_score"] = 100 - rank
            return reranked

    data_path = tmp_path / "chunks.json"
    index_path = tmp_path / "lexical.pkl"
    records = [
        {
            "doc_id": "1",
            "chunk_id": "1_0",
            "url": "https://example.com/1",
            "topic": "news",
            "raw_text": "Nguyen Quoc Bao va Minh Hang.",
            "chunk_unaccented": "nguyen_quoc_bao minh_hang",
        },
        {
            "doc_id": "2",
            "chunk_id": "2_0",
            "url": "https://example.com/2",
            "topic": "news",
            "raw_text": "Nguyen Quoc Bao xuat hien trong su kien.",
            "chunk_unaccented": "nguyen_quoc_bao su_kien",
        },
    ]
    data_path.write_text(json.dumps(records), encoding="utf-8")

    monkeypatch.setattr(search_route.settings, "lexical_data_path", data_path)
    monkeypatch.setattr(search_route.settings, "lexical_index_path", index_path)
    monkeypatch.setattr(search_route.settings, "lexical_build_limit", 10)
    monkeypatch.setattr(search_route.settings, "search_backend", "keyword")
    monkeypatch.setattr(search_route.settings, "reranker_top_k", 10)
    monkeypatch.setattr(search_route, "get_reranker", lambda: FakeReranker())
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get(
        "/search",
        params={"q": "Nguyen Quoc Bao", "mode": "keyword", "rerank": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["explain"]["rerank"] is True
    assert payload["results"][0]["metadata"]["reranked"] is True


def test_diagnostics_reports_missing_semantic_artifacts(tmp_path, monkeypatch):
    data_path = tmp_path / "chunks.json"
    data_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(search_route.settings, "search_backend", "keyword")
    monkeypatch.setattr(search_route.settings, "lexical_data_path", data_path)
    monkeypatch.setattr(search_route.settings, "lexical_index_path", tmp_path / "missing.pkl")
    monkeypatch.setattr(search_route.settings, "semantic_index_path", tmp_path / "missing.faiss")
    monkeypatch.setattr(search_route.settings, "semantic_metadata_path", tmp_path / "missing.parquet")
    monkeypatch.setattr(search_route.settings, "semantic_config_path", tmp_path / "missing.json")
    search_route.clear_engine_caches()

    client = TestClient(app)
    response = client.get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["indexes"]["keyword"]["ready"] is True
    assert payload["indexes"]["semantic"]["ready"] is False
    assert sorted(payload["indexes"]["semantic"]["missing"]) == ["config", "index", "metadata"]
