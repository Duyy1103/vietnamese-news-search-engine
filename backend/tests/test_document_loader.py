import json

from backend.app.services.indexing.document_loader import load_documents


def test_load_documents_from_json_file_maps_dataset_fields(tmp_path):
    data_path = tmp_path / "news.json"
    records = [
        {
            "id": 1,
            "title": "C\u00c3\u00b4ng an Hu\u00e1\u00ba\u00bf",
            "content": "C\u00c3\u00b4ng an \u0111i\u1ec1u tra v\u1ee5 c\u01b0\u1edbp ti\u1ec7m v\u00e0ng.",
            "url": "https://example.com/1",
            "source": "docbao.vn",
            "topic": "Ph\u00e1p lu\u1eadt",
            "author": "T.G",
            "crawled_at": 1659344927355,
            "title_processed": "cong_an hue",
            "content_processed": "cong_an dieu_tra vu cuop tiem_vang",
            "combined_processed": "cong_an hue cong_an dieu_tra",
            "combined_unaccented": "cong_an hue cuop tiem_vang",
            "picture_count": 3,
        }
    ]
    data_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    documents = load_documents(data_path)

    assert len(documents) == 1
    assert documents[0].id == 1
    assert documents[0].title == "C\u00f4ng an Hu\u1ebf"
    assert documents[0].content.startswith("C\u00f4ng an")
    assert documents[0].url == "https://example.com/1"
    assert documents[0].source == "docbao.vn"
    assert documents[0].topic == "Ph\u00e1p lu\u1eadt"
    assert documents[0].author == "T.G"
    assert documents[0].crawled_at == 1659344927355
    assert documents[0].title_processed == "cong_an hue"
    assert documents[0].content_processed == "cong_an dieu_tra vu cuop tiem_vang"
    assert documents[0].combined_unaccented == "cong_an hue cuop tiem_vang"
    assert documents[0].metadata["picture_count"] == 3


def test_load_documents_with_limit(tmp_path):
    data_path = tmp_path / "news.json"
    records = [
        {"id": 1, "title": "A", "content": "A", "url": "https://example.com/1"},
        {"id": 2, "title": "B", "content": "B", "url": "https://example.com/2"},
    ]
    data_path.write_text(json.dumps(records), encoding="utf-8")

    documents = load_documents(data_path, limit=1)

    assert len(documents) == 1
    assert documents[0].id == 1


def test_load_documents_from_jsonl_file(tmp_path):
    data_path = tmp_path / "news.jsonl"
    records = [
        {"id": 1, "title": "A", "content": "A", "url": "https://example.com/1"},
        {"id": 2, "title": "B", "content": "B", "url": "https://example.com/2"},
    ]
    data_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    documents = load_documents(data_path)

    assert [document.id for document in documents] == [1, 2]


def test_load_documents_maps_chunk_dataset_fields(tmp_path):
    data_path = tmp_path / "chunks.json"
    records = [
        {
            "doc_id": "22648",
            "chunk_id": "22648_4",
            "url": "https://example.com/article",
            "topic": "showbiz",
            "raw_text": "Minh H\u00e1\u00ba\u00b1ng v\u00e0 doanh nh\u00c3\u00a2n Nguy\u00e1\u00bb\u2026n Qu\u00e1\u00bb\u2018c B\u00e1\u00ba\u00a3o.",
            "chunk_processed": "minh_hang doanh_nhan nguyen_quoc_bao",
            "chunk_unaccented": "minh_hang doanh_nhan nguyen_quoc_bao",
        }
    ]
    data_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    documents = load_documents(data_path)

    assert len(documents) == 1
    assert documents[0].id == "22648_4"
    assert documents[0].metadata["doc_id"] == "22648"
    assert documents[0].metadata["chunk_id"] == "22648_4"
    assert documents[0].url == "https://example.com/article"
    assert documents[0].topic == "showbiz"
    assert documents[0].combined_unaccented == "minh_hang doanh_nhan nguyen_quoc_bao"
