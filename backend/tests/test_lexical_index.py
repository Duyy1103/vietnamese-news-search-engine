import json

from backend.app.services.indexing.lexical_index import (
    build_lexical_index,
    load_lexical_index,
    save_lexical_index,
)


def test_build_save_and_load_lexical_index(tmp_path):
    data_path = tmp_path / "chunks.json"
    index_path = tmp_path / "lexical.pkl"
    records = [
        {
            "doc_id": "22648",
            "chunk_id": "22648_4",
            "url": "https://example.com/minh-hang",
            "raw_text": "Doanh nhan Nguyen Quoc Bao hon Minh Hang 10 tuoi.",
            "chunk_unaccented": "doanh_nhan nguyen_quoc_bao minh_hang",
        }
    ]
    data_path.write_text(json.dumps(records), encoding="utf-8")

    engine = build_lexical_index(data_path, engine_type="bm25")
    save_lexical_index(engine, index_path)

    loaded_engine = load_lexical_index(index_path)
    results = loaded_engine.search("Nguyen Quoc Bao Minh Hang", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "22648"
    assert results[0].chunk_id == "22648_4"
