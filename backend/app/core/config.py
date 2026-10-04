from pathlib import Path
import os


class Settings:
    app_name = "Vietnamese News Search Engine"
    environment = os.getenv("APP_ENV", "local")
    search_backend = os.getenv("SEARCH_BACKEND", "lexical")
    lexical_engine = os.getenv("LEXICAL_ENGINE", "bm25")
    lexical_index_path = Path(
        os.getenv("LEXICAL_INDEX_PATH", "data/indexes/lexical_bm25.pkl")
    )
    lexical_data_path = Path(
        os.getenv("LEXICAL_DATA_PATH", "data/chunks/corpus_chunks_v2_e5.json")
    )
    lexical_build_limit = int(os.getenv("LEXICAL_BUILD_LIMIT", "1000"))
    semantic_index_path = Path(
        os.getenv("SEMANTIC_INDEX_PATH", "data/embeddings/e5_base/semantic_e5_base.faiss")
    )
    semantic_metadata_path = Path(
        os.getenv("SEMANTIC_METADATA_PATH", "data/embeddings/e5_base/chunks_metadata.parquet")
    )
    semantic_config_path = Path(
        os.getenv("SEMANTIC_CONFIG_PATH", "data/embeddings/e5_base/embedding_config.json")
    )
    semantic_candidate_multiplier = int(os.getenv("SEMANTIC_CANDIDATE_MULTIPLIER", "5"))
    semantic_min_candidates = int(os.getenv("SEMANTIC_MIN_CANDIDATES", "50"))
    hybrid_rrf_k = int(os.getenv("HYBRID_RRF_K", "60"))
    hybrid_candidate_multiplier = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "5"))
    hybrid_require_semantic = os.getenv("HYBRID_REQUIRE_SEMANTIC", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    reranker_enabled = os.getenv("RERANKER_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_top_k = int(os.getenv("RERANKER_TOP_K", "50"))
    reranker_batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
    reranker_max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))


settings = Settings()
