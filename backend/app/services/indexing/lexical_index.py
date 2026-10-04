import pickle
from pathlib import Path

from backend.app.services.indexing.document_loader import load_documents
from backend.app.services.retrieval.base import SearchEngine
from backend.app.services.retrieval.bm25_engine import BM25SearchEngine
from backend.app.services.retrieval.tfidf_engine import TfidfSearchEngine


ENGINE_TYPES = {
    "bm25": BM25SearchEngine,
    "tfidf": TfidfSearchEngine,
}


def create_lexical_engine(engine_type: str = "bm25") -> SearchEngine:
    """
    Tao lexical search engine theo ten.
    """
    normalized_type = engine_type.lower().strip()
    engine_class = ENGINE_TYPES.get(normalized_type)
    if engine_class is None:
        supported = ", ".join(sorted(ENGINE_TYPES))
        raise ValueError(f"Unsupported lexical engine: {engine_type}. Supported: {supported}")

    return engine_class()


def build_lexical_index(
    data_path: str | Path,
    engine_type: str = "bm25",
    limit: int | None = None,
) -> SearchEngine:
    """
    Load dataset va build lexical index.
    """
    documents = load_documents(data_path, limit=limit)
    engine = create_lexical_engine(engine_type)
    engine.build(documents)
    return engine


def save_lexical_index(engine: SearchEngine, path: str | Path) -> None:
    """
    Luu lexical index bang pickle.

    File nay la artifact runtime, khong nen commit vao git.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as file:
        pickle.dump(engine, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_lexical_index(path: str | Path) -> SearchEngine:
    """
    Tai lexical index da build.
    """
    with Path(path).open("rb") as file:
        return pickle.load(file)
