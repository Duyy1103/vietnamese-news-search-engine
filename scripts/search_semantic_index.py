import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.services.retrieval.semantic_engine import SemanticSearchEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a built semantic E5 + FAISS index."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/embeddings/e5_base/semantic_e5_base.faiss"),
        help="Path to built FAISS index.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/embeddings/e5_base/chunks_metadata.parquet"),
        help="Path to semantic metadata parquet.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("data/embeddings/e5_base/embedding_config.json"),
        help="Path to embedding config JSON.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = SemanticSearchEngine.from_artifacts(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        config_path=args.config_path,
    )
    results = engine.search(args.query, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        chunk = f" chunk={result.chunk_id}" if result.chunk_id else ""
        print(f"{rank}. score={result.score:.4f} doc={result.doc_id}{chunk}")
        print(f"   title={result.title}")
        print(f"   topic={result.topic}")
        print(f"   url={result.url}")
        print(f"   snippet={result.snippet}")


if __name__ == "__main__":
    main()
