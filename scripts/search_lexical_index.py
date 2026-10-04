import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.services.indexing.lexical_index import load_lexical_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a built lexical index."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/indexes/lexical_bm25.pkl"),
        help="Path to built lexical pickle index.",
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
    engine = load_lexical_index(args.index_path)
    results = engine.search(args.query, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        chunk = f" chunk={result.chunk_id}" if result.chunk_id else ""
        print(f"{rank}. score={result.score:.4f} doc={result.doc_id}{chunk}")
        print(f"   title={result.title}")
        print(f"   url={result.url}")
        print(f"   snippet={result.snippet}")


if __name__ == "__main__":
    main()
