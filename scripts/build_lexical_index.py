import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.services.indexing.lexical_index import (
    build_lexical_index,
    save_lexical_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lexical search index for Vietnamese news chunks/documents."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/chunks/corpus_chunks_v2_e5.json"),
        help="Path to JSON array or JSONL dataset.",
    )
    parser.add_argument(
        "--engine",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Lexical engine type.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/indexes/lexical_bm25.pkl"),
        help="Path to output pickle index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional record limit for local testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_lexical_index(
        data_path=args.data_path,
        engine_type=args.engine,
        limit=args.limit,
    )
    save_lexical_index(engine, args.output_path)

    print(f"Saved {args.engine} index to {args.output_path}")


if __name__ == "__main__":
    main()
