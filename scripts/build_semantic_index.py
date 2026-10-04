import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.services.indexing.semantic_index import build_semantic_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic E5 + FAISS index for Vietnamese news chunks."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/chunks/corpus_chunks_v2_e5.json"),
        help="Path to JSON array or JSONL chunks dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings/e5_base"),
        help="Directory to write embedding shards and FAISS artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "--text-col",
        default="raw_text",
        help="Record field used as semantic passage text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max tokenizer length.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Number of records per embedding shard.",
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
    result = build_semantic_index(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        text_col=args.text_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shard_size=args.shard_size,
        limit=args.limit,
    )

    print("Saved semantic artifacts:")
    print(f"  index: {result['index_path']}")
    print(f"  metadata: {result['metadata_path']}")
    print(f"  config: {result['config_path']}")
    print(f"  vectors: {result['num_vectors']}")
    print(f"  dim: {result['embedding_dim']}")


if __name__ == "__main__":
    main()
