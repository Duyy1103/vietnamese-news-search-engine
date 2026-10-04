"""Audit an evaluation query set and save document-level retrieval rankings.

The query topic is deliberately not passed to SearchEngine.search(). It is
evaluation metadata, not user input, so using it as a retrieval filter would
leak information into the ranking.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from search_service import SearchEngine


REQUIRED_FIELDS = {"query_id", "query", "topic", "split", "status"}
AVAILABLE_METHODS = (
    "bm25", "semantic", "hybrid",
    "tfidf_accented_with_stopwords", "tfidf_accented_no_stopwords",
    "tfidf_unaccented_with_stopwords", "tfidf_unaccented_no_stopwords",
    "bm25cfg_accented_with_stopwords", "bm25cfg_accented_no_stopwords",
    "bm25cfg_unaccented_with_stopwords", "bm25cfg_unaccented_no_stopwords",
    "semanticcfg_accented_with_stopwords", "semanticcfg_accented_no_stopwords",
    "semanticcfg_unaccented_with_stopwords", "semanticcfg_unaccented_no_stopwords",
    "hybridcfg_accented_with_stopwords", "hybridcfg_accented_no_stopwords",
    "hybridcfg_unaccented_with_stopwords", "hybridcfg_unaccented_no_stopwords",
)
RANKING_FIELDS = ["method", "query_id", "rank", "doc_id", "score"]
DETAIL_FIELDS = [
    "method",
    "query_id",
    "query",
    "query_topic",
    "rank",
    "doc_id",
    "score",
    "title",
    "snippet",
    "source",
    "topic",
    "url",
    "chunk_id",
]
LATENCY_FIELDS = ["method", "query_id", "elapsed_seconds", "result_count", "top_k"]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def read_queries(path: Path) -> list[dict[str, str]]:
    """Read JSONL and fail early when a record cannot be used for evaluation."""
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    queries: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error

            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")

            normalized = {field: str(record[field]).strip() for field in REQUIRED_FIELDS}
            empty = [field for field, value in normalized.items() if not value]
            if empty:
                raise ValueError(f"Line {line_number} has empty fields: {empty}")
            if normalized["query_id"] in seen_ids:
                raise ValueError(f"Duplicate query_id: {normalized['query_id']}")
            seen_ids.add(normalized["query_id"])
            queries.append(normalized)

    if not queries:
        raise ValueError("The query file contains no usable records.")
    return queries


def print_audit(queries: list[dict[str, str]]) -> None:
    """Print the facts needed before accepting a draft query set."""
    print(f"Valid query records: {len(queries)}")
    for field in ("topic", "split", "status"):
        counts = Counter(query[field] for query in queries)
        formatted = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        print(f"{field}: {formatted}")


def clean_value(value: Any, maximum_length: int | None = None) -> str:
    """Convert metadata values to safe text for CSV output."""
    if value is None or value != value:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:maximum_length] if maximum_length is not None else text


def write_csv(rows: list[dict[str, str | int | float]], fieldnames: list[str], path: Path) -> None:
    """Atomically write a CSV after all requested methods complete successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary_file:
        writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path("data/evaluation/queries.jsonl"),
        help="JSONL with query_id, query, topic, split, and status.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=AVAILABLE_METHODS,
        default=["semantic"],
        help="Methods to run. BM25 and hybrid require lexical artefacts.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/rankings_tfidf.csv"),
        help="CSV written with method,query_id,rank,doc_id,score.",
    )
    parser.add_argument(
        "--details-output", type=Path, default=Path("data/evaluation/ranking_details.csv"),
        help="CSV with result metadata for manual relevance annotation.",
    )
    parser.add_argument(
        "--latency-output", type=Path, default=Path("data/evaluation/latency.csv"),
        help="CSV with post-warm-up retrieval time for every query.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Include the initial model/index loading cost in the first timed query.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow records whose status is draft. Use only for query review, not final metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the query file and print its distribution without loading search artefacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    queries = read_queries(args.queries)
    print_audit(queries)
    if args.dry_run:
        return

    draft_ids = [query["query_id"] for query in queries if query["status"].lower() == "draft"]
    if draft_ids and not args.allow_draft:
        preview = ", ".join(draft_ids[:5])
        raise SystemExit(
            f"{len(draft_ids)} query records are still draft ({preview}). "
            "Review them first or use --allow-draft for a non-final ranking run."
        )

    ranking_rows: list[dict[str, str | int | float]] = []
    detail_rows: list[dict[str, str | int | float]] = []
    latency_rows: list[dict[str, str | int | float]] = []
    for method in args.methods:
        # Each configuration can hold a large sparse matrix or FAISS index.
        # A fresh engine per method prevents all 16 configurations accumulating in RAM.
        engine = SearchEngine()
        if not args.no_warmup:
            print(f"Warming up {method}; this time is excluded from latency measurements...")
            try:
                engine.search(queries[0]["query"], mode=method, top_k=1)
            except Exception as error:
                raise RuntimeError(f"{method} warm-up failed: {error}") from error

        elapsed_times: list[float] = []
        progress = tqdm(queries, desc=f"{method} rankings", unit="query")
        for query in progress:
            started_at = time.perf_counter()
            try:
                # Do not pass query["topic"] here: it would use evaluation metadata as a filter.
                results = engine.search(query["query"], mode=method, top_k=args.top_k)
            except Exception as error:
                raise RuntimeError(
                    f"{method} failed on {query['query_id']} ({query['query']!r}): {error}"
                ) from error
            elapsed_times.append(time.perf_counter() - started_at)

            if results.empty:
                raise RuntimeError(
                    f"{method} returned no results for {query['query_id']} ({query['query']!r})."
                )
            if "doc_id" not in results or "score" not in results:
                raise RuntimeError(f"{method} returned a result without doc_id or score.")

            document_ids = results["doc_id"].astype(str).tolist()
            if len(document_ids) != len(set(document_ids)):
                raise RuntimeError(f"{method} returned duplicate doc_id values for {query['query_id']}.")

            for rank, (_, row) in enumerate(results.head(args.top_k).iterrows(), start=1):
                snippet = clean_value(row.get("text"), 500)
                if not snippet:
                    snippet = clean_value(row.get("content"), 500)
                result = {
                    "method": method,
                    "query_id": query["query_id"],
                    "rank": rank,
                    "doc_id": str(row["doc_id"]),
                    "score": float(row["score"]),
                }
                ranking_rows.append(result)
                detail_rows.append(
                    {
                        **result,
                        "query": query["query"],
                        "query_topic": query["topic"],
                        "title": clean_value(row.get("title")),
                        "snippet": snippet,
                        "source": clean_value(row.get("source")),
                        "topic": clean_value(row.get("topic")),
                        "url": clean_value(row.get("url")),
                        "chunk_id": clean_value(row.get("chunk_id")),
                    }
                )
            latency_rows.append(
                {
                    "method": method,
                    "query_id": query["query_id"],
                    "elapsed_seconds": elapsed_times[-1],
                    "result_count": len(results),
                    "top_k": args.top_k,
                }
            )

        average_seconds = sum(elapsed_times) / len(elapsed_times)
        print(f"{method}: {len(queries)} queries completed, mean retrieval time {average_seconds:.3f} s/query")
        del engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    write_csv(ranking_rows, RANKING_FIELDS, args.output)
    write_csv(detail_rows, DETAIL_FIELDS, args.details_output)
    write_csv(latency_rows, LATENCY_FIELDS, args.latency_output)
    print(f"Saved {len(ranking_rows)} ranking rows to: {args.output}")
    print(f"Saved {len(detail_rows)} annotation-detail rows to: {args.details_output}")
    print(f"Saved {len(latency_rows)} post-warm-up latency rows to: {args.latency_output}")


if __name__ == "__main__":
    main()
