"""Pool rankings from 16 retrieval methods into one manual-label CSV.

The output has one row per (query_id, doc_id). Existing manual labels are
reused only when all supplied label sources agree; new or conflicting pairs
remain blank for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETAILS = PROJECT_ROOT / "data" / "evaluation" / "ranking_details_16methods.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "annotation_candidates_16methods.csv"
DEFAULT_CONFLICTS = PROJECT_ROOT / "data" / "evaluation" / "annotation_label_conflicts_16methods.csv"
DEFAULT_QUERIES = PROJECT_ROOT / "data" / "evaluation" / "queries.jsonl"
DEFAULT_LABEL_SOURCES = (
    PROJECT_ROOT / "data" / "evaluation" / "annotation_candidates_with_label.csv",
    PROJECT_ROOT / "data" / "evaluation" / "annotation_candidates_bm25_4configs_labeled.csv",
    PROJECT_ROOT / "data" / "evaluation" / "annotation_candidates_tf-idf_4configs.csv",
)

CONFIGS = (
    "accented_with_stopwords",
    "accented_no_stopwords",
    "unaccented_with_stopwords",
    "unaccented_no_stopwords",
)
EXPECTED_METHODS = tuple(
    [f"tfidf_{config_name}" for config_name in CONFIGS]
    + [f"bm25cfg_{config_name}" for config_name in CONFIGS]
    + [f"semanticcfg_{config_name}" for config_name in CONFIGS]
    + [f"hybridcfg_{config_name}" for config_name in CONFIGS]
)
DETAIL_REQUIRED = {"method", "query_id", "rank", "doc_id", "score"}
BASE_FIELDS = [
    "query_id",
    "query",
    "query_topic",
    "doc_id",
    "relevance",
    "annotator",
    "notes",
    "title",
    "snippet",
    "url",
    "topic",
    "source",
    "methods",
    "best_rank",
]
CONFLICT_FIELDS = ["query_id", "doc_id", "labels_found", "label_sources"]


def read_queries(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    queries: dict[str, dict[str, str]] = {}
    order: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = {key: str(value).strip() for key, value in json.loads(line).items()}
        query_id = record.get("query_id", "")
        if not query_id or not record.get("query", ""):
            raise ValueError(f"Missing query_id or query at {path}:{line_number}")
        if query_id in queries:
            raise ValueError(f"Duplicate query_id in {path}: {query_id}")
        order[query_id] = len(order)
        queries[query_id] = record
    return queries, order


def read_details(path: Path, queries: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Ranking-detail file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not DETAIL_REQUIRED.issubset(reader.fieldnames):
            missing = sorted(DETAIL_REQUIRED - set(reader.fieldnames or []))
            raise ValueError(f"Ranking-detail file is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Ranking-detail file contains no rows.")

    methods = sorted({row["method"].strip() for row in rows})
    unexpected = sorted(set(methods) - set(EXPECTED_METHODS))
    missing = sorted(set(EXPECTED_METHODS) - set(methods))
    if unexpected or missing:
        parts = []
        if missing:
            parts.append(f"missing methods: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unexpected methods: {', '.join(unexpected)}")
        raise ValueError("The ranking detail file must contain exactly 16 expected methods; " + "; ".join(parts))

    seen: set[tuple[str, str, int, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        method = row["method"].strip()
        query_id = row["query_id"].strip()
        doc_id = row["doc_id"].strip()
        try:
            rank = int(row["rank"])
            float(row["score"])
        except ValueError as error:
            raise ValueError(f"Invalid rank or score at row {row_number}") from error
        if not query_id or query_id not in queries or not doc_id or rank < 1:
            raise ValueError(f"Invalid method/query/rank/doc_id at row {row_number}")
        key = (method, query_id, rank, doc_id)
        if key in seen:
            raise ValueError(f"Duplicate ranking-detail row at {row_number}: {key}")
        seen.add(key)
    return rows, methods


def read_label_sources(paths: list[Path]) -> dict[tuple[str, str], list[tuple[str, str, str, str]]]:
    labels: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Label source not found: {path}")
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"query_id", "doc_id", "relevance"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                missing = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"Label source {path} is missing columns: {missing}")
            for row in reader:
                relevance = row.get("relevance", "").strip()
                query_id = row.get("query_id", "").strip()
                doc_id = row.get("doc_id", "").strip()
                if relevance not in {"0", "1", "2"} or not query_id or not doc_id:
                    continue
                labels[(query_id, doc_id)].append(
                    (
                        relevance,
                        row.get("annotator", "").strip(),
                        row.get("notes", "").strip(),
                        str(path),
                    )
                )
    return labels


def first_text(rows: list[dict[str, str]], field: str) -> str:
    for row in rows:
        value = row.get(field, "").strip()
        if value:
            return value
    return ""


def output_fields(methods: list[str]) -> list[str]:
    fields = list(BASE_FIELDS)
    for method in methods:
        fields.extend([f"rank_{method}", f"score_{method}"])
    return fields


def build_candidates(
    details: list[dict[str, str]],
    methods: list[str],
    queries: dict[str, dict[str, str]],
    query_order: dict[str, int],
    labels: dict[tuple[str, str], list[tuple[str, str, str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in details:
        grouped[(row["query_id"].strip(), row["doc_id"].strip())].append(row)

    candidates: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    for (query_id, doc_id), rows in grouped.items():
        rows.sort(key=lambda row: (int(row["rank"]), row["method"]))
        label_rows = labels.get((query_id, doc_id), [])
        unique_labels = sorted({label[0] for label in label_rows})
        if len(unique_labels) == 1:
            selected = next(label for label in label_rows if label[0] == unique_labels[0])
            relevance, annotator, notes = selected[:3]
        else:
            relevance = annotator = notes = ""
            if unique_labels:
                conflicts.append(
                    {
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "labels_found": ",".join(unique_labels),
                        "label_sources": " | ".join(sorted({label[3] for label in label_rows})),
                    }
                )

        query = queries[query_id]
        candidate: dict[str, Any] = {
            "query_id": query_id,
            "query": query["query"],
            "query_topic": query.get("topic", ""),
            "doc_id": doc_id,
            "relevance": relevance,
            "annotator": annotator,
            "notes": notes,
            "title": first_text(rows, "title"),
            "snippet": first_text(rows, "snippet"),
            "url": first_text(rows, "url"),
            "topic": first_text(rows, "topic"),
            "source": first_text(rows, "source"),
            "methods": ",".join(sorted({row["method"].strip() for row in rows})),
            "best_rank": min(int(row["rank"]) for row in rows),
        }
        for method in methods:
            method_row = next((row for row in rows if row["method"].strip() == method), None)
            candidate[f"rank_{method}"] = method_row["rank"] if method_row else ""
            candidate[f"score_{method}"] = (
                f"{float(method_row['score']):.10f}" if method_row else ""
            )
        candidates.append(candidate)

    candidates.sort(key=lambda row: (query_order[row["query_id"]], row["best_rank"], row["doc_id"]))
    conflicts.sort(key=lambda row: (query_order[row["query_id"]], row["doc_id"]))
    return candidates, conflicts


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conflicts-output", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--label-sources", type=Path, nargs="+", default=list(DEFAULT_LABEL_SOURCES))
    args = parser.parse_args()

    queries, query_order = read_queries(args.queries)
    details, methods = read_details(args.details, queries)
    labels = read_label_sources(args.label_sources)
    candidates, conflicts = build_candidates(details, methods, queries, query_order, labels)
    write_csv(args.output, output_fields(methods), candidates)
    write_csv(args.conflicts_output, CONFLICT_FIELDS, conflicts)

    labelled = sum(row["relevance"] in {"0", "1", "2"} for row in candidates)
    print(f"Saved {len(candidates)} pooled candidates to: {args.output}")
    print(f"Transferred {labelled} agreed labels; {len(candidates) - labelled} need manual review.")
    print(f"Saved {len(conflicts)} label conflicts to: {args.conflicts_output}")


if __name__ == "__main__":
    main()
