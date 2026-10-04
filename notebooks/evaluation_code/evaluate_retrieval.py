"""Calculate document-level retrieval metrics from complete manual qrels."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def precision_at_k(relevances: list[int], k: int) -> float:
    return sum(relevance >= 1 for relevance in relevances[:k]) / k


def reciprocal_rank_at_k(relevances: list[int], k: int) -> float:
    for rank, relevance in enumerate(relevances[:k], start=1):
        if relevance >= 1:
            return 1 / rank
    return 0.0


def ndcg_at_k(relevances: list[int], ideal_relevances: list[int], k: int) -> float:
    """Use the project's direct 0/1/2 relevance gain convention."""
    dcg = sum(relevance / math.log2(rank + 1) for rank, relevance in enumerate(relevances[:k], start=1))
    ideal = sorted(ideal_relevances, reverse=True)[:k]
    idcg = sum(relevance / math.log2(rank + 1) for rank, relevance in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"query_id", "doc_id", "relevance"}
    if not rows:
        raise ValueError("qrels contains no labelled rows.")
    if missing := required - set(rows[0]):
        raise ValueError(f"qrels is missing columns: {sorted(missing)}")

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        query_id, doc_id = row["query_id"].strip(), row["doc_id"].strip()
        if not query_id or not doc_id:
            raise ValueError("qrels contains an empty query_id or doc_id.")
        if doc_id in qrels[query_id]:
            raise ValueError(f"Duplicate qrels row: {query_id}, {doc_id}")
        relevance = int(row["relevance"])
        if relevance not in {0, 1, 2}:
            raise ValueError(f"Invalid relevance for {query_id}, {doc_id}: {relevance}")
        qrels[query_id][doc_id] = relevance
    return qrels


def read_rankings(path: Path) -> dict[str, dict[str, list[str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"method", "query_id", "rank", "doc_id", "score"}
    if not rows:
        raise ValueError("rankings contains no rows.")
    if missing := required - set(rows[0]):
        raise ValueError(f"rankings is missing columns: {sorted(missing)}")

    unsorted: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method, query_id, doc_id = row["method"].strip(), row["query_id"].strip(), row["doc_id"].strip()
        rank = int(row["rank"])
        if not method or not query_id or not doc_id or rank < 1:
            raise ValueError(f"Invalid ranking row: {row}")
        unsorted[method][query_id].append((rank, doc_id))

    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for method, queries in unsorted.items():
        for query_id, rows_for_query in queries.items():
            rows_for_query.sort()
            ranks = [rank for rank, _ in rows_for_query]
            doc_ids = [doc_id for _, doc_id in rows_for_query]
            if len(ranks) != len(set(ranks)) or len(doc_ids) != len(set(doc_ids)):
                raise ValueError(f"Duplicate rank or doc_id for {method}, {query_id}")
            rankings[method][query_id] = doc_ids
    return rankings


def read_latency(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    raw_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        raw_values[row["method"].strip()][row["query_id"].strip()].append(float(row["elapsed_seconds"]))
    return {
        method: {query_id: sum(samples) / len(samples) for query_id, samples in queries.items()}
        for method, queries in raw_values.items()
    }


def write_csv(rows: list[dict[str, str | int | float]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=Path("data/evaluation/qrels.csv"))
    parser.add_argument("--rankings", type=Path, required=True, help="CSV: method,query_id,rank,doc_id,score")
    parser.add_argument("--latency", type=Path, default=Path("data/evaluation/latency_benchmark.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/evaluation_results.csv"))
    parser.add_argument("--per-query-output", type=Path, default=Path("data/evaluation/evaluation_per_query.csv"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--allow-unjudged", action="store_true")
    args = parser.parse_args()
    if args.k < 1:
        raise SystemExit("--k must be at least 1.")

    qrels = read_qrels(args.qrels)
    rankings = read_rankings(args.rankings)
    latency = read_latency(args.latency)
    unjudged: list[str] = []
    for method, queries in rankings.items():
        for query_id, doc_ids in queries.items():
            missing = [doc_id for doc_id in doc_ids if doc_id not in qrels.get(query_id, {})]
            if missing:
                unjudged.append(f"{method}/{query_id}: {', '.join(missing[:3])}")
    if unjudged and not args.allow_unjudged:
        raise SystemExit(
            f"{len(unjudged)} ranking groups contain documents without qrels. "
            f"Example: {unjudged[0]}"
        )

    metric_names = [f"precision@{args.k}", f"mrr@{args.k}", f"ndcg@{args.k}"]
    per_query_rows: list[dict[str, str | int | float]] = []
    aggregate_rows: list[dict[str, str | int | float]] = []
    for method, queries in sorted(rankings.items()):
        method_rows: list[dict[str, str | int | float]] = []
        for query_id, document_ids in sorted(queries.items()):
            labels = qrels.get(query_id, {})
            if not labels:
                continue
            relevances = [labels.get(doc_id, 0) for doc_id in document_ids]
            first_relevant_rank = next((rank for rank, label in enumerate(relevances[: args.k], 1) if label >= 1), 0)
            row: dict[str, str | int | float] = {
                "method": method,
                "query_id": query_id,
                "retrieved_count": len(document_ids),
                "relevant_retrieved": sum(label >= 1 for label in relevances[: args.k]),
                "first_relevant_rank": first_relevant_rank,
                "latency_seconds": latency.get(method, {}).get(query_id, ""),
                metric_names[0]: precision_at_k(relevances, args.k),
                metric_names[1]: reciprocal_rank_at_k(relevances, args.k),
                metric_names[2]: ndcg_at_k(relevances, list(labels.values()), args.k),
            }
            method_rows.append(row)
            per_query_rows.append(row)
        if method_rows:
            latency_values = [float(row["latency_seconds"]) for row in method_rows if row["latency_seconds"] != ""]
            aggregate_rows.append(
                {
                    "method": method,
                    "queries_evaluated": len(method_rows),
                    metric_names[0]: sum(float(row[metric_names[0]]) for row in method_rows) / len(method_rows),
                    metric_names[1]: sum(float(row[metric_names[1]]) for row in method_rows) / len(method_rows),
                    metric_names[2]: sum(float(row[metric_names[2]]) for row in method_rows) / len(method_rows),
                    "mean_latency_seconds": sum(latency_values) / len(latency_values) if latency_values else "",
                }
            )

    aggregate_fields = ["method", "queries_evaluated", *metric_names, "mean_latency_seconds"]
    per_query_fields = [
        "method", "query_id", "retrieved_count", "relevant_retrieved", "first_relevant_rank",
        "latency_seconds", *metric_names,
    ]
    write_csv(aggregate_rows, aggregate_fields, args.output)
    write_csv(per_query_rows, per_query_fields, args.per_query_output)
    print(f"Wrote {len(aggregate_rows)} method rows to: {args.output}")
    print(f"Wrote {len(per_query_rows)} per-query rows to: {args.per_query_output}")


if __name__ == "__main__":
    main()
