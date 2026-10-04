from collections import Counter
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.services.indexing.lexical_index import (
    build_lexical_index,
    load_lexical_index,
    save_lexical_index,
)
from backend.app.services.preprocessing.text_normalizer import normalize_text, strip_accents, tokenize
from backend.app.services.retrieval.base import SearchDocument, SearchResult
from backend.app.services.retrieval.hybrid_engine import HybridSearchEngine
from backend.app.services.retrieval.reranker import Reranker
from backend.app.services.retrieval.semantic_engine import SemanticSearchEngine


router = APIRouter()
logger = logging.getLogger(__name__)


class ClickEvent(BaseModel):
    article_id: str
    query: str = ""
    position: int | None = None


def normalize_search_mode(mode: str = "") -> str:
    normalized = (mode or settings.search_backend).lower().strip()
    if normalized in {"", "lexical", "keyword", "bm25", "tfidf"}:
        return "keyword"
    if normalized == "semantic":
        return "semantic"
    if normalized == "hybrid":
        return "hybrid"
    raise HTTPException(
        status_code=400,
        detail="Unsupported search mode. Use keyword, semantic, or hybrid.",
    )


@lru_cache(maxsize=4)
def get_search_engine(mode: str = "") -> Any:
    normalized_mode = normalize_search_mode(mode)
    if normalized_mode == "semantic":
        return get_semantic_engine()
    if normalized_mode == "hybrid":
        try:
            return HybridSearchEngine(
                keyword_engine=get_keyword_engine(),
                semantic_engine=get_semantic_engine(),
                rrf_k=settings.hybrid_rrf_k,
                candidate_multiplier=settings.hybrid_candidate_multiplier,
            )
        except Exception:
            if settings.hybrid_require_semantic:
                raise
            logger.exception("semantic backend unavailable, falling back to keyword")
            return get_keyword_engine()

    return get_keyword_engine()


@lru_cache(maxsize=1)
def get_keyword_engine() -> Any:
    index_path = settings.lexical_index_path
    if index_path.exists():
        return load_lexical_index(index_path)

    engine = build_lexical_index(
        data_path=settings.lexical_data_path,
        engine_type=settings.lexical_engine,
        limit=settings.lexical_build_limit,
    )
    save_lexical_index(engine, index_path)
    return engine


@lru_cache(maxsize=1)
def get_semantic_engine() -> SemanticSearchEngine:
    _assert_artifacts_exist(
        {
            "semantic index": settings.semantic_index_path,
            "semantic metadata": settings.semantic_metadata_path,
            "semantic config": settings.semantic_config_path,
        }
    )
    engine = SemanticSearchEngine.from_artifacts(
        index_path=settings.semantic_index_path,
        metadata_path=settings.semantic_metadata_path,
        config_path=settings.semantic_config_path,
        candidate_multiplier=settings.semantic_candidate_multiplier,
        min_candidates=settings.semantic_min_candidates,
    )
    logger.info(
        "loaded semantic artifacts index=%s metadata=%s vectors=%s model=%s",
        settings.semantic_index_path,
        settings.semantic_metadata_path,
        getattr(engine.index, "ntotal", None),
        engine.config.get("model_name"),
    )
    return engine


def _assert_artifacts_exist(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        detail = "Missing artifacts: " + "; ".join(missing)
        logger.error(detail)
        raise HTTPException(status_code=503, detail=detail)


def clear_engine_caches() -> None:
    for cached_function in (
        get_search_engine,
        get_keyword_engine,
        get_semantic_engine,
        get_reranker,
    ):
        cache_clear = getattr(cached_function, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    try:
        reranker = Reranker(
            model_name=settings.reranker_model,
            max_length=settings.reranker_max_length,
        )
    except RuntimeError as exc:
        logger.exception("failed to load reranker model=%s", settings.reranker_model)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("loaded reranker model=%s", settings.reranker_model)
    return reranker


def should_rerank(rerank: bool | None) -> bool:
    return settings.reranker_enabled if rerank is None else rerank


def maybe_rerank_results(
    *,
    query: str,
    results: list[SearchResult],
    top_k: int,
    rerank: bool | None,
) -> tuple[list[SearchResult], bool]:
    if not should_rerank(rerank) or not query or not results:
        return results[:top_k], False

    candidates = results[: max(top_k, settings.reranker_top_k)]
    reranked = get_reranker().rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
        batch_size=settings.reranker_batch_size,
    )
    return reranked, True


def get_search_mode(mode: str = "") -> str:
    return normalize_search_mode(mode)


def get_documents(engine: Any) -> list[SearchDocument]:
    return list(getattr(engine, "documents", []))


def result_to_payload(
    result: SearchResult,
    result_id: str,
    rank: int,
    *,
    query: str = "",
    highlight: bool = False,
) -> dict[str, Any]:
    metadata = result.metadata or {}
    highlighted = make_keyword_highlight(result, query) if highlight else None
    return {
        "id": result_id,
        "doc_id": str(result.doc_id),
        "chunk_id": str(result.chunk_id or ""),
        "score": result.score,
        "title": result.title,
        "summary": result.snippet,
        "url": result.url,
        "category": result.topic or "",
        "author": result.author or "",
        "source": result.source or source_from_url(result.url),
        "published_at": normalize_timestamp(result.crawled_at),
        "highlight": highlighted,
        "rank": rank,
        "metadata": metadata,
    }


def document_to_payload(document: SearchDocument, rank: int) -> dict[str, Any]:
    metadata = document.metadata or {}
    doc_id = metadata.get("doc_id", document.id)
    chunk_id = metadata.get("chunk_id", "")
    return {
        "id": str(chunk_id or doc_id),
        "doc_id": str(doc_id),
        "chunk_id": str(chunk_id),
        "score": 0.0,
        "title": document.title,
        "summary": document.content[:300],
        "url": document.url,
        "category": document.topic or "",
        "author": document.author or "",
        "source": document.source or source_from_url(document.url),
        "published_at": normalize_timestamp(document.crawled_at),
        "highlight": None,
        "rank": rank,
        "metadata": metadata,
    }


def normalize_timestamp(value: str | int | float | None) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, (int, float)):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
    return str(value)


def source_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        return url.split("/")[2].replace("www.", "")
    except IndexError:
        return ""


def make_facets(documents: list[SearchDocument]) -> dict[str, list[dict[str, Any]]]:
    def top_values(values: list[str]) -> list[dict[str, Any]]:
        counts = Counter(value for value in values if value)
        return [
            {"value": value, "count": count}
            for value, count in counts.most_common(20)
        ]

    return {
        "category": top_values([document.topic for document in documents]),
        "author": top_values([document.author for document in documents]),
        "source": top_values(
            [document.source or source_from_url(document.url) for document in documents]
        ),
    }


def make_keyword_highlight(result: SearchResult, query: str) -> dict[str, list[str]] | None:
    query_tokens = [
        token for token in dict.fromkeys(tokenize(query))
        if len(token) > 1
    ]
    if not query_tokens:
        return None

    highlighted_title = highlight_text(result.title, query_tokens)
    highlighted_summary = highlight_text(result.snippet, query_tokens)
    payload: dict[str, list[str]] = {}

    if highlighted_title != result.title:
        payload["title"] = [highlighted_title]
    if highlighted_summary != result.snippet:
        payload["summary"] = [highlighted_summary]

    return payload or None


def highlight_text(text: str, query_tokens: list[str]) -> str:
    folded_text, position_map = fold_text_with_positions(text)
    spans: list[tuple[int, int]] = []

    for token in query_tokens:
        pattern = re.compile(rf"(?<![0-9a-zA-Z_]){re.escape(token)}(?![0-9a-zA-Z_])")
        for match in pattern.finditer(folded_text):
            start = position_map[match.start()]
            end = position_map[match.end() - 1] + 1
            spans.append((start, end))

    if not spans:
        return text

    merged_spans = merge_spans(spans)
    parts: list[str] = []
    cursor = 0
    for start, end in merged_spans:
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        parts.append("<em>")
        parts.append(text[start:end])
        parts.append("</em>")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def fold_text_with_positions(text: str) -> tuple[str, list[int]]:
    folded_chars: list[str] = []
    position_map: list[int] = []

    for index, char in enumerate(text):
        folded = strip_accents(char).lower()
        for folded_char in folded:
            folded_chars.append(folded_char)
            position_map.append(index)

    return "".join(folded_chars), position_map


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def apply_filters(
    results: list[SearchResult],
    category: str,
    author: str,
    source: str,
    from_date: str = "",
    to_date: str = "",
) -> list[SearchResult]:
    filtered = []
    for result in results:
        result_source = result.source or source_from_url(result.url)
        if category and result.topic != category:
            continue
        if author and result.author != author:
            continue
        if source and result_source != source:
            continue
        # Loc theo khoang thoi gian neu co from_date hoac to_date
        if from_date or to_date:
            result_date = _parse_result_date(result.crawled_at)
            if from_date and result_date and result_date < from_date:
                continue
            if to_date and result_date and result_date > to_date:
                continue
        filtered.append(result)
    return filtered


def _parse_result_date(crawled_at: str | int | float | None) -> str:
    """
    Chuyen crawled_at bat ky dinh dang nao sang chuoi YYYY-MM-DD.
    Tra ve chuoi rong neu khong parse duoc.
    """
    if crawled_at in ("", None):
        return ""
    if isinstance(crawled_at, (int, float)):
        # Unix timestamp: giay hoac mili-giay
        ts = crawled_at / 1000 if crawled_at > 10_000_000_000 else crawled_at
        return time.strftime("%Y-%m-%d", time.gmtime(ts))
    date_str = str(crawled_at)
    # Lay 10 ky tu dau (YYYY-MM-DD) neu co
    if len(date_str) >= 10:
        return date_str[:10]
    return ""


def sort_results_by_date(results: list[SearchResult]) -> list[SearchResult]:
    """Sap xep ket qua theo ngay moi nhat truoc (crawled_at giam dan)."""
    return sorted(
        results,
        key=lambda r: _parse_result_date(r.crawled_at) or "",
        reverse=True,
    )


def make_result_facets(results: list[SearchResult]) -> dict[str, list[dict[str, Any]]]:
    """
    Tinh facets tu danh sach SearchResult.

    Khac voi make_facets() nhan SearchDocument (keyword corpus),
    ham nay nhan ket qua search truc tiep — dung cho ca semantic va hybrid mode.
    """
    def top_values(values: list[str]) -> list[dict[str, Any]]:
        counts = Counter(value for value in values if value)
        return [
            {"value": value, "count": count}
            for value, count in counts.most_common(20)
        ]

    return {
        "category": top_values([r.topic for r in results]),
        "author": top_values([r.author for r in results]),
        "source": top_values([r.source or source_from_url(r.url) for r in results]),
    }


def deduplicate_results_by_document(results: list[SearchResult]) -> list[SearchResult]:
    deduplicated: list[SearchResult] = []
    seen_documents: set[str] = set()

    for result in results:
        document_key = str(result.doc_id or result.chunk_id or "")
        if not document_key:
            deduplicated.append(result)
            continue
        if document_key in seen_documents:
            continue
        seen_documents.add(document_key)
        deduplicated.append(result)

    return deduplicated


def deduplicate_documents_by_document(documents: list[SearchDocument]) -> list[SearchDocument]:
    deduplicated: list[SearchDocument] = []
    seen_documents: set[str] = set()

    for document in documents:
        metadata = document.metadata or {}
        document_key = str(metadata.get("doc_id") or document.id or "")
        if not document_key:
            deduplicated.append(document)
            continue
        if document_key in seen_documents:
            continue
        seen_documents.add(document_key)
        deduplicated.append(document)

    return deduplicated



def _resolve_semantic_engine(engine: Any) -> Any | None:
    """
    Lay SemanticSearchEngine tu bat ky engine nao (semantic hoac hybrid).
    Tra ve None neu engine khong co semantic component.
    """
    if hasattr(engine, "is_encoder_loaded"):
        return engine
    if hasattr(engine, "semantic_engine"):
        return engine.semantic_engine
    return None


@router.get("/semantic/status")
def semantic_status() -> dict[str, Any]:
    """
    Kiem tra trang thai E5 model:
    - ready=true: model da load, search chay ngay
    - ready=false, loading=true: dang load trong background
    - ready=false, loading=false: chua duoc kich hoat
    """
    try:
        engine = get_search_engine("semantic")
        sem = _resolve_semantic_engine(engine)
        ready = bool(sem and sem.is_encoder_loaded)
        return {"ready": ready, "loading": not ready}
    except Exception:
        return {"ready": False, "loading": False}


@router.post("/semantic/warmup")
def semantic_warmup() -> dict[str, str]:
    """
    Kich hoat load E5 model trong background thread.
    Return ngay lap tuc — dung de pre-warm model truoc khi search.
    """
    try:
        engine = get_search_engine("semantic")
        sem = _resolve_semantic_engine(engine)
        if sem is None:
            return {"status": "unavailable", "message": "Semantic engine chua duoc cau hinh."}
        if sem.is_encoder_loaded:
            return {"status": "ready", "message": "E5 model da san sang."}
        sem.warmup()
        return {"status": "loading", "message": "E5 model dang duoc load trong background."}
    except Exception as exc:
        logger.warning("semantic warmup failed: %s", exc)
        return {"status": "error", "message": str(exc)}


@router.get("/search")
def search(
    q: str = Query("", alias="q"),
    mode: str = Query("", description="Search mode: keyword, semantic, or hybrid."),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    rerank: bool | None = Query(
        None,
        description="Optionally rerank keyword, semantic, or hybrid candidates.",
    ),
    sort: str = Query("relevance"),
    category: str = Query(""),
    author: str = Query(""),
    source: str = Query(""),
    from_date: str = Query(""),
    to_date: str = Query(""),
) -> dict[str, Any]:
    started = time.perf_counter()
    selected_mode = normalize_search_mode(mode)
    query = q.strip()
    try:
        engine = get_search_engine(selected_mode)
        documents = get_documents(engine)
        offset = (page - 1) * page_size
        raw_results: list[SearchResult] = []

        # Neu la semantic/hybrid va model chua load:
        # kich hoat background load va tra ve tin hieu cho frontend ngay lap tuc.
        # Tranh ECONNRESET do proxy timeout khi cho model load 60s.
        if query and selected_mode in {"semantic", "hybrid"}:
            sem = _resolve_semantic_engine(engine)
            # getattr voi default=True de backward-compat voi FakeEngine trong test
            # (chi trigger warmup neu engine thuc su co is_encoder_loaded=False)
            if sem is not None and not getattr(sem, "is_encoder_loaded", True):
                sem.warmup()  # non-blocking, chay trong background thread
                latency_ms = round((time.perf_counter() - started) * 1000)
                logger.info("semantic model loading, returning early for query=%r", query)
                return {
                    "query": query,
                    "mode": get_search_mode(selected_mode),
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "latency_ms": latency_ms,
                    "results": [],
                    "facets": {"category": [], "author": [], "source": []},
                    "semantic_loading": True,
                    "explain": None,
                }

        if query:
            rerank_enabled = should_rerank(rerank)
            candidate_k = max(offset + page_size * 5, 200)
            if rerank_enabled:
                candidate_k = max(candidate_k, offset + settings.reranker_top_k)
            raw_results = engine.search(query, top_k=candidate_k)
            filtered_results = deduplicate_results_by_document(
                apply_filters(raw_results, category, author, source, from_date, to_date)
            )
            total = len(filtered_results)
            filtered_results, did_rerank = maybe_rerank_results(
                query=query,
                results=filtered_results,
                top_k=max(offset + page_size, page_size),
                rerank=rerank,
            )
            # Ap dung sap xep theo ngay neu user chon sort=newest
            # (override thu tu relevance/rerank)
            if sort == "newest":
                filtered_results = sort_results_by_date(filtered_results)
            page_results = filtered_results[offset: offset + page_size]
            payload_results = [
                result_to_payload(
                    result,
                    str(result.chunk_id or result.doc_id),
                    offset + index + 1,
                    query=query,
                    highlight=selected_mode in {"keyword", "hybrid"},
                )
                for index, result in enumerate(page_results)
            ]
        else:
            did_rerank = False
            filtered_documents = deduplicate_documents_by_document(
                [
                    document for document in documents
                    if (not category or document.topic == category)
                    and (not author or document.author == author)
                    and (not source or (document.source or source_from_url(document.url)) == source)
                ]
            )
            total = len(filtered_documents)
            page_documents = filtered_documents[offset: offset + page_size]
            payload_results = [
                document_to_payload(document, offset + index + 1)
                for index, document in enumerate(page_documents)
            ]

        latency_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "search mode=%s query=%r latency_ms=%s result_count=%s",
            selected_mode,
            query,
            latency_ms,
            total,
        )
        return {
            "query": query,
            "mode": get_search_mode(selected_mode),
            "total": total,
            "page": page,
            "page_size": page_size,
            "latency_ms": latency_ms,
            "results": payload_results,
            "facets": make_result_facets(raw_results) if query else make_facets(documents),
            "explain": {
                "query": query,
                "folded_query": normalize_text(query),
                "rerank": did_rerank,
                "reranker_model": settings.reranker_model if did_rerank else None,
                "raw_fields": ["title", "content", "raw_text"],
                "folded_fields": ["chunk_processed", "chunk_unaccented", "combined_unaccented"],
                "filters": {
                    "category": category or None,
                    "author": author or None,
                    "source": source or None,
                    "from_date": from_date or None,
                    "to_date": to_date or None,
                },
                "sort": sort,
            },
        }
    except HTTPException:
        logger.exception("search failed mode=%s query=%r", selected_mode, query)
        raise
    except Exception as e:
        import traceback
        logger.exception("search failed mode=%s query=%r: %s", selected_mode, query, str(e))
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": traceback.format_exc()})


@router.get("/suggest")
def suggest(q: str = Query("", alias="q")) -> dict[str, Any]:
    started = time.perf_counter()
    engine = get_search_engine("keyword")
    query = normalize_text(q)
    suggestions: list[dict[str, Any]] = []

    if query:
        seen = set()
        for document in get_documents(engine):
            title = document.title.strip()
            folded = normalize_text(title)
            if query in folded and title not in seen:
                seen.add(title)
                suggestions.append({"text": title, "type": "title", "weight": 1.0})
            if len(suggestions) >= 8:
                break

    return {
        "query": q,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "suggestions": suggestions,
    }


@router.get("/diagnostics")
def diagnostics() -> dict[str, Any]:
    keyword_ready = settings.lexical_index_path.exists() or settings.lexical_data_path.exists()
    semantic = inspect_semantic_artifacts()
    active_mode = normalize_search_mode("")

    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "mode": active_mode,
        "backend": {
            "name": active_mode,
            "source": str(Path.cwd() / settings.lexical_index_path),
            "reachable": (
                keyword_ready
                if active_mode == "keyword"
                else semantic["ready"]
                if active_mode == "semantic"
                else keyword_ready and semantic["ready"]
            ),
            "error": diagnostics_error(active_mode, keyword_ready, semantic["ready"]),
        },
        "indexes": {
            "article_data_path": str(Path.cwd() / settings.lexical_data_path),
            "suggestion_data_path": "",
            "article_count": semantic.get("num_vectors") or 0,
            "suggestion_count": 0,
            "vocabulary_size": 0,
            "postings_count": 0,
            "article_index_ready": keyword_ready,
            "suggestion_index_ready": keyword_ready,
            "keyword": {
                "engine": settings.lexical_engine,
                "index_path": str(Path.cwd() / settings.lexical_index_path),
                "data_path": str(Path.cwd() / settings.lexical_data_path),
                "ready": keyword_ready,
            },
            "semantic": semantic,
            "hybrid": {
                "ready": keyword_ready and semantic["ready"],
                "require_semantic": settings.hybrid_require_semantic,
                "rrf_k": settings.hybrid_rrf_k,
            },
            "reranker": {
                "enabled_by_default": settings.reranker_enabled,
                "model": settings.reranker_model,
                "top_k": settings.reranker_top_k,
                "batch_size": settings.reranker_batch_size,
                "max_length": settings.reranker_max_length,
            },
        },
        "reports": {
            "preprocessing_report_available": False,
            "preprocessing_summary": None,
            "evaluation_metrics_available": False,
            "evaluation_metrics": [],
        },
    }


def inspect_semantic_artifacts() -> dict[str, Any]:
    index_exists = settings.semantic_index_path.exists()
    metadata_exists = settings.semantic_metadata_path.exists()
    config_exists = settings.semantic_config_path.exists()
    config: dict[str, Any] = {}
    row_count = None
    error = None

    if config_exists:
        try:
            config = json.loads(settings.semantic_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            error = f"Invalid semantic config: {exc}"

    if metadata_exists and not error:
        # Lay row_count tu config.json (da doc o tren) thay vi load lai parquet.
        # Tranh tieu ton ~200MB RAM moi lan goi /diagnostics.
        row_count = config.get("num_vectors")

    missing = [
        name
        for name, exists in {
            "index": index_exists,
            "metadata": metadata_exists,
            "config": config_exists,
        }.items()
        if not exists
    ]

    return {
        "index_path": str(Path.cwd() / settings.semantic_index_path),
        "metadata_path": str(Path.cwd() / settings.semantic_metadata_path),
        "config_path": str(Path.cwd() / settings.semantic_config_path),
        "index_exists": index_exists,
        "metadata_exists": metadata_exists,
        "config_exists": config_exists,
        "ready": index_exists and metadata_exists and config_exists and error is None,
        "missing": missing,
        "error": error,
        "model_name": config.get("model_name"),
        "num_vectors": config.get("num_vectors", row_count),
        "embedding_dim": config.get("embedding_dim"),
    }


def diagnostics_error(mode: str, keyword_ready: bool, semantic_ready: bool) -> str | None:
    if mode == "keyword":
        return None if keyword_ready else "Missing lexical data/index"
    if mode == "semantic":
        return None if semantic_ready else "Missing semantic index/metadata/config"
    if keyword_ready and semantic_ready:
        return None
    missing = []
    if not keyword_ready:
        missing.append("keyword")
    if not semantic_ready:
        missing.append("semantic")
    return "Missing " + " and ".join(missing) + " backend artifacts"


@router.post("/events/click")
def track_click(_: ClickEvent) -> dict[str, str]:
    return {"status": "ok"}
