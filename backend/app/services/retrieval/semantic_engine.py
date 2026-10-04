from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from backend.app.services.preprocessing.text_normalizer import clean_display_text
from backend.app.services.retrieval.base import SearchDocument, SearchResult

if TYPE_CHECKING:
    import pandas as pd


class SemanticSearchEngine:
    """
    Semantic search engine dung embedding E5 va FAISS artifacts.

    Engine nay khong build embedding tu SearchDocument luc runtime. No load lai
    cac artifact da tao tu notebook/script: FAISS index, metadata parquet va
    embedding_config.json.

    RAM optimization:
    - metadata duoc giu nguyen dang pd.DataFrame thay vi chuyen sang list[dict]
      (tiet kiem ~1-1.5GB cho 560K rows)
    - E5 model chi duoc load khi search() duoc goi lan dau (lazy init)
      (tiet kiem ~550MB luc startup)
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        config_path: str | Path | None = None,
        *,
        candidate_multiplier: int = 5,
        min_candidates: int = 50,
        index: Any | None = None,
        metadata: Any | None = None,
        config: dict[str, Any] | None = None,
        query_encoder: Callable[[list[str]], Any] | None = None,
    ):
        self.candidate_multiplier = candidate_multiplier
        self.min_candidates = min_candidates

        self.index = index
        # Giu metadata la DataFrame neu co the — tranh chuyen sang list[dict]
        # Chi accept list[dict] khi test inject metadata dang list (backward compat).
        self._metadata_df: pd.DataFrame | None = None
        self._metadata_list: list[dict[str, Any]] | None = None
        self._ingest_metadata(metadata)

        self.config = config or {}
        # query_encoder co the inject tu ngoai (vi du test), hoac se lazy-init
        self._query_encoder: Callable[[list[str]], Any] | None = query_encoder
        self._encoder_lock = threading.Lock()

        self.documents: list[SearchDocument] = []

        self._assert_artifact_paths(index_path, metadata_path, config_path)

        self.index_path = index_path
        self.metadata_path = metadata_path

        # config.json load is fast, we can keep it here
        if not self.config and config_path is not None:
            self.config = self._load_config(config_path)

        # KHONG load index, metadata hay model o day — se lazy-init trong _ensure_artifacts()
        # khi search() hoac warmup() duoc goi de tranh tieu ton 13s I/O va ~550MB RAM luc startup.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
        config_path: str | Path,
        *,
        candidate_multiplier: int = 5,
        min_candidates: int = 50,
    ) -> "SemanticSearchEngine":
        return cls(
            index_path=index_path,
            metadata_path=metadata_path,
            config_path=config_path,
            candidate_multiplier=candidate_multiplier,
            min_candidates=min_candidates,
        )

    def build(self, documents: list[SearchDocument]) -> None:
        """
        Giu interface chung voi SearchEngine.

        Semantic index duoc build offline bang embedding pipeline, nen runtime
        khong rebuild tu documents.
        """
        self.documents = documents

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query = query.strip()
        if not query:
            return []

        # Lazy-init artifacts va encoder — chi load model lan dau khi search duoc goi
        encoder = self._ensure_artifacts()

        if self.index is None or self._is_metadata_empty():
            return []

        candidate_k = min(
            max(top_k * self.candidate_multiplier, self.min_candidates),
            int(getattr(self.index, "ntotal", self._metadata_len())),
        )
        if candidate_k <= 0:
            return []

        query_prefix = self.config.get("query_prefix", "query: ")
        query_embedding = encoder([query_prefix + query])
        scores, ids = self.index.search(query_embedding, candidate_k)

        results: list[SearchResult] = []
        seen_chunks: set[str] = set()

        for score, row_index in zip(scores[0], ids[0]):
            if int(row_index) == -1:
                continue
            row = self._get_row(int(row_index))
            chunk_key = str(row.get("chunk_id") or row.get("embedding_index") or row_index)
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(chunk_key)

            results.append(self._row_to_result(row, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Internal: metadata helpers
    # ------------------------------------------------------------------

    def _ingest_metadata(self, metadata: Any | None) -> None:
        """
        Phan loai metadata dau vao:
        - DataFrame -> giu nguyen trong _metadata_df
        - list[dict] (vi du tu tests) -> giu trong _metadata_list
        - None -> bo trong
        """
        if metadata is None:
            return
        if hasattr(metadata, "iloc"):
            # Da la DataFrame
            self._metadata_df = metadata
        elif isinstance(metadata, list):
            # Backward compat: test inject list[dict]
            self._metadata_list = metadata
        elif hasattr(metadata, "to_dict"):
            # DataFrame-like
            self._metadata_df = metadata
        else:
            self._metadata_list = list(metadata)

    def _is_metadata_empty(self) -> bool:
        if self._metadata_df is not None:
            return len(self._metadata_df) == 0
        if self._metadata_list is not None:
            return len(self._metadata_list) == 0
        return True

    def _metadata_len(self) -> int:
        if self._metadata_df is not None:
            return len(self._metadata_df)
        if self._metadata_list is not None:
            return len(self._metadata_list)
        return 0

    def _get_row(self, row_index: int) -> dict[str, Any]:
        """
        Lay mot row theo integer index.
        - DataFrame: dung .iloc[row_index] — chi tao dict cho 1 row thay vi toan bo corpus
        - list[dict]: truy cap truc tiep
        """
        if self._metadata_df is not None:
            series = self._metadata_df.iloc[row_index]
            return {k: _to_native(v) for k, v in series.items()}
        if self._metadata_list is not None:
            return self._metadata_list[row_index]
        raise IndexError(f"No metadata available for row {row_index}")

    # ------------------------------------------------------------------
    # Internal: lazy encoder
    # Internal: lazy artifacts
    # ------------------------------------------------------------------

    def _ensure_artifacts(self) -> Callable[[list[str]], Any]:
        """
        Tra ve query_encoder, lazy-init index, metadata va encoder neu chua co.
        Su dung threading.Lock de dam bao chi load lan dau.
        """
        if self._query_encoder is not None and self.index is not None and not self._is_metadata_empty():
            return self._query_encoder

        with self._encoder_lock:
            # Double-checked locking
            if self.index is None and self.index_path is not None:
                import logging
                logging.getLogger("uvicorn.error").info("Lazy loading FAISS index...")
                self.index = self._load_faiss_index(self.index_path)
            if self._is_metadata_empty() and self.metadata_path is not None:
                import logging
                logging.getLogger("uvicorn.error").info("Lazy loading Semantic metadata...")
                self._metadata_df = self._load_metadata_df(self.metadata_path)
            if self._query_encoder is None:
                if not self.config:
                    raise RuntimeError(
                        "Semantic query encoder is not initialized and config is missing."
                    )
                import logging
                logging.getLogger("uvicorn.error").info("Lazy loading E5 encoder...")
                self._query_encoder = self._create_e5_query_encoder()
        return self._query_encoder  # type: ignore[return-value]

    @property
    def is_encoder_loaded(self) -> bool:
        """True neu E5 model da duoc load va san sang dung."""
        return self._query_encoder is not None

    def warmup(self) -> None:
        """
        Kich hoat lazy-load E5 encoder trong background thread.

        Goi ham nay ngay khi nhan biet user can semantic search de
        model san sang truoc khi co query thuc su.
        Khong block — return ngay lap tuc.
        """
        if self.is_encoder_loaded:
            return
        import threading
        t = threading.Thread(
            target=self._ensure_artifacts,
            daemon=True,
            name="e5-warmup",
        )
        t.start()

    # ------------------------------------------------------------------
    # Internal: conversion helpers
    # ------------------------------------------------------------------

    def _row_to_result(self, row: dict[str, Any], score: float) -> SearchResult:
        doc_id = row.get("doc_id", row.get("id", ""))
        chunk_id = row.get("chunk_id")
        text = clean_display_text(row.get("text") or row.get("raw_text") or "")
        title = clean_display_text(row.get("title") or text, max_chars=120)

        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "doc_id",
                "chunk_id",
                "title",
                "text",
                "raw_text",
                "url",
                "source",
                "topic",
                "author",
                "crawled_at",
            }
        }

        return SearchResult(
            doc_id=doc_id,
            chunk_id=chunk_id,
            score=score,
            title=title,
            snippet=clean_display_text(text, max_chars=300),
            url=str(row.get("url") or ""),
            source=clean_display_text(row.get("source") or ""),
            topic=clean_display_text(row.get("topic") or ""),
            author=clean_display_text(row.get("author") or ""),
            crawled_at=row.get("crawled_at"),
            metadata=metadata,
        )

    def _metadata_to_documents(self, metadata: list[dict[str, Any]]) -> list[SearchDocument]:
        documents = []
        for row in metadata:
            result = self._row_to_result(row, score=0.0)
            documents.append(
                SearchDocument(
                    id=result.chunk_id or result.doc_id,
                    title=result.title,
                    content=result.snippet,
                    url=result.url,
                    source=result.source,
                    topic=result.topic,
                    author=result.author,
                    crawled_at=result.crawled_at,
                    metadata={
                        **result.metadata,
                        "doc_id": result.doc_id,
                        "chunk_id": result.chunk_id,
                    },
                )
            )
        return documents

    # ------------------------------------------------------------------
    # Internal: artifact loaders
    # ------------------------------------------------------------------

    def _assert_artifact_paths(
        self,
        index_path: str | Path | None,
        metadata_path: str | Path | None,
        config_path: str | Path | None,
    ) -> None:
        paths = {
            "semantic index": index_path,
            "semantic metadata": metadata_path,
            "semantic config": config_path,
        }
        missing = [
            f"{name}: {path}"
            for name, path in paths.items()
            if path is not None and not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError("Missing artifacts: " + "; ".join(missing))

    def _load_metadata_df(self, metadata_path: str | Path) -> "pd.DataFrame":
        """
        Load metadata parquet thanh DataFrame — KHONG chuyen sang list[dict].
        Giu nguyen dang DataFrame de tiet kiem RAM (~1-1.5GB cho 560K rows).
        Khi can truy cap tung row, dung _get_row(idx) -> chi tao 1 dict tai thoi diem do.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'pandas'. Install pandas and pyarrow "
                "to load semantic metadata parquet."
            ) from exc

        return pd.read_parquet(metadata_path)

    def _load_faiss_index(self, index_path: str | Path) -> Any:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'faiss'. Install faiss-cpu or faiss-gpu "
                "to use semantic search."
            ) from exc

        return faiss.read_index(str(index_path))

    def _load_config(self, config_path: str | Path) -> dict[str, Any]:
        import json

        with Path(config_path).open("r", encoding="utf-8") as file:
            return json.load(file)

    def _create_e5_query_encoder(self) -> Callable[[list[str]], Any]:
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing semantic dependencies. Install torch, transformers "
                "and numpy to encode semantic queries."
            ) from exc

        model_name = self.config["model_name"]
        max_length = int(self.config.get("max_length", 512))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()

        def mean_pooling(last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state.float() * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        @torch.no_grad()
        def encode(texts: list[str]):
            inputs = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(**inputs)

            embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().astype(np.float32)

        return encode


def _is_na(value: Any) -> bool:
    """Kiem tra NaN/NaT/pd.NA an toan, khong import pandas o top-level."""
    if value is None:
        return False
    type_name = type(value).__name__
    if type_name in {"NAType", "NaTType"}:
        return True
    # Handle float, numpy.float32, numpy.float64 va moi kieu so co the la NaN.
    # Dung float(value) truoc de catch numpy scalar, tranh check isinstance cung.
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _to_native(value: Any) -> Any:
    """Convert numpy scalar to Python native type, and handle NAs."""
    if _is_na(value):
        return None
    # Neu la numpy scalar (np.int64, np.float32, v.v.), dung .item()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value
