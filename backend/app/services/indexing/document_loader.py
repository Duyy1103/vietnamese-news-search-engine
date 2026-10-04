import json
from pathlib import Path
from typing import Iterator

from backend.app.services.preprocessing.text_normalizer import as_text, fix_mojibake
from backend.app.services.retrieval.base import SearchDocument


DOCUMENT_FIELDS = {
    "id",
    "doc_id",
    "chunk_id",
    "title",
    "content",
    "raw_text",
    "url",
    "source",
    "topic",
    "author",
    "crawled_at",
    "title_processed",
    "content_processed",
    "combined_processed",
    "combined_unaccented",
    "chunk_processed",
    "chunk_unaccented",
}


def iter_json_array(
    file_obj,
    limit: int | None = None,
    chunk_size: int = 1_048_576,
) -> Iterator[dict]:
    """
    Doc streaming JSON array lon ma khong load ca file vao RAM.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    started = False
    count = 0
    eof = False

    while True:
        if not eof:
            chunk = file_obj.read(chunk_size)
            if chunk == "":
                eof = True
            buffer += chunk

        while True:
            n = len(buffer)

            while pos < n and buffer[pos].isspace():
                pos += 1

            if not started:
                if pos >= n:
                    break
                if buffer[pos] != "[":
                    raise ValueError("Expected a JSON array.")
                started = True
                pos += 1
                continue

            while pos < n and (buffer[pos].isspace() or buffer[pos] == ","):
                pos += 1

            if pos >= n:
                break
            if buffer[pos] == "]":
                return

            try:
                item, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                break

            yield item
            count += 1
            if limit is not None and count >= limit:
                return

            pos = end

        if eof:
            return

        buffer = buffer[pos:]
        pos = 0


def iter_json_records(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    """
    Doc dataset dang JSON array hoac JSONL.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8", errors="replace") as file:
        prefix = file.read(4096)
        file.seek(0)
        first = next((char for char in prefix if not char.isspace()), "")

        if first == "[":
            yield from iter_json_array(file, limit=limit)
            return

        count = 0
        for line in file:
            line = line.strip().rstrip(",")
            if not line:
                continue

            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                return


def record_to_document(item: dict) -> SearchDocument:
    """
    Chuyen mot record JSON cua dataset bao thanh SearchDocument.
    """
    metadata = {key: value for key, value in item.items() if key not in DOCUMENT_FIELDS}

    if "chunk_id" in item:
        chunk_id = item.get("chunk_id", "")
        raw_text = fix_mojibake(item.get("raw_text", ""))
        metadata.update(
            {
                "doc_id": item.get("doc_id", ""),
                "chunk_id": chunk_id,
            }
        )

        return SearchDocument(
            id=chunk_id,
            title=clean_chunk_title(raw_text),
            content=raw_text,
            url=as_text(item.get("url", "")),
            source=fix_mojibake(item.get("source", "")),
            topic=fix_mojibake(item.get("topic", "")),
            author=fix_mojibake(item.get("author", "")),
            crawled_at=item.get("crawled_at"),
            content_processed=fix_mojibake(item.get("chunk_processed", "")),
            combined_processed=fix_mojibake(item.get("chunk_processed", "")),
            combined_unaccented=fix_mojibake(item.get("chunk_unaccented", "")),
            metadata=metadata,
        )

    return SearchDocument(
        id=item.get("id", ""),
        title=fix_mojibake(item.get("title", "")),
        content=fix_mojibake(item.get("content", "")),
        url=as_text(item.get("url", "")),
        source=fix_mojibake(item.get("source", "")),
        topic=fix_mojibake(item.get("topic", "")),
        author=fix_mojibake(item.get("author", "")),
        crawled_at=item.get("crawled_at"),
        title_processed=fix_mojibake(item.get("title_processed", "")),
        content_processed=fix_mojibake(item.get("content_processed", "")),
        combined_processed=fix_mojibake(item.get("combined_processed", "")),
        combined_unaccented=fix_mojibake(item.get("combined_unaccented", "")),
        metadata=metadata,
    )


def clean_chunk_title(text: str, max_chars: int = 120) -> str:
    """
    Tao title ngan cho chunk khi dataset khong co field title rieng.
    """
    text = " ".join(as_text(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def load_documents(
    path: str | Path,
    limit: int | None = None,
) -> list[SearchDocument]:
    """
    Doc dataset va chuyen tung record thanh SearchDocument.

    Tham so limit dung de test nhanh tren mot tap nho.
    """
    return [record_to_document(item) for item in iter_json_records(path, limit=limit)]
