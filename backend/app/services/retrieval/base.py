from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SearchDocument:
    """
    Document dau vao dung de build search index.

    Dataset bao hien tai co ca field raw text va field da xu ly san.
    Cac search engine nen doc tu object nay thay vi dung dict truc tiep
    de tranh sai key va de ranking API co contract on dinh.
    """

    id: str | int
    title: str
    content: str
    url: str
    source: str = ""
    topic: str = ""
    author: str = ""
    crawled_at: str | int | float | None = None

    # Cac field da preprocess trong dataset. Neu khong co thi de rong.
    title_processed: str = ""
    content_processed: str = ""
    combined_processed: str = ""
    combined_unaccented: str = ""

    # Noi chua cac field phu khong duoc khai bao rieng.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    Ket qua tra ve sau khi search.
    """

    doc_id: str | int
    score: float
    title: str
    snippet: str
    url: str
    chunk_id: str | int | None = None
    source: str = ""
    topic: str = ""
    author: str = ""
    crawled_at: str | int | float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchEngine(Protocol):
    """
    Interface chung cho moi search engine.

    BM25SearchEngine va TfidfSearchEngine chi can implement dung hai
    method nay la co the thay the nhau trong API.
    """

    def build(self, documents: list[SearchDocument]) -> None:
        """
        Build index tu danh sach documents.

        Method nay thuong duoc goi khi khoi dong app hoac khi can
        rebuild index sau khi co data moi.
        """
        ...

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Search documents theo query nguoi dung nhap vao.

        Args:
            query: Cau truy van goc cua user.
            top_k: So ket qua muon lay.

        Returns:
            Danh sach ket qua da sap xep giam dan theo score.
        """
        ...
