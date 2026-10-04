import re
import unicodedata


TOKEN_RE = re.compile(r"[0-9a-zA-Z_]+")
MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c4",
    "\u00c6",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00e1\u00ba",
    "\u00e1\u00bb",
)
MOJIBAKE_START_CHARS = {
    "\u00c3",
    "\u00c4",
    "\u00c6",
    "\u00c2",
    "\u00e2",
    "\u00e1",
}

BOILERPLATE_PATTERNS = (
    r"\(adsbygoogle.*?push\(\{\}\);?",
    r"adsbygoogle\s+window\s+adsbygoogle\s+push",
    r"https?://\S+",
    r"trang thong tin dien tu.*",
    r"chinh sach bao mat.*",
    r"\brss\b",
)


def as_text(value) -> str:
    """
    Convert value trong JSON ve string an toan.
    """
    if value is None:
        return ""
    return str(value)


def mojibake_score(text: str) -> int:
    """
    Dem dau hieu loi encoding dang "TÃªn", "cÆ°á»›p".
    """
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def _decode_mojibake_fragment(fragment: str) -> str:
    candidates = [fragment]
    for encoding in ("cp1252", "latin1"):
        try:
            candidates.append(fragment.encode(encoding).decode("utf-8"))
        except UnicodeError:
            continue

    best = min(candidates, key=mojibake_score)
    return best if mojibake_score(best) < mojibake_score(fragment) else fragment


def _fix_mojibake_spans(text: str) -> str:
    """
    Sua cac cum mojibake nam xen trong chuoi da co mot phan dung.
    """
    result: list[str] = []
    index = 0
    text_len = len(text)

    while index < text_len:
        char = text[index]
        if char not in MOJIBAKE_START_CHARS:
            result.append(char)
            index += 1
            continue

        best_fragment = char
        best_repaired = char
        best_len = 1

        for length in range(6, 1, -1):
            end = index + length
            if end > text_len:
                continue

            fragment = text[index:end]
            if any(part.isspace() for part in fragment):
                continue

            repaired = _decode_mojibake_fragment(fragment)
            if mojibake_score(repaired) < mojibake_score(fragment):
                best_fragment = fragment
                best_repaired = repaired
                best_len = length
                break

        result.append(best_repaired)
        index += best_len

    return "".join(result)


def fix_mojibake(text: str) -> str:
    """
    Sua loi UTF-8 bi doc nham thanh cp1252/latin1 neu co.

    Ham nay giu nguyen text neu text da la tieng Viet dung. Neu chuoi bi
    loi xen ke voi tieng Viet dung, ham se sua theo tung cum nho.
    """
    text = as_text(text)
    original_score = mojibake_score(text)
    if original_score == 0:
        return text

    candidates = [
        text,
        _decode_mojibake_fragment(text),
        _fix_mojibake_spans(text),
    ]
    best = min(candidates, key=mojibake_score)
    return best if mojibake_score(best) < original_score else text


def strip_accents(text: str) -> str:
    """
    Bo dau tieng Viet de query co dau va khong dau deu match.
    """
    text = text.replace("\u0111", "d").replace("\u0110", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    """
    Chuan hoa text truoc khi dua vao search index.

    Pipeline nay duoc dung cho ca document va query de dam bao cung
    lowercase, cung bo dau, cung cach tach token.
    """
    text = fix_mojibake(text).lower()
    text = strip_accents(text)

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)

    tokens = TOKEN_RE.findall(text)
    expanded_tokens: list[str] = []

    for token in tokens:
        if len(token) == 1 and not token.isdigit():
            continue

        expanded_tokens.append(token)

        # Giu token ghep va them ca cac thanh phan rieng.
        # Vi du: "bong_da" -> "bong_da", "bong", "da".
        if "_" in token:
            expanded_tokens.extend(part for part in token.split("_") if len(part) > 1)

    return " ".join(expanded_tokens)


def tokenize(text: str) -> list[str]:
    """
    Chuyen text thanh list token de BM25/TF-IDF su dung.
    """
    return normalize_text(text).split()


def clean_display_text(text: str, max_chars: int | None = None) -> str:
    """
    Lam sach text de hien thi trong title/snippet ket qua search.
    """
    cleaned = re.sub(r"\s+", " ", fix_mojibake(text)).strip()
    if max_chars is not None and len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "..."
    return cleaned
