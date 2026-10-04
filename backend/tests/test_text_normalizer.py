from backend.app.services.preprocessing.text_normalizer import (
    clean_display_text,
    fix_mojibake,
    normalize_text,
    strip_accents,
    tokenize,
)


def test_strip_accents_removes_vietnamese_accents():
    assert strip_accents("b\u00f3ng \u0111\u00e1") == "bong da"
    assert strip_accents("\u0111\u1ea1i \u00fay c\u00f4ng an") == "dai uy cong an"


def test_fix_mojibake_repairs_common_dataset_encoding_errors():
    assert fix_mojibake("C\u00c3\u00b4ng an") == "C\u00f4ng an"
    assert fix_mojibake("c\u00c6\u00b0\u00e1\u00bb\u203ap") == "c\u01b0\u1edbp"


def test_normalize_text_lowercase_unaccent_and_expand_underscore():
    result = normalize_text("B\u00f3ng_\u0111\u00e1 Liverpool d\u1ef1_\u0111o\u00e1n")

    assert "bong_da" in result
    assert "bong" in result
    assert "da" in result
    assert "liverpool" in result
    assert "du_doan" in result
    assert "du" in result
    assert "doan" in result


def test_normalize_text_handles_mojibake_before_tokenizing():
    text = "C\u00c3\u00b4ng an c\u00c6\u00b0\u00e1\u00bb\u203ap ti\u00e1\u00bb\u2021m v\u00c3\u00a0ng"

    tokens = tokenize(text)

    assert "cong" in tokens
    assert "an" in tokens
    assert "cuop" in tokens
    assert "tiem" in tokens
    assert "vang" in tokens


def test_normalize_text_removes_ad_boilerplate():
    result = normalize_text("Tin ch\u00ednh adsbygoogle window adsbygoogle push")

    assert "tin" in result
    assert "adsbygoogle" not in result


def test_tokenize_returns_list_of_tokens():
    tokens = tokenize("C\u01b0\u1edbp ti\u1ec7m_v\u00e0ng Hu\u1ebf")

    assert tokens == ["cuop", "tiem_vang", "tiem", "vang", "hue"]


def test_clean_display_text_fixes_encoding_and_truncates():
    result = clean_display_text("C\u00c3\u00b4ng   an Hu\u00e1\u00ba\u00bf", max_chars=7)

    assert result == "C\u00f4ng an..."
