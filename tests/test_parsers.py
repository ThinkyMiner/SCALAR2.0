import pytest
from core.parsers import parse_file


def test_parse_txt(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world\nLine two")
    pages = parse_file(str(f), "doc.txt")
    assert len(pages) >= 1
    combined = " ".join(p["text"] for p in pages)
    assert "Hello world" in combined


def test_parse_md(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome markdown content")
    pages = parse_file(str(f), "doc.md")
    assert any("Title" in p["text"] or "markdown" in p["text"] for p in pages)


def test_parse_unsupported_type_raises(tmp_path):
    f = tmp_path / "doc.xyz"
    f.write_text("data")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(str(f), "doc.xyz")


def test_parse_returns_page_dicts(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("content here")
    pages = parse_file(str(f), "doc.txt")
    for page in pages:
        assert "text" in page
        assert "page_number" in page


def test_parse_empty_txt_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   ")
    with pytest.raises(ValueError):
        parse_file(str(f), "empty.txt")


def test_page_numbers_are_ints(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("some text here")
    pages = parse_file(str(f), "doc.txt")
    for page in pages:
        assert isinstance(page["page_number"], int)
        assert page["page_number"] >= 1
