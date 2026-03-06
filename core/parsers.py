import os
from typing import List, Dict


def parse_file(file_path: str, filename: str) -> List[Dict]:
    """
    Parse a file into a list of page dicts: [{"text": str, "page_number": int}].
    Supported formats: .pdf, .docx, .txt, .md

    Raises:
        ValueError: Unsupported file type, or no text could be extracted.
        RuntimeError: File could not be opened/read.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext in (".txt", ".md"):
        return _parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Accepted: .pdf, .docx, .txt, .md")


def _parse_pdf(file_path: str) -> List[Dict]:
    import fitz  # PyMuPDF
    pages = []
    try:
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    pages.append({"text": text, "page_number": i + 1})
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}") from e
    if not pages:
        raise ValueError("No text could be extracted from the PDF.")
    return pages


def _parse_docx(file_path: str) -> List[Dict]:
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to read DOCX: {e}")
    if not text.strip():
        raise ValueError("No text could be extracted from the DOCX file.")
    return [{"text": text, "page_number": 1}]


def _parse_text(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")
    if not text.strip():
        raise ValueError("The file is empty.")
    return [{"text": text, "page_number": 1}]
