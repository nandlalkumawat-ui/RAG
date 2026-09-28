"""STEP 1: Load books (PDF or TXT) from a folder and read their text.

For a PDF we keep the text page by page, so we can later tell the user
*which page* an answer came from. A TXT file is treated as a single page.
"""

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = [".pdf", ".txt"]


def read_pdf(file_path: Path) -> list[dict]:
    """Return one item per page: {"file": ..., "page": ..., "text": ...}"""
    pages = []
    reader = PdfReader(file_path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip empty pages (e.g. pictures only)
            pages.append({"file": file_path.name, "page": page_number, "text": text})
    return pages


def read_txt(file_path: Path) -> list[dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [{"file": file_path.name, "page": 1, "text": text}]


def load_book(file_path: Path) -> list[dict]:
    """Read one book. Returns an empty list if the file is broken or unsupported."""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            pages = read_pdf(file_path)
        elif suffix == ".txt":
            pages = read_txt(file_path)
        else:
            logger.warning("Skipping %s (only PDF and TXT are supported)", file_path.name)
            return []
    except Exception as error:  # a broken / encrypted / fake PDF should not crash the app
        logger.error("Could not read %s: %s", file_path.name, error)
        return []

    if not pages:
        logger.warning("No text found in %s (is it a scanned image PDF?)", file_path.name)
    else:
        logger.info("Loaded %s: %d pages with text", file_path.name, len(pages))
    return pages


def list_books(folder: str) -> list[Path]:
    """All PDF/TXT files in the folder, sorted by name."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Books folder not found: {folder}")
    return sorted(
        p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_TYPES
    )
