"""Unit tests for the SQLite Q&A history.  Run:  pytest tests/test_history_db.py -v"""

from rag_app.history_db import HistoryDB

SOURCES = [{"citation": "book.pdf, page 1, chunk 0", "page": 1}]


def test_save_and_read_back(tmp_path):
    db = HistoryDB(str(tmp_path / "history.db"))
    qa_id = db.save("What is RAG?", "Retrieval + generation.", SOURCES, top_k=3)

    item = db.get_one(qa_id)
    assert item["question"] == "What is RAG?"
    assert item["answer"] == "Retrieval + generation."
    assert item["sources"] == SOURCES  # JSON text is turned back into a list
    assert item["top_k"] == 3


def test_get_all_newest_first(tmp_path):
    db = HistoryDB(str(tmp_path / "history.db"))
    db.save("first", "a", [], 3)
    db.save("second", "b", [], 3)
    assert [item["question"] for item in db.get_all()] == ["second", "first"]


def test_delete(tmp_path):
    db = HistoryDB(str(tmp_path / "history.db"))
    first = db.save("first", "a", [], 3)
    db.save("second", "b", [], 3)

    assert db.delete_one(first) is True
    assert db.delete_one(first) is False  # already gone
    assert db.get_one(first) is None
    assert db.delete_all() == 1
    assert db.get_all() == []
