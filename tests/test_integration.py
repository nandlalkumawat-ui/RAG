"""End-to-end tests through the real API (like clicking in Swagger).

Uses the 3 small documents in tests/sample_docs and a temporary folder for the
databases, so your real books and history are never touched.

    pytest tests/test_integration.py -v

Most tests replace the LLM with a fake one so they run fast and without Ollama.
`test_real_llm_answer` uses the real Ollama and is skipped if Ollama is not running.
"""

import shutil
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_app import rag_pipeline
from rag_app.main import create_app
from rag_app.settings import load_config

SAMPLE_DOCS = Path(__file__).parent / "sample_docs"


def ollama_is_running(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rag")
    books = tmp / "books"
    shutil.copytree(SAMPLE_DOCS, books)
    cfg = load_config()
    cfg.update(books_folder=str(books),
               chroma_folder=str(tmp / "chroma_db"),
               history_db_file=str(tmp / "history.db"))
    # Test the plain vector search path (no 1 GB reranker download).
    # Reranking is covered with a fake model in test_reranker.py.
    cfg["reranker"] = {**cfg["reranker"], "enabled": False}
    return cfg


@pytest.fixture(scope="module")
def client(config):
    with TestClient(create_app(config)) as test_client:  # loads the models once
        yield test_client


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the real LLM with one that just echoes the first source."""
    def fake_ask_llm(prompt, config):
        return "FAKE ANSWER based on [1]"
    monkeypatch.setattr(rag_pipeline, "ask_llm", fake_ask_llm)


# The tests below run in this order and build on each other.

def test_ask_before_ingest_gives_clear_error(client):
    response = client.post("/ask", json={"question": "What is FAISS?"})
    assert response.status_code == 400
    assert "ingest" in response.json()["detail"]


def test_ingest_all_sample_docs(client):
    response = client.post("/ingest")
    assert response.status_code == 200
    body = response.json()
    assert {b["file"] for b in body["books"]} == {"sample1.txt", "sample2.txt", "sample3.txt"}
    assert all(b["status"] == "indexed" for b in body["books"])

    # Running it again must not add the same chunks twice
    again = client.post("/ingest").json()
    assert all(b["status"] == "already indexed" for b in again["books"])
    assert again["total_chunks"] == body["total_chunks"]


def test_search_finds_the_right_document(client):
    response = client.post("/search", json={"question": "vector search and nearest neighbours", "top_k": 1})
    assert response.status_code == 200
    assert response.json()[0]["file"] == "sample3.txt"  # the FAISS document


def test_ask_returns_answer_with_citations_and_saves_history(client, fake_llm):
    response = client.post("/ask", json={"question": "What are graph neural networks?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "FAKE ANSWER based on [1]"
    assert len(body["sources"]) == 2
    assert body["sources"][0]["file"] == "sample1.txt"  # the GNN document
    assert body["sources"][0]["citation"] == "sample1.txt, page 1, chunk 0"

    saved = client.get(f"/history/{body['id']}").json()
    assert saved["question"] == "What are graph neural networks?"
    assert saved["sources"][0]["file"] == "sample1.txt"


def test_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 400


def test_llm_down_gives_503(client, config, monkeypatch):
    monkeypatch.setitem(config, "ollama_url", "http://localhost:1")  # nothing listens here
    response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_upload_rejects_wrong_file_type(client):
    response = client.post("/books/upload", files={"file": ("notes.docx", b"hello")})
    assert response.status_code == 400


def test_upload_broken_pdf_does_not_crash(client):
    response = client.post("/books/upload", files={"file": ("broken.pdf", b"this is not a pdf")})
    assert response.status_code == 200
    assert response.json()["status"].startswith("skipped")


def test_delete_book_removes_its_chunks(client):
    assert client.delete("/books/sample2.txt").status_code == 200
    files = {b["file"] for b in client.get("/books").json()}
    assert "sample2.txt" not in files
    assert client.delete("/books/sample2.txt").status_code == 404


def test_history_delete(client):
    items = client.get("/history").json()
    assert len(items) >= 1
    assert client.delete(f"/history/{items[0]['id']}").status_code == 200
    assert client.get(f"/history/{items[0]['id']}").status_code == 404


def test_real_llm_answer(client, config):
    if not ollama_is_running(config["ollama_url"]):
        pytest.skip("Ollama is not running")
    response = client.post("/ask", json={"question": "What is FAISS used for?", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"][0]["file"] == "sample3.txt"
