"""The web server. Start it with:

    uvicorn rag_app.main:app --reload

Then open http://127.0.0.1:8000/docs to use the Swagger UI.
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .rag_pipeline import RAGPipeline
from .settings import load_config
from .step1_loader import SUPPORTED_TYPES
from .step5_llm import LLMError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------- Request / response shapes (these show up in Swagger) ----------------
class QuestionRequest(BaseModel):
    question: str = Field(..., examples=["What is a rational agent?"])
    top_k: int | None = Field(None, ge=1, le=20, description="How many chunks to use. Empty = value from config.yaml")


class Source(BaseModel):
    source_number: int | None = None
    citation: str
    file: str
    page: int
    chunk: int
    similarity: float
    text: str


class AnswerResponse(BaseModel):
    id: int
    question: str
    answer: str
    sources: list[Source]


class HistoryItem(AnswerResponse):
    top_k: int
    created_at: str


# ------------------------------------ The app ------------------------------------
def create_app(config: dict | None = None) -> FastAPI:
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once when the server starts: load models + open databases
        app.state.rag = RAGPipeline(config)
        yield

    app = FastAPI(
        title="Book Q&A with RAG",
        description="Ask questions about the books in the `books/` folder.\n\n"
                    "**First time?** Run `POST /ingest` once, then use `POST /ask`.",
        lifespan=lifespan,
    )

    def rag() -> RAGPipeline:
        return app.state.rag

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/docs")

    # ---------------- 1. Books ----------------
    @app.get("/books", tags=["1. Books"], summary="List books and how many chunks each has")
    def list_books():
        return rag().list_indexed_books()

    @app.post("/books/upload", tags=["1. Books"], summary="Upload a PDF/TXT book and index it")
    def upload_book(file: UploadFile):
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED_TYPES:
            raise HTTPException(400, "Only .pdf and .txt files are allowed")
        destination = Path(config["books_folder"]) / name
        if destination.exists():
            raise HTTPException(409, f"{name} already exists. Delete it first to replace it.")
        with open(destination, "wb") as out:
            shutil.copyfileobj(file.file, out)
        return rag().ingest_book(destination)

    @app.delete("/books/{file_name}", tags=["1. Books"], summary="Delete a book and its chunks")
    def delete_book(file_name: str):
        try:
            rag().delete_book(file_name)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error))
        return {"deleted": file_name}

    @app.post("/ingest", tags=["1. Books"],
              summary="Chunk + embed all books in the books folder (run this first)")
    def ingest(rebuild: bool = False):
        """`rebuild=true` deletes all stored chunks and indexes every book again.
        Use it after you change `chunk_size`, `chunk_overlap` or `embedding_model`."""
        try:
            report = rag().ingest_folder(rebuild=rebuild)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error))
        return {"total_chunks": rag().vector_store.count_chunks(), "books": report}

    # ---------------- 2. Ask ----------------
    @app.post("/ask", tags=["2. Ask"], response_model=AnswerResponse,
              summary="Ask a question: retrieve chunks, then the LLM answers with citations")
    def ask(request: QuestionRequest):
        try:
            return rag().ask(request.question, request.top_k)
        except ValueError as error:  # empty question / nothing indexed
            raise HTTPException(400, str(error))
        except LLMError as error:  # Ollama is down, too slow, ...
            raise HTTPException(503, str(error))

    @app.post("/search", tags=["2. Ask"], response_model=list[Source],
              summary="Retrieval only (no LLM): see which chunks match a question")
    def search(request: QuestionRequest):
        try:
            return rag().search(request.question, request.top_k)
        except ValueError as error:
            raise HTTPException(400, str(error))

    # ---------------- 3. History ----------------
    @app.get("/history", tags=["3. History"], response_model=list[HistoryItem],
             summary="Past questions and answers (newest first)")
    def get_history(limit: int = 50):
        return rag().history.get_all(limit)

    @app.get("/history/{qa_id}", tags=["3. History"], response_model=HistoryItem)
    def get_history_item(qa_id: int):
        item = rag().history.get_one(qa_id)
        if item is None:
            raise HTTPException(404, f"No history item with id {qa_id}")
        return item

    @app.delete("/history/{qa_id}", tags=["3. History"])
    def delete_history_item(qa_id: int):
        if not rag().history.delete_one(qa_id):
            raise HTTPException(404, f"No history item with id {qa_id}")
        return {"deleted": qa_id}

    @app.delete("/history", tags=["3. History"], summary="Delete ALL history")
    def delete_all_history():
        return {"deleted_rows": rag().history.delete_all()}

    return app


app = create_app()
