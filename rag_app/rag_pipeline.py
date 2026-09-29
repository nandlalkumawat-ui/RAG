"""The RAG pipeline: connects step 1 to step 5 and the history database.

TWO jobs:

  INGEST (run once per book)                 ASK (run for every question)
  ------------------------------             ------------------------------
  book file                                  question
    -> step1: read text page by page           -> step3: embed the question
    -> step2: cut into chunks                  -> step4: find top_k closest chunks
    -> step3: embed every chunk                -> step4b: (optional) rerank them and
    -> step4: save in ChromaDB                            keep the best rerank_top_n
                                               -> step5: LLM answers from those chunks
                                               -> save question + answer in SQLite

With the reranker enabled, step 4 fetches retrieval_top_k candidates instead
of top_k, so the reranker has more chunks to choose from.
"""

import logging
import time
from pathlib import Path

from .history_db import HistoryDB
from .step1_loader import list_books, load_book
from .step2_chunker import load_tokenizer, split_into_chunks
from .step3_embedder import Embedder
from .step4_vector_store import VectorStore
from .step4b_reranker import Reranker
from .step5_llm import ask_llm, build_prompt

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = load_tokenizer(config["embedding_model"])
        self.embedder = Embedder(config["embedding_model"])
        self.vector_store = VectorStore(config["chroma_folder"])
        self.history = HistoryDB(config["history_db_file"])
        self.reranker = self._load_reranker()
        logger.info("RAG pipeline ready. %d chunks already in the vector DB.",
                    self.vector_store.count_chunks())

    def _load_reranker(self) -> Reranker | None:
        """None means "no reranking": search() then works exactly like before."""
        if not self.config["reranker"]["enabled"]:
            logger.info("Reranker disabled: using the top %d chunks from the vector DB",
                        self.config["top_k"])
            return None
        if self.config["rerank_top_n"] > self.config["retrieval_top_k"]:
            logger.warning("rerank_top_n (%d) > retrieval_top_k (%d): will fetch %d candidates "
                           "instead, so the reranker has nothing to filter out",
                           self.config["rerank_top_n"], self.config["retrieval_top_k"],
                           self.config["rerank_top_n"])
        try:
            return Reranker(self.config["reranker"]["model"])
        except Exception:
            logger.exception("Could not load reranker model %s. Continuing WITHOUT reranking.",
                             self.config["reranker"]["model"])
            return None

    def default_top_k(self) -> int:
        """How many chunks go to the LLM when the request does not say."""
        return self.config["rerank_top_n"] if self.reranker else self.config["top_k"]

    # ------------------------------------------------------------------ INGEST
    def ingest_book(self, book_path: Path) -> dict:  #responsible for putting books into the RAG system. 
        """Read -> chunk -> embed -> store one book. Returns a small report."""
        start_time = time.time()
        pages = load_book(book_path)  # STEP 1
        if not pages:
            return {"file": book_path.name, "status": "skipped: no readable text", "chunks": 0}

        ids, texts, metadatas = [], [], []
        for page in pages:
            chunks = split_into_chunks(  # STEP 2
                page["text"], self.tokenizer,
                self.config["chunk_size"], self.config["chunk_overlap"],
            )
            for chunk_number, chunk in enumerate(chunks):
                ids.append(f"{page['file']}|p{page['page']}|c{chunk_number}")
                texts.append(chunk)
                metadatas.append({"file": page["file"], "page": page["page"], "chunk": chunk_number})
        logger.info("%s: created %d chunks from %d pages", book_path.name, len(texts), len(pages))

        embeddings = self.embedder.embed(texts, show_progress=True)  # STEP 3
        self.vector_store.add(ids, texts, embeddings, metadatas)  # STEP 4

        seconds = round(time.time() - start_time, 1)
        logger.info("%s: stored %d chunks in %.1f s", book_path.name, len(texts), seconds)
        return {"file": book_path.name, "status": "indexed", "chunks": len(texts),
                "pages": len(pages), "seconds": seconds}

    def ingest_folder(self, rebuild: bool = False) -> list[dict]:
        """Index every book in the books folder.

        rebuild=False: only NEW books are indexed (fast when you add one book).
        rebuild=True : wipe the vector DB and index everything again
                       (needed after you change chunk_size or the embedding model).
        """
        books = list_books(self.config["books_folder"])
        logger.info("Found %d books in %s", len(books), self.config["books_folder"])
        if rebuild:
            logger.info("Rebuild requested: deleting all stored chunks")
            self.vector_store.delete_everything() # like chunk size we reduce then 

        report = []
        for book in books:
            if self.vector_store.has_file(book.name):
                report.append({"file": book.name, "status": "already indexed",
                               "chunks": self.vector_store.count_chunks(book.name)})
            else:
                report.append(self.ingest_book(book))
        return report

    def list_indexed_books(self) -> list[dict]:
        books = list_books(self.config["books_folder"])
        return [{"file": b.name,
                 "size_mb": round(b.stat().st_size / 1_000_000, 2),
                 "chunks_in_vector_db": self.vector_store.count_chunks(b.name)}
                for b in books]

    def delete_book(self, file_name: str) -> None:
        """Remove a book from the folder AND its chunks from the vector DB."""
        path = Path(self.config["books_folder"]) / Path(file_name).name
        if not path.is_file():
            raise FileNotFoundError(f"No book named {file_name}")
        self.vector_store.delete_file(path.name)
        path.unlink()  # Deletes the actual PDF file from the books folder.
        logger.info("Deleted book %s", path.name)

    # --------------------------------------------------------------------- ASK
    def search(self, question: str, top_k: int | None = None) -> list[dict]:
        """Retrieval only: find the top_k chunks that best match the question.

        Reranker off: the top_k closest chunks from the vector DB.
        Reranker on : fetch retrieval_top_k candidates, rerank, keep the best top_k.
        """
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")
        if self.vector_store.count_chunks() == 0:
            raise ValueError("No books are indexed yet. Run POST /ingest first.")

        top_k = top_k or self.default_top_k()
        # Never fetch fewer candidates than the number of chunks we must return
        n_candidates = max(self.config["retrieval_top_k"], top_k) if self.reranker else top_k
        question_vector = self.embedder.embed([question])[0]  # STEP 3
        hits = self.vector_store.search(question_vector, n_candidates)  # STEP 4

        for hit in hits:
            hit["citation"] = f"{hit['file']}, page {hit['page']}, chunk {hit['chunk']}"
        logger.info("Retrieved %d candidates for %r: %s", len(hits), question[:60],
                    [(h["citation"], h["similarity"]) for h in hits])

        if self.reranker and hits: # and reranker and hits both are exist then only excute this step
            hits = self._rerank(question, hits, top_k)  # hits means retrieved chunks [question: xyz, hits: [chunk1, chunk2,...], top_k: 5]
        return hits

    def _rerank(self, question: str, hits: list[dict], top_k: int) -> list[dict]:
        logger.info("Reranking %d candidates", len(hits))
        try:
            reranked = self.reranker.rerank(question, hits, top_k)
        except Exception:
            # Better a slightly worse answer than no answer: keep the vector DB order
            logger.exception("Reranking failed, using the vector DB order instead")
            return hits[:top_k]
        logger.info("Selected top %d chunks after reranking: %s", len(reranked),
                    [{"citation": r["citation"], "similarity": r["similarity"],
                      "rerank_score": r["rerank_score"]} for r in reranked])
        return reranked

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Full RAG: retrieve chunks, let the LLM answer, save to history."""
        top_k = top_k or self.default_top_k()
        sources = self.search(question, top_k)

        prompt = build_prompt(question, sources)
        start_time = time.time()
        answer = ask_llm(prompt, self.config)  # STEP 5
        logger.info("LLM answered in %.1f s", time.time() - start_time)

        for number, source in enumerate(sources, start=1):
            source["source_number"] = number  # matches [1], [2]... in the answer

        qa_id = self.history.save(question.strip(), answer, sources, top_k)
        return {"id": qa_id, "question": question.strip(), "answer": answer, "sources": sources}
