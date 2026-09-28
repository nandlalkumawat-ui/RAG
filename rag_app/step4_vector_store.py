"""STEP 4: Save chunk vectors in ChromaDB and search them.

ChromaDB is a "vector database". It stores, for every chunk:
    id         -> "book_1.pdf|p12|c0"
    embedding  -> the 384 numbers from step 3
    document   -> the chunk text
    metadata   -> {"file": "book_1.pdf", "page": 12, "chunk": 0}

It is saved in a folder on disk (data/chroma_db), so the books are only
embedded once, not every time the server starts.

Search: we give it the question's vector and it returns the chunks whose
vectors are closest (cosine distance). similarity = 1 - distance, so
1.0 means identical meaning and 0.0 means unrelated.
"""

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "books"
MAX_BATCH = 5000  # Chroma refuses very large single inserts


class VectorStore:
    def __init__(self, folder: str):
        self.client = chromadb.PersistentClient(
            path=folder, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._open_collection()

    def _open_collection(self):
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, ids, texts, embeddings, metadatas):
        for i in range(0, len(ids), MAX_BATCH):
            self.collection.add(
                ids=ids[i : i + MAX_BATCH],
                documents=texts[i : i + MAX_BATCH],
                embeddings=embeddings[i : i + MAX_BATCH],
                metadatas=metadatas[i : i + MAX_BATCH],
            )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        # Chroma returns lists-of-lists (one list per query). We sent 1 query -> take [0].
        hits = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                {
                    "file": meta["file"],
                    "page": meta["page"],
                    "chunk": meta["chunk"],
                    "similarity": round(1 - distance, 4),
                    "text": text,
                }
            )
        return hits

    def has_file(self, file_name: str) -> bool:
        found = self.collection.get(where={"file": file_name}, limit=1)
        return len(found["ids"]) > 0

    def count_chunks(self, file_name: str | None = None) -> int:
        if file_name is None:
            return self.collection.count()
        return len(self.collection.get(where={"file": file_name}, include=[])["ids"])

    def delete_file(self, file_name: str):
        self.collection.delete(where={"file": file_name})

    def delete_everything(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self._open_collection()
