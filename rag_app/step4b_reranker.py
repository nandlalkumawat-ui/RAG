"""STEP 4b: Rerank the chunks found by the vector DB with a cross-encoder.

Step 4 (ChromaDB) is fast but rough: the question and each chunk were turned
into vectors SEPARATELY, and we only compare the two vectors. That is how it
can search thousands of chunks in milliseconds.

A cross-encoder reads the question and ONE chunk TOGETHER, word by word, and
outputs a relevance score (here 0..1, higher = more relevant). That is much
more accurate, but also much slower, so we only run it on the few candidates
step 4 already found:

    all chunks --(step 4, fast)--> 20 candidates --(step 4b, accurate)--> best 5

The vector "similarity" of each chunk is kept as it is; the cross-encoder
score is added next to it as "rerank_score".
"""

import logging

import sentence_transformers

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        logger.info("Loading reranker model %s ...", model_name)
        # max_length 512 = the most bge-reranker can read (question + chunk together)
        self.model = sentence_transformers.CrossEncoder(model_name, max_length=512, device="cpu")

    def rerank(self, query: str, documents: list[dict], top_n: int) -> list[dict]:
        """Score every (query, document["text"]) pair and return the top_n best.

        Each document is a hit from step 4 (text, citation, similarity, ...).
        The returned dicts are copies with an extra "rerank_score", sorted
        best first. The input dicts are not changed.
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        # A chunk without text cannot answer anything, so it is not worth scoring
        documents = [doc for doc in documents if doc.get("text", "").strip()]
        if not documents:
            return []

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        scored = [{**doc, "rerank_score": round(float(score), 4)}
                  for doc, score in zip(documents, scores)]
        scored.sort(key=lambda doc: doc["rerank_score"], reverse=True)
        return scored[:top_n]
