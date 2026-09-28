"""STEP 3: Turn text into embeddings (vectors = lists of numbers).

An embedding model reads a sentence and outputs 384 numbers that describe its
MEANING. Texts with similar meaning get similar numbers, even if they use
different words:
    "How does a robot decide what to do?"   and
    "An agent chooses an action..."          -> vectors close together
    "Recipe for chocolate cake"              -> vector far away

all-MiniLM-L6-v2 is small (~90 MB), free, and runs on a normal CPU.
"""

import logging

import sentence_transformers

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str):
        logger.info("Loading embedding model %s ...", model_name)
        self.model = sentence_transformers.SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # length 1 vectors, so cosine similarity works well
            show_progress_bar=show_progress,
        )
        return vectors.tolist()
