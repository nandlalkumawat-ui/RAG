"""Tests for the reranker (step 4b) and how the pipeline uses it.

The real cross-encoder is a 1 GB download, so every test here uses a FAKE model
that gives each chunk a fixed score. The embedding model, ChromaDB and the
history DB are faked too, so these tests run in a fraction of a second.

    pytest tests/test_reranker.py -v
"""

import copy

import pytest

from rag_app import rag_pipeline, step4b_reranker
from rag_app.rag_pipeline import RAGPipeline
from rag_app.settings import PROJECT_DIR, load_config
from rag_app.step4b_reranker import Reranker

QUESTION = "What is a rational agent?"

# The vector DB order (by similarity) is A, B, C.
# The cross-encoder knows better: C answers the question, B is about cake.
HIT_A = {"file": "book_1.pdf", "page": 10, "chunk": 0, "similarity": 0.80,
         "text": "Agents act in environments.",
         "citation": "book_1.pdf, page 10, chunk 0"}
HIT_B = {"file": "book_2.pdf", "page": 42, "chunk": 1, "similarity": 0.75,
         "text": "Chocolate cake recipe.",
         "citation": "book_2.pdf, page 42, chunk 1"}
HIT_C = {"file": "book_1.pdf", "page": 15, "chunk": 2, "similarity": 0.70,
         "text": "A rational agent maximises expected utility.",
         "citation": "book_1.pdf, page 15, chunk 2"}

FAKE_SCORES = {HIT_A["text"]: 0.60, HIT_B["text"]: 0.01, HIT_C["text"]: 0.95}


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.calls = []  # every list of pairs predict() was given

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return [FAKE_SCORES.get(text, 0.5) for _question, text in pairs]


class BrokenCrossEncoder:
    def __init__(self, model_name, **kwargs):
        raise OSError(f"{model_name} not found")


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    monkeypatch.setattr(step4b_reranker.sentence_transformers, "CrossEncoder", FakeCrossEncoder)


@pytest.fixture
def reranker(fake_cross_encoder):
    return Reranker("fake/reranker")


def hits():
    """Fresh copies, so one test cannot change another test's data."""
    return copy.deepcopy([HIT_A, HIT_B, HIT_C])


# ------------------------------------------------------------ the Reranker alone

def test_sorts_by_rerank_score(reranker):
    result = reranker.rerank(QUESTION, hits(), top_n=3)
    assert [r["citation"] for r in result] == [HIT_C["citation"], HIT_A["citation"], HIT_B["citation"]]
    assert [r["rerank_score"] for r in result] == [0.95, 0.60, 0.01]


def test_sends_query_document_pairs_to_the_model(reranker):
    reranker.rerank(QUESTION, hits(), top_n=2)
    assert reranker.model.calls == [[(QUESTION, HIT_A["text"]), (QUESTION, HIT_B["text"]),
                                     (QUESTION, HIT_C["text"])]]


def test_top_n_is_respected(reranker):
    result = reranker.rerank(QUESTION, hits(), top_n=2)
    assert [r["citation"] for r in result] == [HIT_C["citation"], HIT_A["citation"]]


def test_fewer_documents_than_top_n(reranker):
    result = reranker.rerank(QUESTION, hits(), top_n=5)
    assert len(result) == 3


def test_empty_input_returns_empty_and_skips_the_model(reranker):
    assert reranker.rerank(QUESTION, [], top_n=5) == []
    assert reranker.model.calls == []


def test_empty_text_chunks_are_skipped(reranker):
    empty = {**HIT_A, "text": "   ", "citation": "empty"}
    result = reranker.rerank(QUESTION, [empty, *hits()], top_n=5)
    assert "empty" not in [r["citation"] for r in result]
    assert len(reranker.model.calls[0]) == 3  # the empty chunk was never scored


def test_metadata_and_citations_stay_with_their_chunk(reranker):
    for r in reranker.rerank(QUESTION, hits(), top_n=3):
        original = next(h for h in [HIT_A, HIT_B, HIT_C] if h["citation"] == r["citation"])
        assert {k: r[k] for k in original} == original  # text, file, page, chunk, similarity


def test_similarity_unchanged_and_input_not_modified(reranker):
    given = hits()
    result = reranker.rerank(QUESTION, given, top_n=3)
    assert given == [HIT_A, HIT_B, HIT_C]  # no rerank_score added to the caller's dicts
    assert {r["citation"]: r["similarity"] for r in result} == {
        HIT_A["citation"]: 0.80, HIT_B["citation"]: 0.75, HIT_C["citation"]: 0.70}


def test_rerank_score_is_a_plain_float(reranker, monkeypatch):
    np = pytest.importorskip("numpy")
    # The real model returns numpy float32, which json.dumps (history DB) cannot save
    monkeypatch.setattr(reranker.model, "predict",
                        lambda pairs, **kw: np.array([0.123456, 0.5, 0.9], dtype=np.float32))
    result = reranker.rerank(QUESTION, hits(), top_n=3)
    assert all(type(r["rerank_score"]) is float for r in result)
    assert result[0]["rerank_score"] == 0.9


def test_top_n_below_one_is_rejected(reranker):
    with pytest.raises(ValueError):
        reranker.rerank(QUESTION, hits(), top_n=0)


# ------------------------------------------------------------ inside the pipeline

class FakeVectorStore:
    """Returns stored hits in similarity order and remembers how many were asked for."""

    def __init__(self, folder):
        self.stored = hits() + [
            {"file": "book_3.pdf", "page": n, "chunk": 0, "similarity": round(0.5 - n / 100, 4),
             "text": f"filler chunk {n}"} for n in range(22)
        ]
        for hit in self.stored:
            hit.pop("citation", None)  # the pipeline adds citations itself
        self.asked_for = []

    def search(self, query_embedding, top_k):
        self.asked_for.append(top_k)
        return copy.deepcopy(self.stored[:top_k])

    def count_chunks(self, file_name=None):
        return len(self.stored)


class FakeEmbedder:
    def __init__(self, model_name):
        pass

    def embed(self, texts, show_progress=False):
        return [[0.0] for _ in texts]


class FakeHistory:
    def __init__(self, db_file):
        self.saved = []

    def save(self, question, answer, sources, top_k):
        self.saved.append({"sources": sources, "top_k": top_k})
        return len(self.saved)


@pytest.fixture
def make_pipeline(monkeypatch, fake_cross_encoder):
    monkeypatch.setattr(rag_pipeline, "load_tokenizer", lambda name: None)
    monkeypatch.setattr(rag_pipeline, "Embedder", FakeEmbedder)
    monkeypatch.setattr(rag_pipeline, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(rag_pipeline, "HistoryDB", FakeHistory)
    monkeypatch.setattr(rag_pipeline, "ask_llm", lambda prompt, config: "FAKE ANSWER [1]")

    def make(enabled=True, top_k=3, retrieval_top_k=20, rerank_top_n=2):
        config = load_config()
        config.update(top_k=top_k, retrieval_top_k=retrieval_top_k, rerank_top_n=rerank_top_n,
                      reranker={"enabled": enabled, "model": "fake/reranker"})
        return RAGPipeline(config)
    return make


def test_pipeline_retrieves_many_then_keeps_few(make_pipeline):
    rag = make_pipeline(retrieval_top_k=20, rerank_top_n=2)
    result = rag.search(QUESTION)
    assert rag.vector_store.asked_for == [20]  # 20 candidates, NOT 2
    assert len(rag.reranker.model.calls[0]) == 20  # all 20 were scored
    assert [r["citation"] for r in result] == [HIT_C["citation"], HIT_A["citation"]]


def test_pipeline_citations_and_similarity_survive_reranking(make_pipeline):
    result = make_pipeline().search(QUESTION)
    assert result[0]["citation"] == "book_1.pdf, page 15, chunk 2"
    assert result[0]["page"] == 15 and result[0]["similarity"] == 0.70
    assert result[0]["rerank_score"] == 0.95


def test_pipeline_disabled_behaves_like_before(make_pipeline):
    rag = make_pipeline(enabled=False, top_k=3)
    result = rag.search(QUESTION)
    assert rag.reranker is None
    assert rag.vector_store.asked_for == [3]  # top_k, not retrieval_top_k
    assert [r["citation"] for r in result] == [HIT_A["citation"], HIT_B["citation"], HIT_C["citation"]]
    assert all("rerank_score" not in r for r in result)


def test_pipeline_top_n_bigger_than_retrieval_top_k(make_pipeline):
    rag = make_pipeline(retrieval_top_k=2, rerank_top_n=4)
    result = rag.search(QUESTION)
    assert rag.vector_store.asked_for == [4]  # fetch at least as many as we return
    assert len(result) == 4


def test_pipeline_request_top_k_overrides_rerank_top_n(make_pipeline):
    rag = make_pipeline(rerank_top_n=2)
    assert len(rag.search(QUESTION, top_k=3)) == 3
    assert rag.vector_store.asked_for == [20]


def test_pipeline_empty_retrieval(make_pipeline):
    rag = make_pipeline()
    rag.vector_store.search = lambda vector, top_k: []
    assert rag.search(QUESTION) == []
    assert rag.reranker.model.calls == []


def test_ask_gives_reranked_sources_to_llm_and_history(make_pipeline):
    rag = make_pipeline(rerank_top_n=2)
    result = rag.ask(QUESTION)
    assert [s["citation"] for s in result["sources"]] == [HIT_C["citation"], HIT_A["citation"]]
    assert [s["source_number"] for s in result["sources"]] == [1, 2]
    assert rag.history.saved[0]["top_k"] == 2


def test_model_load_error_falls_back_to_vector_search(make_pipeline, monkeypatch):
    monkeypatch.setattr(step4b_reranker.sentence_transformers, "CrossEncoder", BrokenCrossEncoder)
    rag = make_pipeline(enabled=True, top_k=3)
    assert rag.reranker is None
    assert len(rag.search(QUESTION)) == 3
    assert rag.vector_store.asked_for == [3]


def test_reranking_error_falls_back_to_vector_order(make_pipeline, monkeypatch):
    rag = make_pipeline(rerank_top_n=2)

    def broken_predict(pairs, **kwargs):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(rag.reranker.model, "predict", broken_predict)
    result = rag.search(QUESTION)
    assert [r["citation"] for r in result] == [HIT_A["citation"], HIT_B["citation"]]


# ------------------------------------------------------------------- config

def write_config(tmp_path, extra: str):
    text = (PROJECT_DIR / "config.yaml").read_text()
    config_file = tmp_path / "config.yaml"
    config_file.write_text(text + "\n" + extra)
    return config_file


def test_config_without_reranker_section_means_disabled(tmp_path):
    original = (PROJECT_DIR / "config.yaml").read_text()
    start = original.index("# ---- Reranking")
    end = original.index("# ---- Embedding model")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(original[:start] + original[end:])
    assert load_config(config_file)["reranker"] == {"enabled": False}


@pytest.mark.parametrize("extra", ["rerank_top_n: 0", "retrieval_top_k: -5",
                                   "rerank_top_n: five", "top_k: 0"])
def test_invalid_numbers_are_rejected(tmp_path, extra):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, extra))


def test_enabled_reranker_needs_a_model(tmp_path):
    with pytest.raises(ValueError, match="reranker.model"):
        load_config(write_config(tmp_path, "reranker:\n  enabled: true\n  model: ''"))
