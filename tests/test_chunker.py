"""Unit tests for step 2 (chunking).  Run:  pytest tests/test_chunker.py -v"""

import pytest

from rag_app.step2_chunker import load_tokenizer, make_windows, split_into_chunks

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL)


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------- make_windows: the pure math ----------------
def test_windows_example_from_docstring():
    # 10 tokens, chunk_size 4, overlap 1  ->  A-D, D-G, G-J
    assert make_windows(10, chunk_size=4, overlap=1) == [(0, 4), (3, 7), (6, 10)]


def test_windows_never_bigger_than_chunk_size():
    for start, end in make_windows(1000, chunk_size=250, overlap=50):
        assert end - start <= 250


def test_windows_overlap_exactly():
    windows = make_windows(1000, chunk_size=250, overlap=50)
    for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
        assert previous_end - next_start == 50


def test_windows_cover_every_token():
    windows = make_windows(1000, chunk_size=250, overlap=50)
    assert windows[0][0] == 0
    assert windows[-1][1] == 1000


def test_short_text_gives_one_window():
    assert make_windows(10, chunk_size=250, overlap=50) == [(0, 10)]


def test_empty_text_gives_no_windows():
    assert make_windows(0, chunk_size=250, overlap=50) == []


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_bad_settings_raise_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        make_windows(1000, chunk_size, overlap)


# ---------------- split_into_chunks: real text + real tokenizer ----------------
LONG_TEXT = " ".join(
    f"Sentence number {i} talks about search, agents, and Bayes nets." for i in range(300)
)


def test_chunks_respect_token_limit(tokenizer):
    chunks = split_into_chunks(LONG_TEXT, tokenizer, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(tokenizer, chunk) <= 100


def test_neighbouring_chunks_share_text(tokenizer):
    chunks = split_into_chunks(LONG_TEXT, tokenizer, chunk_size=100, overlap=20)
    for previous, following in zip(chunks, chunks[1:]):
        start_of_next = following[:30]
        assert start_of_next in previous  # the overlap is repeated at the start of the next chunk


def test_chunks_keep_original_text(tokenizer):
    # The tokenizer lowercases, but our chunks are cut from the original text.
    chunks = split_into_chunks("Alpha-Beta Pruning in CHESS.", tokenizer, chunk_size=50, overlap=5)
    assert chunks == ["Alpha-Beta Pruning in CHESS."]


def test_empty_text_gives_no_chunks(tokenizer):
    assert split_into_chunks("   ", tokenizer, chunk_size=50, overlap=5) == []
