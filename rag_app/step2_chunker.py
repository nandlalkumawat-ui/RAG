"""STEP 2: Cut long text into small, overlapping chunks.

Why chunk at all?
  * The embedding model can only read 256 tokens at a time.
  * A small chunk is about one topic, so its vector matches questions well.
  * Only the best few chunks are sent to the LLM, which keeps the prompt short.

Why overlap?
  If a sentence is cut in half at a chunk border, the overlap makes sure the
  whole sentence is still inside at least one chunk.

Example with chunk_size=4 and overlap=1 (each letter is one token):
    tokens:  A B C D E F G H I J
    chunk 1: A B C D
    chunk 2:       D E F G         <- starts 3 tokens later (4 - 1)
    chunk 3:             G H I J
"""

from transformers import AutoTokenizer


def load_tokenizer(model_name: str):
    """Use the SAME tokenizer as the embedding model, so our token counts are exact.

    model_max_length is raised only to silence the "sequence is too long"
    warning. We never feed these long token lists to the model, we just count them.
    """
    return AutoTokenizer.from_pretrained(model_name, model_max_length=10**9)


def make_windows(total_tokens: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) token positions for every chunk. Pure math, easy to test."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    windows = []
    step = chunk_size - overlap  # how far we move forward each time
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        windows.append((start, end))
        if end == total_tokens:  # reached the end of the text
            break
        start += step
    return windows


def split_into_chunks(text: str, tokenizer, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` tokens.

    The tokenizer tells us, for every token, where it starts and ends in the
    original text ("offset_mapping"). We use that to cut the ORIGINAL text,
    so chunks keep their capital letters, punctuation and spacing.

    Note: if a cut lands in the middle of a word, re-tokenizing the chunk can
    give 1 extra token (in book_1.pdf: 1 chunk of 443 had 251 tokens). That is
    still below the model's 256 token limit, so nothing gets cut off.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]  # e.g. [(0, 5), (6, 11), ...]

    chunks = []
    for start, end in make_windows(len(offsets), chunk_size, overlap):
        first_char = offsets[start][0]
        last_char = offsets[end - 1][1]
        chunk = text[first_char:last_char].strip()
        if chunk:
            chunks.append(chunk)
    return chunks
