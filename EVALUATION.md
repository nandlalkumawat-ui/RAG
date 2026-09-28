# Evaluation

**Setup:** `books/book_1.pdf` (an AI textbook, 187 pages, 443 chunks), `top_k = 3`,
`chunk_size = 250`, `chunk_overlap = 50`, embeddings `all-MiniLM-L6-v2`, LLM `llama3.2`
through Ollama on CPU. Run `python evaluate.py` to reproduce. The full answers and
retrieved chunks are in [evaluation_results.md](evaluation_results.md).

## How each metric was measured

| Metric | How |
|---|---|
| **Retrieval precision@3** | Automatic. Each question has a known page range (from the book's table of contents). Precision = retrieved chunks inside that range / 3. |
| **Answer relevance (1-5)** | By hand. 5 = correct and complete, 3 = partly correct or muddled, 1 = wrong or off topic. |
| **Citation accuracy** | By hand. I opened every cited chunk and checked that it really contains the fact the answer credits to it. |

## Results

| # | Question | Retrieval precision | Answer relevance | Citation accuracy |
|---|---|---|---|---|
| 1 | Difference between BFS and DFS? | 3/3 | 4 | Mostly. The final sentence ("explores as far as possible along each branch...") is correct but comes from the LLM's own knowledge, not from the chunks. |
| 2 | What makes a heuristic admissible? | 3/3 | 5 | Yes. Chunks [1] and [3] both contain `0 ≤ h(n) ≤ h*(n)`. |
| 3 | How does alpha-beta pruning work? | 3/3 | 3 | Yes, but the answer's explanation of *when* to prune is muddled. |
| 4 | What is the Bellman equation? | 3/3 | 5 | Yes. The formula appears verbatim in [1]. |
| 5 | Exploration vs exploitation trade-off? | 3/3 | 3 | **Partly.** [1] is the right chunk, but the LLM restated it wrongly ("more frequent but less informed actions"). The book says a large ε keeps the agent acting randomly, and a small ε makes learning slow. |
| 6 | How does particle filtering work? | 3/3 | 5 | Partly. The two update steps come from [3] (page 141), but the answer credits [1] and [2]. |
| 7 | What is d-separation? | 3/3 | 5 | Yes (page 105). |
| 8 | Who won the 2018 FIFA World Cup? *(not in the book)* | n/a | 5 | Correct refusal: "I could not find the answer in the books." |

**Average retrieval precision@3: 1.00. Average answer relevance: 4.4 / 5.**

## What this tells us

- **Retrieval is the strong part.** On a well-structured textbook every retrieved chunk
  came from the correct section. In-book questions scored a similarity of about 0.45–0.66.
  The out-of-scope question scored only about 0.16, so a *minimum similarity threshold*
  (for example 0.3) would be a cheap way to refuse out-of-scope questions before calling the LLM.
- **Generation is the weak part.** A small 3B-parameter local model sometimes
  paraphrases the right chunk wrongly (Q5), adds facts from its own memory (Q1), or
  puts the wrong source number on a fact (Q6). A larger model, or a stricter
  prompt asking it to quote the source, would help.
- **Speed:** retrieval takes about 30 ms. The LLM answer takes 40–65 s on this CPU, so almost all
  of the waiting time is the LLM.

## Memory footprint (measured)

| What | Size |
|---|---|
| Vector DB on disk for book_1.pdf (443 chunks) | 5.1 MB (`data/chroma_db`) |
| Raw vectors alone: 443 × 384 numbers × 4 bytes | 0.68 MB (the rest is chunk text + index) |
| Q&A history DB | < 0.1 MB |
| Server RAM after startup (embedding model + Chroma) | ~880 MB (mostly PyTorch) |
| Ollama llama3.2 (separate process) | ~2–3 GB |

**Scaling to 10–50 books:** I inserted 22,000 random vectors (about 50 books of this size)
into a test Chroma collection. Inserting took 10 s, one search took **2.9 ms**, and the disk size was
218 MB (about 10 KB per chunk). Search stays fast. The slow part of adding books is
embedding them: about 30 s per 190-page PDF on CPU, and it only happens once per book,
because `POST /ingest` skips books that are already indexed.
