# Book Q&A with RAG

Ask questions about your books (PDF/TXT) and get answers **with citations**
(file + page + chunk). Every question and answer is saved in a database.
Everything runs **locally and for free**: no API key needed.

```
                 INGEST (once per book)                          ASK (every question)
 books/*.pdf ──► read pages ──► chunks ──► embeddings ──►  ┌────────────┐
                 (step 1)      (step 2)    (step 3)        │  ChromaDB  │ (step 4)
                                                           └─────┬──────┘
 question ──► embedding ──► find top 3 closest chunks ◄──────────┘
                                   │
                                   ▼
                     LLM (llama3.2 via Ollama) writes an answer    (step 5)
                     using ONLY those chunks, citing [1] [2] [3]
                                   │
                                   ▼
                     saved in SQLite (data/history.db)
```

| Part | Tool |
|---|---|
| API + Swagger UI | FastAPI |
| Reading PDFs | pypdf |
| Chunking | tokenizer of the embedding model (token-exact) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (saved on disk, no server needed) |
| LLM | `llama3.2` through [Ollama](https://ollama.com) (local, open-source) |
| Q&A history | SQLite (built into Python) |

## 1. Setup (one time)

```bash
cd RAG
pip install -r requirements.txt

# Install Ollama from https://ollama.com, then download the model (~2 GB):
ollama pull llama3.2
```

## 2. Run

```bash
uvicorn rag_app.main:app --reload
```

Open **http://127.0.0.1:8000/docs**. That is the Swagger UI.
In Swagger: click an endpoint → **Try it out** → fill in the fields → **Execute**.

1. **`POST /ingest`** → Execute. This chunks and embeds every book in `books/` (~30 s for book_1.pdf).
   You only need to do this once. The vectors are saved in `data/chroma_db`.
2. **`POST /ask`** with body:
   ```json
   { "question": "What is a rational agent?" }
   ```
   The answer comes back with its sources:
   ```json
   {
     "id": 1,
     "question": "What is a rational agent?",
     "answer": "A rational agent is an entity that has goals or preferences and tries to perform a series of actions that yield the best/optimal expected outcome given these goals [1].",
     "sources": [
       { "source_number": 1, "citation": "book_1.pdf, page 3, chunk 0", "similarity": 0.576, "text": "..." },
       ...
     ]
   }
   ```
   The first question is slow (~60 s) because Ollama loads the model into memory.
3. **`GET /history`** shows every question you asked and its answer.

### All endpoints

| Endpoint | What it does |
|---|---|
| `GET /books` | List books and how many chunks each has in the vector DB |
| `POST /books/upload` | Upload a new PDF/TXT; it is indexed right away |
| `DELETE /books/{file_name}` | Delete a book and its chunks |
| `POST /ingest` | Index all *new* books in `books/`. `rebuild=true` re-indexes everything |
| `POST /ask` | Full RAG: retrieve chunks, LLM answers with citations, saved in history |
| `POST /search` | Retrieval only (no LLM). Useful to see *what* the LLM would be given |
| `GET /history` | Past Q&A (newest first) |
| `GET /history/{id}` / `DELETE /history/{id}` / `DELETE /history` | Read or delete history |

**Adding books:** drop PDF/TXT files into `books/` and call `POST /ingest`,
or use `POST /books/upload`. Only new books are processed.

## 3. Settings: `config.yaml`

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | 250 | Max tokens per chunk |
| `chunk_overlap` | 50 | Tokens shared by neighbouring chunks |
| `top_k` | 3 | Chunks retrieved per question when the reranker is off (can also be set per request) |
| `retrieval_top_k` | 20 | Reranker on: candidate chunks fetched from ChromaDB |
| `rerank_top_n` | 5 | Reranker on: best chunks kept after reranking and given to the LLM (per-request `top_k` overrides it) |
| `reranker.enabled` | true | `false` = plain vector search, exactly as before |
| `reranker.model` | BAAI/bge-reranker-base | Cross-encoder that scores (question, chunk) pairs |
| `embedding_model` | all-MiniLM-L6-v2 | Model that turns text into vectors |
| `llm_model` | llama3.2 | Any model you have in Ollama (`ollama list`) |
| `temperature` | 0.1 | Lower = more factual |
| `max_answer_tokens` | 400 | Longest allowed answer |

After changing `chunk_size`, `chunk_overlap` or `embedding_model`, run
`POST /ingest?rebuild=true`. The old vectors were made with the old settings.

## 4. Code: read in this order

```
rag_app/
  settings.py            reads config.yaml
  step1_loader.py        PDF/TXT -> text, page by page
  step2_chunker.py       text -> overlapping chunks
  step3_embedder.py      chunk -> vector of 384 numbers
  step4_vector_store.py  store + search vectors in ChromaDB
  step4b_reranker.py     cross-encoder picks the best of the retrieved chunks
  step5_llm.py           prompt + call Ollama
  history_db.py          SQLite table for questions/answers
  rag_pipeline.py        connects steps 1-5 (ingest / search / ask)
  main.py                FastAPI endpoints (Swagger)
tests/                   unit + integration tests
evaluate.py              runs 8 test questions and measures quality
```

Logs in the terminal show each step: books loaded, chunks created, and the retrieval
similarity scores (plus rerank scores, when the reranker is on) for every question.

## 5. Tests and evaluation

```bash
pytest -v            # 28 tests, ~20 s. Uses tests/sample_docs, never touches your real data
python evaluate.py   # 8 questions on book_1.pdf, ~8 min (LLM on CPU)
```

- `tests/test_chunker.py`: chunk size limits, exact overlap, bad settings.
- `tests/test_history_db.py`: save, read and delete history.
- `tests/test_integration.py`: end-to-end through the API. Covers ingest, search, ask with
  citations, history, error cases (empty question, Ollama down → 503, wrong file
  type, broken PDF), plus one test with the real LLM (skipped if Ollama is off).

Results are in **[EVALUATION.md](EVALUATION.md)**: retrieval precision@3 = 1.00,
answer relevance 4.4/5, a citation check for every answer, and the measured memory footprint.

## 6. What to learn from this project

**How embeddings capture meaning.** The embedding model turns text into 384 numbers.
It was trained so that texts with similar *meaning* land close together. "How do agents
pick actions?" finds the chunk about "rational agents... perform a series of actions"
even though the words differ. Closeness is measured with cosine similarity: 1 means
the same meaning, 0 means unrelated. Try `POST /search` and look at the `similarity` values.

**Why the chunking strategy matters.** This is a trade-off.
- *Big chunks* carry more context, but each one mixes several topics. Its vector becomes
  a blurry average, so retrieval gets less precise. Also, all-MiniLM only reads **256
  tokens**, so anything longer is silently cut off and never searchable.
  That is why `chunk_size` is 250, not 512.
- *Small chunks* match precisely, but a single chunk may not hold the whole answer.
- *Overlap* (50 tokens) makes sure a sentence split at a border still exists whole in one chunk.
- We chunk page by page, so every citation can point to a page.

**RAG vs. fine-tuning.**
- *Fine-tuning* changes the model's weights by training on your data. It is expensive and slow
  to update, it cannot tell you *where* a fact came from, and the model can still misremember.
- *RAG* leaves the model unchanged and hands it the relevant text at question time.
  Adding a book only takes an ingest, answers come with citations you can check,
  and you can delete data at any time. Fine-tuning is better for teaching a *style* or
  *format*. RAG is better for *knowledge*.

**Common failure modes** (and ones seen in EVALUATION.md):
- *Irrelevant retrieval*: the right chunk is not in the top k, so the LLM cannot answer.
  Check with `/search`. Fixes: different chunk size, larger `top_k`, a better embedding model.
- *Hallucination*: the LLM adds facts that are not in the chunks (Q1), or restates a chunk
  wrongly (Q5). The prompt in `step5_llm.py` forbids outside knowledge, but small models
  don't always obey.
- *Wrong citation*: a fact is credited to the wrong source number (Q6).
- *Out-of-scope questions*: the system still retrieves the "closest" chunks even when
  nothing is relevant (similarity about 0.16). The prompt makes the LLM say "I could not find the answer."
- *Bad PDF text*: scanned PDFs have no text layer (they are skipped with a warning),
  and some PDFs lose spaces between words ("rationalagent").

**How to evaluate RAG.** Evaluate the two halves separately:
1. *Retrieval*: are the retrieved chunks relevant? (precision@k, measured automatically
   in `evaluate.py` using known page ranges)
2. *Generation*: does the answer address the question (relevance 1-5), and is every cited
   source really supporting the claim (citation accuracy)? A human checked these.
If retrieval is bad, no LLM can fix it. If retrieval is good but answers are bad, improve the
prompt or the LLM.
