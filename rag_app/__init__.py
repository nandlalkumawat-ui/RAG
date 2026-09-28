"""A simple RAG (Retrieval-Augmented Generation) app for asking questions about books.

Read the files in this order:
    settings.py          -> reads config.yaml
    step1_loader.py      -> reads PDF/TXT books into text
    step2_chunker.py     -> cuts text into small overlapping chunks
    step3_embedder.py    -> turns each chunk into a vector (list of numbers)
    step4_vector_store.py-> saves vectors in ChromaDB and searches them
    step5_llm.py         -> asks the LLM (Ollama) to answer using the chunks
    history_db.py        -> saves every question + answer in SQLite
    rag_pipeline.py      -> connects all the steps together
    main.py              -> the FastAPI web server (Swagger UI at /docs)
"""

import os

# We only use PyTorch. Without this line `transformers` also loads TensorFlow
# (if it is installed), which costs ~500 MB of RAM and slows down startup.
os.environ.setdefault("USE_TF", "0")
