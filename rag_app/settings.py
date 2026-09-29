"""Reads config.yaml into a normal Python dictionary."""

from pathlib import Path # Path is a Python class used to work with file and folder paths

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent # resolve() converts into absolute path , /home/user/my_rag_project/rag_app/settings.py -- parent.parent --> /home/user/my_rag_project

# Settings that holds a folder or file path
PATH_SETTINGS = ["books_folder", "chroma_folder", "history_db_file"]


def load_config(config_file: Path = PROJECT_DIR / "config.yaml") -> dict:  # Load and prepare the configuration and return it as a dictionary
    with open(config_file) as f:   # with statement makes sure Python properly closes the file afterward
        config = yaml.safe_load(f) # safe_load() :- doesn't allow arbitrary Python object construction from the YAML content.


    for key in PATH_SETTINGS:
        path = Path(config[key])
        if not path.is_absolute():  # books is the relavtive path 
            path = PROJECT_DIR / path
        config[key] = str(path)

    if config["chunk_overlap"] >= config["chunk_size"]:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Older config files have no reranker section: behave like before (no reranking)
    reranker = config.get("reranker") or {}
    config["reranker"] = reranker
    reranker["enabled"] = bool(reranker.get("enabled", False))

    positive_ints = ["top_k"]
    if reranker["enabled"]:
        positive_ints += ["retrieval_top_k", "rerank_top_n"]
        if not reranker.get("model"):
            raise ValueError("reranker.model must be set when reranker.enabled is true")
    for key in positive_ints:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a whole number of at least 1, got {value!r}")

    return config
