"""Reads config.yaml into a normal Python dictionary."""

from pathlib import Path # Path is a Python class used to work with file and folder paths

import yaml

# The project folder (the folder that contains config.yaml)
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Settings that hold a folder or file path
PATH_SETTINGS = ["books_folder", "chroma_folder", "history_db_file"]


def load_config(config_file: Path = PROJECT_DIR / "config.yaml") -> dict:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Turn relative paths like "books" into full paths like /home/.../RAG/books,
    # so the app works no matter which folder you start it from.
    for key in PATH_SETTINGS:
        path = Path(config[key])
        if not path.is_absolute():
            path = PROJECT_DIR / path
        config[key] = str(path)

    if config["chunk_overlap"] >= config["chunk_size"]:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return config
