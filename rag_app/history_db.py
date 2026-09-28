"""Save every question and answer in a small SQLite database (data/history.db).

SQLite is built into Python (the `sqlite3` module), so nothing extra to install.
The whole database is a single file. You can open it with "DB Browser for SQLite".

Table: qa_history
    id | question | answer | sources (JSON text) | top_k | created_at
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


class HistoryDB:
    def __init__(self, db_file: str):
        self.db_file = db_file
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    question   TEXT NOT NULL,
                    answer     TEXT NOT NULL,
                    sources    TEXT NOT NULL,
                    top_k      INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self):
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row  # lets us read rows like dicts: row["question"]
        return conn

    @staticmethod
    def _row_to_dict(row) -> dict:
        item = dict(row)
        item["sources"] = json.loads(item["sources"])
        return item

    def save(self, question: str, answer: str, sources: list[dict], top_k: int) -> int:
        created_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO qa_history (question, answer, sources, top_k, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, answer, json.dumps(sources), top_k, created_at),
            )
            return cursor.lastrowid

    def get_all(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_one(self, qa_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM qa_history WHERE id = ?", (qa_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def delete_one(self, qa_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM qa_history WHERE id = ?", (qa_id,))
            return cursor.rowcount > 0

    def delete_all(self) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM qa_history").rowcount
