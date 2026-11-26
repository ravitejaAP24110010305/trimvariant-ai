"""SQLite persistence (stdlib sqlite3, no ORM needed)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    view          TEXT NOT NULL,
    original_path TEXT NOT NULL,
    prepared_path TEXT NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    orig_width    INTEGER NOT NULL,
    orig_height   INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS variants (
    id              TEXT PRIMARY KEY,
    image_id        TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    feature         TEXT NOT NULL,
    option          TEXT,
    mode            TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT,
    params          TEXT,
    seed            INTEGER,
    compare_group   TEXT,
    status          TEXT NOT NULL,
    error           TEXT,
    output_path     TEXT,
    raw_path        TEXT,
    mask_path       TEXT,
    control_path    TEXT,
    metrics         TEXT,
    qc              TEXT,
    review_decision TEXT NOT NULL DEFAULT 'pending',
    review_note     TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    duration_s      REAL
);

CREATE INDEX IF NOT EXISTS idx_variants_image ON variants(image_id);
CREATE INDEX IF NOT EXISTS idx_variants_group ON variants(compare_group);
"""

JSON_COLUMNS = {"params", "metrics", "qc"}
VARIANT_COLUMNS = {
    "feature", "option", "mode", "prompt", "negative_prompt", "params", "seed",
    "compare_group", "status", "error", "output_path", "raw_path", "mask_path",
    "control_path", "metrics", "qc", "review_decision", "review_note",
    "started_at", "finished_at", "duration_s",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    def fail_interrupted_jobs(self) -> int:
        """Jobs left queued/running by a previous process can never finish."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE variants SET status='failed', error='Interrupted by server restart', "
                "finished_at=? WHERE status IN ('queued','running')",
                (utcnow(),),
            )
            return cur.rowcount

    # ---------------------------------------------------------------- images
    def insert_image(self, **f: Any) -> dict:
        image_id = new_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO images (id, filename, view, original_path, prepared_path, width, height, "
                "orig_width, orig_height, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (image_id, f["filename"], f["view"], str(f["original_path"]), str(f["prepared_path"]),
                 f["width"], f["height"], f["orig_width"], f["orig_height"], utcnow()),
            )
        return self.get_image(image_id)  # type: ignore[return-value]

    def get_image(self, image_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return dict(row) if row else None

    def list_images(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM images ORDER BY created_at DESC, rowid DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_image(self, image_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM images WHERE id=?", (image_id,)).rowcount > 0

    # -------------------------------------------------------------- variants
    def insert_variant(self, image_id: str, feature: str, option: str | None, mode: str, prompt: str,
                       negative_prompt: str, params: dict, seed: int, compare_group: str | None = None) -> str:
        variant_id = new_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO variants (id, image_id, feature, option, mode, prompt, negative_prompt, params, "
                "seed, compare_group, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (variant_id, image_id, feature, option, mode, prompt, negative_prompt,
                 json.dumps(params), seed, compare_group, "queued", utcnow()),
            )
        return variant_id

    def update_variant(self, variant_id: str, **fields: Any) -> None:
        unknown = set(fields) - VARIANT_COLUMNS
        if unknown:
            raise ValueError(f"Unknown variant columns: {sorted(unknown)}")
        if not fields:
            return
        values = [json.dumps(v) if k in JSON_COLUMNS and v is not None else v for k, v in fields.items()]
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE variants SET {assignments} WHERE id=?", (*values, variant_id))

    @staticmethod
    def _variant_from_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        for col in JSON_COLUMNS:
            if data.get(col):
                data[col] = json.loads(data[col])
        return data

    def get_variant(self, variant_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM variants WHERE id=?", (variant_id,)).fetchone()
        return self._variant_from_row(row) if row else None

    def list_variants(self, image_id: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM variants", ()
        if image_id:
            query += " WHERE image_id=?"
            args = (image_id,)
        query += " ORDER BY created_at DESC, rowid DESC"
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [self._variant_from_row(r) for r in rows]

    def delete_variant(self, variant_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM variants WHERE id=?", (variant_id,)).rowcount > 0
