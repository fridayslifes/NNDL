"""
Inference prediction logging (Application Lead requirement).

Every time the model scores a customer, one record is appended to

* ``logs/predictions.jsonl`` — JSON Lines: one JSON object per line, and
* ``logs/predictions_log.db`` — an SQLite table ``predictions`` (optional mirror).

Example record::

    {"timestamp": "2026-09-30T12:34:56.789+00:00", "customer_id": "7590-VHVEG",
     "churn_probability": 0.82, "risk_tier": "High", "threshold_used": 0.08,
     "flagged_for_offer": true, "batch_id": "3f2c…", "source": "csv_upload",
     "model_version": "weighted_bce-1a2b3c4d"}

Why JSON Lines?
---------------
It is **append-only** (we never rewrite the file, so a crash can damage at most
the final line), human-readable, and loads in one call:
``pd.read_json("logs/predictions.jsonl", lines=True)``.

Why log predictions at all? (viva Q4)
-------------------------------------
* **Monitoring drift** — if the distribution of predicted probabilities shifts
  over time, the customer base (or the data pipeline) has changed and the model
  may need retraining.
* **Auditability** — we can reconstruct *why* a customer did or did not receive
  an offer: which model version, which threshold, which score.
* **Measuring impact** — joining logged predictions with later churn outcomes
  gives the real precision/recall and retention rate of the campaign.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

VALID_BACKENDS = {"jsonl", "sqlite", "both"}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    customer_id       TEXT    NOT NULL,
    churn_probability REAL    NOT NULL,
    risk_tier         TEXT    NOT NULL,
    threshold_used    REAL    NOT NULL,
    flagged_for_offer INTEGER NOT NULL,
    batch_id          TEXT,
    source            TEXT,
    model_version     TEXT
)
"""
_COLUMNS = ["timestamp", "customer_id", "churn_probability", "risk_tier", "threshold_used",
            "flagged_for_offer", "batch_id", "source", "model_version"]


def log_clear_enabled() -> bool:
    """
    Whether ``DELETE /api/logs`` (the dashboard's "Clear log" button) is allowed.

    Enabled by default for local demos. Set the environment variable
    ``CHURN_ALLOW_LOG_CLEAR=0`` on a public deployment so that visitors cannot
    wipe the audit trail.
    """
    return os.environ.get("CHURN_ALLOW_LOG_CLEAR", "1").strip().lower() not in {"0", "false", "no", "off"}


class PredictionLoggingError(RuntimeError):
    """Raised when the primary log (JSONL) cannot be written."""


class PredictionLogger:
    """
    Thread-safe writer/reader for prediction logs.

    Parameters
    ----------
    log_dir : path
        Directory for the log files (created if missing).
    backend : {"jsonl", "sqlite", "both"}, default "both"
        Where to write. JSONL is the primary store; SQLite is a queryable mirror.
        If SQLite fails (e.g. file locking on a network drive such as Google
        Drive), it is disabled with a warning and JSONL logging continues.
    """

    def __init__(self, log_dir: str | Path, backend: str = "both",
                 jsonl_name: str = "predictions.jsonl", db_name: str = "predictions_log.db") -> None:
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.jsonl_path = self.log_dir / jsonl_name
        self.db_path = self.log_dir / db_name
        self._lock = threading.Lock()   # FastAPI runs sync work in a thread pool: serialise writes
        self._use_jsonl = backend in {"jsonl", "both"}
        self._use_sqlite = backend in {"sqlite", "both"}
        if self._use_sqlite:
            self._init_sqlite()

    # ------------------------------------------------------------------
    def _init_sqlite(self) -> None:
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute(_CREATE_TABLE)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp)")
        except sqlite3.Error as exc:
            logger.warning("SQLite logging disabled (%s); continuing with JSONL only.", exc)
            self._use_sqlite = False
            if not self._use_jsonl:   # never end up with no log at all
                self._use_jsonl = True

    @property
    def active_backends(self) -> list[str]:
        """Backends currently being written to."""
        return [name for name, on in (("jsonl", self._use_jsonl), ("sqlite", self._use_sqlite)) if on]

    @staticmethod
    def make_record(
        customer_id: str,
        churn_probability: float,
        risk_tier: str,
        threshold_used: float,
        *,
        batch_id: str | None = None,
        source: str = "csv_upload",
        model_version: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Build one log record (UTC ISO-8601 timestamp with milliseconds)."""
        return {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "customer_id": str(customer_id),
            "churn_probability": round(float(churn_probability), 6),
            "risk_tier": str(risk_tier),
            "threshold_used": round(float(threshold_used), 4),
            "flagged_for_offer": bool(float(churn_probability) >= float(threshold_used)),
            "batch_id": batch_id,
            "source": source,
            "model_version": model_version,
        }

    def log(self, records: Iterable[dict[str, Any]]) -> int:
        """
        Append records to every active backend.

        Returns
        -------
        int
            Number of records written.

        Raises
        ------
        PredictionLoggingError
            If the JSONL file cannot be written (disk full, permissions...).
            SQLite errors never raise: that backend is disabled with a warning.
        """
        records = list(records)
        if not records:
            return 0
        with self._lock:
            if self._use_jsonl:
                try:
                    with self.jsonl_path.open("a", encoding="utf-8") as fh:
                        for record in records:
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                except OSError as exc:
                    raise PredictionLoggingError(f"Could not write {self.jsonl_path}: {exc}") from exc
            if self._use_sqlite:
                try:
                    with sqlite3.connect(self.db_path, timeout=10) as conn:
                        conn.executemany(
                            f"INSERT INTO predictions ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                            [tuple(int(r[c]) if c == "flagged_for_offer" else r.get(c) for c in _COLUMNS)
                             for r in records],
                        )
                except sqlite3.Error as exc:
                    logger.warning("SQLite write failed (%s); disabling SQLite mirror.", exc)
                    self._use_sqlite = False
        return len(records)

    def clear(self) -> int:
        """
        Permanently delete every logged prediction from all backends.

        The JSONL file is truncated (kept, but emptied) and the SQLite table is
        emptied with ``DELETE``; the table itself stays so logging continues to
        work. Intended for resetting the demo — in production, logs are an audit
        trail and should be archived rather than deleted.

        Returns
        -------
        int
            Number of records that were removed (counted from the JSONL file,
            or from SQLite when JSONL is not in use).
        """
        removed = 0
        with self._lock:
            if self.jsonl_path.exists():
                with self.jsonl_path.open("r", encoding="utf-8") as fh:
                    removed = sum(1 for line in fh if line.strip())
                try:
                    self.jsonl_path.write_text("", encoding="utf-8")
                except OSError as exc:
                    raise PredictionLoggingError(f"Could not clear {self.jsonl_path}: {exc}") from exc
            if self._use_sqlite:
                try:
                    with sqlite3.connect(self.db_path, timeout=10) as conn:
                        cursor = conn.execute("DELETE FROM predictions")
                        if not self._use_jsonl:
                            removed = cursor.rowcount
                except sqlite3.Error as exc:
                    logger.warning("Could not clear the SQLite mirror (%s).", exc)
        return removed

    # ------------------------------------------------------------------
    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        The ``limit`` most recent records, newest first.

        Reads the tail of the JSONL file with a bounded ``deque`` so memory stays
        O(limit) even for a very large log; malformed lines are skipped.
        """
        limit = max(1, int(limit))
        if self._use_jsonl or not self._use_sqlite:
            if not self.jsonl_path.exists():
                return []
            with self._lock, self.jsonl_path.open("r", encoding="utf-8") as fh:
                tail = deque(fh, maxlen=limit)
            out = []
            for line in reversed(tail):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return out
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(r), "flagged_for_offer": bool(r["flagged_for_offer"])} for r in rows]

    def stats(self) -> dict[str, Any]:
        """Totals for the logs tab: number of predictions, batches and records per risk tier."""
        total, tiers, batches = 0, Counter(), set()
        if self.jsonl_path.exists():
            with self._lock, self.jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    tiers[record.get("risk_tier", "?")] += 1
                    batches.add(record.get("batch_id"))
        return {"total_predictions": total, "batches": len(batches - {None}), "by_tier": dict(tiers),
                "clear_enabled": log_clear_enabled(),
                "backends": self.active_backends, "jsonl_path": str(self.jsonl_path)}
