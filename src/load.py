"""
Load module — persists DataFrames into a local SQLite database.

Three tables are maintained:
* ``raw_indicators`` — long-format extraction data
* ``transformed``    — wide-format with engineered features
* ``forecasts``      — model predictions with confidence intervals

The database file lives at ``data/latam_economic.db`` and is created
automatically on first run.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.utils import DB_PATH

logger = logging.getLogger(__name__)


def _get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection.

    Parameters
    ----------
    db_path : str
        Path to the ``.db`` file.  Parent directory is created if needed.

    Returns
    -------
    sqlite3.Connection
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    """Create the ``pipeline_meta`` table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    conn.commit()


def save_raw(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """
    Write the raw extraction DataFrame to the ``raw_indicators`` table.

    The table is replaced on each run (full refresh).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format raw data.
    db_path : str
        Database file path.

    Returns
    -------
    int
        Number of rows written.
    """
    if df.empty:
        logger.warning("save_raw called with empty DataFrame — skipping.")
        return 0

    conn = _get_connection(db_path)
    try:
        df.to_sql("raw_indicators", conn, if_exists="replace", index=False)
        _update_meta(conn, "raw_last_updated")
        logger.info("Saved %d raw records to '%s'.", len(df), db_path)
        return len(df)
    finally:
        conn.close()


def save_transformed(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """
    Write the transformed wide-format DataFrame to the ``transformed`` table.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format transformed data.
    db_path : str
        Database file path.

    Returns
    -------
    int
        Number of rows written.
    """
    if df.empty:
        logger.warning("save_transformed called with empty DataFrame — skipping.")
        return 0

    conn = _get_connection(db_path)
    try:
        df.to_sql("transformed", conn, if_exists="replace", index=False)
        _update_meta(conn, "transformed_last_updated")
        logger.info("Saved %d transformed records to '%s'.", len(df), db_path)
        return len(df)
    finally:
        conn.close()


def save_forecasts(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """
    Write forecast results to the ``forecasts`` table.

    Parameters
    ----------
    df : pd.DataFrame
        Forecast DataFrame with columns:
        ``country_code, country_name, indicator, indicator_name,
        year, predicted_value, model, confidence_lower, confidence_upper``.
    db_path : str
        Database file path.

    Returns
    -------
    int
        Number of rows written.
    """
    if df.empty:
        logger.warning("save_forecasts called with empty DataFrame — skipping.")
        return 0

    conn = _get_connection(db_path)
    try:
        df.to_sql("forecasts", conn, if_exists="replace", index=False)
        _update_meta(conn, "forecasts_last_updated")
        logger.info("Saved %d forecast records to '%s'.", len(df), db_path)
        return len(df)
    finally:
        conn.close()


def load_table(table: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Read an entire table from the database.

    Parameters
    ----------
    table : str
        Table name (``raw_indicators``, ``transformed``, ``forecasts``).
    db_path : str
        Database file path.

    Returns
    -------
    pd.DataFrame
        Contents of the table, or an empty DataFrame if it does not exist.
    """
    if not os.path.exists(db_path):
        logger.warning("Database '%s' does not exist.", db_path)
        return pd.DataFrame()

    conn = _get_connection(db_path)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if table not in tables:
            logger.warning("Table '%s' not found in database.", table)
            return pd.DataFrame()
        return pd.read_sql(f"SELECT * FROM [{table}]", conn)
    finally:
        conn.close()


def get_meta(key: str, db_path: str = DB_PATH) -> Optional[str]:
    """
    Retrieve a metadata value from ``pipeline_meta``.

    Parameters
    ----------
    key : str
        Metadata key (e.g. ``"raw_last_updated"``).
    db_path : str
        Database file path.

    Returns
    -------
    str | None
        The stored value, or ``None`` if not found.
    """
    if not os.path.exists(db_path):
        return None

    conn = _get_connection(db_path)
    try:
        _ensure_meta_table(conn)
        cur = conn.execute("SELECT value FROM pipeline_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _update_meta(conn: sqlite3.Connection, key: str) -> None:
    """Write a timestamp to ``pipeline_meta`` for the given key."""
    _ensure_meta_table(conn)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_meta (key, value) VALUES (?, ?)",
        (key, now),
    )
    conn.commit()
