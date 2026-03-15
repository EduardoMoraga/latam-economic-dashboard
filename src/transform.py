"""
Transform module — cleans, pivots, and engineers features from raw World Bank data.

Responsibilities:
* Drop records with missing values where appropriate
* Pivot from long to wide format (one row per country-year)
* Compute year-over-year growth rates for each indicator
* Normalize indicators to 0-1 range for cross-country comparison
* Create lag features (t-1, t-2) for ML consumption
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils import INDICATOR_CODES, INDICATOR_SHORT, INDICATORS

logger = logging.getLogger(__name__)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the raw extraction output.

    * Drops rows with ``value=None``.
    * Ensures correct dtypes.
    * Removes duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format DataFrame from ``extract.extract_all()``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (still long format).
    """
    if df.empty:
        return df

    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["year"] = out["year"].astype(int)
    out = out.drop_duplicates(subset=["country_code", "indicator", "year"])
    out = out.sort_values(["country_code", "indicator", "year"]).reset_index(drop=True)

    logger.info("Cleaning: %d -> %d records after dropping nulls/dupes.", len(df), len(out))
    return out


def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long to wide format: one row per (country, year).

    Each indicator becomes its own column using short names for readability.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned long-format DataFrame.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by ``(country_code, country_name, year)``.
    """
    if df.empty:
        return df

    # Map indicator codes to short labels for column names
    df = df.copy()
    df["indicator_short"] = df["indicator"].map(INDICATOR_SHORT)

    pivot = df.pivot_table(
        index=["country_code", "country_name", "year"],
        columns="indicator_short",
        values="value",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.sort_values(["country_code", "year"]).reset_index(drop=True)

    logger.info("Pivoted to wide format: %d rows x %d columns.", len(pivot), len(pivot.columns))
    return pivot


def add_yoy_growth(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add year-over-year percentage growth columns for selected indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame (from ``pivot_wide``).
    columns : list[str] | None
        Columns to compute growth for. Defaults to all numeric indicator columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``<col>_yoy`` columns.
    """
    if df.empty:
        return df

    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = columns or [c for c in numeric_cols if c != "year"]

    for col in target_cols:
        if col not in out.columns:
            continue
        yoy_col = f"{col}_yoy"
        out[yoy_col] = out.groupby("country_code")[col].pct_change() * 100

    logger.info("Added YoY growth for %d columns.", len(target_cols))
    return out


def normalize_indicators(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Min-max normalize indicator columns to [0, 1] for radar/comparison charts.

    Normalization is computed across all countries so that values are
    globally comparable.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame.
    columns : list[str] | None
        Columns to normalize.  Defaults to the main indicator short labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``<col>_norm`` columns.
    """
    if df.empty:
        return df

    out = df.copy()
    short_labels = list(INDICATOR_SHORT.values())
    target_cols = columns or [c for c in short_labels if c in out.columns]

    for col in target_cols:
        cmin = out[col].min()
        cmax = out[col].max()
        norm_col = f"{col}_norm"
        if cmax - cmin == 0:
            out[norm_col] = 0.0
        else:
            out[norm_col] = (out[col] - cmin) / (cmax - cmin)

    logger.info("Normalized %d columns.", len(target_cols))
    return out


def add_lag_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: List[int] = [1, 2],
) -> pd.DataFrame:
    """
    Create lagged versions of indicator columns for ML models.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame sorted by ``(country_code, year)``.
    columns : list[str] | None
        Columns to lag.  Defaults to all short-label indicator columns present.
    lags : list[int]
        Lag periods (in years) to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``<col>_lag<n>`` columns.
    """
    if df.empty:
        return df

    out = df.copy().sort_values(["country_code", "year"])
    short_labels = list(INDICATOR_SHORT.values())
    target_cols = columns or [c for c in short_labels if c in out.columns]

    for col in target_cols:
        if col not in out.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            out[lag_col] = out.groupby("country_code")[col].shift(lag)

    logger.info("Added lag features (lags=%s) for %d columns.", lags, len(target_cols))
    return out


def run_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full transformation pipeline.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw long-format data from extraction.

    Returns
    -------
    pd.DataFrame
        Fully transformed wide-format DataFrame ready for ML and visualization.
    """
    cleaned = clean_raw(raw_df)
    wide = pivot_wide(cleaned)
    wide = add_yoy_growth(wide)
    wide = normalize_indicators(wide)
    wide = add_lag_features(wide)

    logger.info(
        "Transformation complete: %d rows, %d features.",
        len(wide),
        len(wide.columns),
    )
    return wide
