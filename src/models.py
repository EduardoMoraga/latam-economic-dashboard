"""
ML models module — forecasting and feature-importance analysis.

Models included:
* **Linear Regression** — baseline for each indicator time series
* **Random Forest**     — GDP growth prediction from other indicators
* **Exponential Smoothing** — Holt-Winters for univariate time series forecasting

All models use proper time-series-aware cross-validation (``TimeSeriesSplit``).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils import COUNTRIES, INDICATOR_SHORT, INDICATORS

logger = logging.getLogger(__name__)

# Suppress convergence warnings from statsmodels during batch runs
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# ── Evaluation helpers ───────────────────────────────────────────────────────


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE, and R-squared for a set of predictions."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }


# ── Linear Regression baseline ───────────────────────────────────────────────


def linear_forecast(
    series: pd.Series,
    forecast_years: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit a simple linear trend and forecast forward.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by integer year.
    forecast_years : int
        Number of years to project beyond the last observed year.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        * DataFrame with columns ``year, predicted_value, confidence_lower, confidence_upper``
        * Dictionary of in-sample metrics (RMSE, MAE, R2).
    """
    s = series.dropna().sort_index()
    if len(s) < 4:
        return pd.DataFrame(), {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    X = s.index.values.reshape(-1, 1)
    y = s.values

    model = LinearRegression()
    model.fit(X, y)

    y_pred_in = model.predict(X)
    metrics = _compute_metrics(y, y_pred_in)

    # Forecast
    last_year = int(s.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + forecast_years).reshape(-1, 1)
    y_forecast = model.predict(future_years)

    # Simple confidence interval based on in-sample residual std
    residual_std = float(np.std(y - y_pred_in))
    fc_df = pd.DataFrame(
        {
            "year": future_years.ravel(),
            "predicted_value": y_forecast,
            "confidence_lower": y_forecast - 1.96 * residual_std,
            "confidence_upper": y_forecast + 1.96 * residual_std,
        }
    )
    return fc_df, metrics


# ── Exponential Smoothing ────────────────────────────────────────────────────


def exponential_smoothing_forecast(
    series: pd.Series,
    forecast_years: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit a Holt (double exponential smoothing) model and forecast forward.

    Falls back to simple exponential smoothing if the series is too short
    for trend estimation.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by integer year.
    forecast_years : int
        Projection horizon.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Forecast DataFrame and in-sample metrics.
    """
    s = series.dropna().sort_index()
    if len(s) < 4:
        return pd.DataFrame(), {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    try:
        model = ExponentialSmoothing(
            s.values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=True)
    except Exception:
        # Fallback: no trend
        try:
            model = ExponentialSmoothing(
                s.values,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
        except Exception as exc:
            logger.warning("ExponentialSmoothing failed: %s", exc)
            return pd.DataFrame(), {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    y_pred_in = fit.fittedvalues
    metrics = _compute_metrics(s.values, y_pred_in)

    forecast = fit.forecast(forecast_years)
    last_year = int(s.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + forecast_years)

    residual_std = float(np.std(s.values - y_pred_in))
    fc_df = pd.DataFrame(
        {
            "year": future_years,
            "predicted_value": forecast,
            "confidence_lower": forecast - 1.96 * residual_std,
            "confidence_upper": forecast + 1.96 * residual_std,
        }
    )
    return fc_df, metrics


# ── Random Forest — GDP Growth Predictor ─────────────────────────────────────


def train_gdp_growth_rf(
    df: pd.DataFrame,
    target_col: str = "GDP Growth",
    n_splits: int = 3,
) -> Tuple[RandomForestRegressor, Dict[str, float], pd.Series]:
    """
    Train a Random Forest to predict GDP growth from other indicators.

    Uses ``TimeSeriesSplit`` cross-validation so that future data never leaks
    into the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format transformed DataFrame.
    target_col : str
        Column name for the target variable.
    n_splits : int
        Number of time-series cross-validation folds.

    Returns
    -------
    tuple[RandomForestRegressor, dict, pd.Series]
        * Trained model.
        * Averaged CV metrics (RMSE, MAE, R2).
        * Feature importances as a named Series.
    """
    feature_candidates = [
        c
        for c in INDICATOR_SHORT.values()
        if c != target_col and c in df.columns
    ]

    # Also include lag and yoy features
    extra = [c for c in df.columns if ("_lag" in c or "_yoy" in c) and target_col not in c]
    feature_cols = feature_candidates + extra

    sub = df[["country_code", "year", target_col] + [c for c in feature_cols if c in df.columns]].dropna()
    feature_cols = [c for c in feature_cols if c in sub.columns]

    if len(sub) < 10 or not feature_cols:
        logger.warning("Not enough data to train RF (rows=%d, features=%d).", len(sub), len(feature_cols))
        empty_model = RandomForestRegressor(n_estimators=10, random_state=42)
        return (
            empty_model,
            {"rmse": np.nan, "mae": np.nan, "r2": np.nan},
            pd.Series(dtype=float),
        )

    sub = sub.sort_values("year")
    X = sub[feature_cols].values
    y = sub[target_col].values

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(sub) // 5)))
    cv_metrics: List[Dict[str, float]] = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        cv_metrics.append(_compute_metrics(y_test, preds))

    avg_metrics = {
        k: float(np.mean([m[k] for m in cv_metrics]))
        for k in ("rmse", "mae", "r2")
    }

    # Final model on all data
    final_rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    final_rf.fit(X, y)

    importances = pd.Series(final_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    logger.info(
        "RF trained: %d samples, %d features.  CV RMSE=%.3f, R2=%.3f",
        len(sub),
        len(feature_cols),
        avg_metrics["rmse"],
        avg_metrics["r2"],
    )
    return final_rf, avg_metrics, importances


# ── Batch forecasting ────────────────────────────────────────────────────────


def generate_all_forecasts(
    raw_df: pd.DataFrame,
    forecast_years: int = 5,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Generate forecasts for every (country, indicator) combination.

    Uses Exponential Smoothing as the primary model and Linear Regression
    as a fallback / comparison baseline.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Cleaned long-format DataFrame with columns
        ``country_code, country_name, indicator, indicator_name, year, value``.
    forecast_years : int
        Number of years to forecast ahead.
    progress_callback : callable | None
        Optional ``(current, total, message)`` callback.

    Returns
    -------
    pd.DataFrame
        Consolidated forecast table with columns:
        ``country_code, country_name, indicator, indicator_name, year,
        predicted_value, model, confidence_lower, confidence_upper,
        rmse, mae, r2``.
    """
    results: List[Dict] = []
    groups = raw_df.groupby(["country_code", "indicator"])
    total = len(groups)
    current = 0

    for (cc, ind), grp in groups:
        current += 1
        country_name = COUNTRIES.get(cc, cc)
        indicator_name = INDICATORS.get(ind, ind)

        if progress_callback:
            progress_callback(current, total, f"Forecasting {country_name} — {indicator_name}")

        series = grp.set_index("year")["value"].sort_index()

        # Primary: Exponential Smoothing
        es_fc, es_metrics = exponential_smoothing_forecast(series, forecast_years)
        if not es_fc.empty:
            for _, row in es_fc.iterrows():
                results.append(
                    {
                        "country_code": cc,
                        "country_name": country_name,
                        "indicator": ind,
                        "indicator_name": indicator_name,
                        "year": int(row["year"]),
                        "predicted_value": row["predicted_value"],
                        "model": "ExponentialSmoothing",
                        "confidence_lower": row["confidence_lower"],
                        "confidence_upper": row["confidence_upper"],
                        "rmse": es_metrics["rmse"],
                        "mae": es_metrics["mae"],
                        "r2": es_metrics["r2"],
                    }
                )

        # Baseline: Linear Regression
        lr_fc, lr_metrics = linear_forecast(series, forecast_years)
        if not lr_fc.empty:
            for _, row in lr_fc.iterrows():
                results.append(
                    {
                        "country_code": cc,
                        "country_name": country_name,
                        "indicator": ind,
                        "indicator_name": indicator_name,
                        "year": int(row["year"]),
                        "predicted_value": row["predicted_value"],
                        "model": "LinearRegression",
                        "confidence_lower": row["confidence_lower"],
                        "confidence_upper": row["confidence_upper"],
                        "rmse": lr_metrics["rmse"],
                        "mae": lr_metrics["mae"],
                        "r2": lr_metrics["r2"],
                    }
                )

    forecast_df = pd.DataFrame(results)
    logger.info("Generated %d forecast records.", len(forecast_df))
    return forecast_df
