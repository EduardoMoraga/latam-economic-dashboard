"""
Extract module — pulls economic indicator data from the World Bank API v2.

The World Bank API is free and requires no authentication.  This module
fetches data for six LATAM countries across seven indicators covering GDP,
inflation, trade, employment, and technology adoption (2000-2024).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.utils import (
    COUNTRIES,
    COUNTRY_CODES,
    INDICATOR_CODES,
    INDICATORS,
    WB_API_BASE,
    WB_DATE_RANGE,
    WB_PER_PAGE,
)

logger = logging.getLogger(__name__)


def fetch_indicator(
    country_code: str,
    indicator_code: str,
    date_range: str = WB_DATE_RANGE,
    per_page: int = WB_PER_PAGE,
    max_retries: int = 3,
) -> List[Dict]:
    """
    Fetch a single indicator for a single country from the World Bank API.

    Parameters
    ----------
    country_code : str
        ISO 3-letter country code (e.g. ``"CHL"``).
    indicator_code : str
        World Bank indicator ID (e.g. ``"NY.GDP.PCAP.CD"``).
    date_range : str
        Year range in ``"YYYY:YYYY"`` format.
    per_page : int
        Maximum records per API page.
    max_retries : int
        Number of retry attempts on transient failures.

    Returns
    -------
    list[dict]
        List of ``{"country_code", "country_name", "indicator", "year", "value"}``
        dictionaries.  Entries with ``null`` values are included (cleaned later).
    """
    url = (
        f"{WB_API_BASE}/country/{country_code}/indicator/{indicator_code}"
        f"?format=json&per_page={per_page}&date={date_range}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # The API returns a 2-element list: [metadata, records]
            if not isinstance(data, list) or len(data) < 2:
                logger.warning(
                    "Unexpected response structure for %s/%s (attempt %d)",
                    country_code,
                    indicator_code,
                    attempt,
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return []

            records = data[1]
            if records is None:
                return []

            parsed: List[Dict] = []
            for rec in records:
                parsed.append(
                    {
                        "country_code": country_code,
                        "country_name": COUNTRIES.get(country_code, country_code),
                        "indicator": indicator_code,
                        "indicator_name": INDICATORS.get(indicator_code, indicator_code),
                        "year": int(rec.get("date", 0)),
                        "value": rec.get("value"),
                    }
                )
            return parsed

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed for %s/%s (attempt %d/%d): %s",
                country_code,
                indicator_code,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                logger.error(
                    "All %d attempts exhausted for %s/%s",
                    max_retries,
                    country_code,
                    indicator_code,
                )
                return []

    return []  # pragma: no cover


def extract_all(
    countries: Optional[List[str]] = None,
    indicators: Optional[List[str]] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Extract all indicator data for all configured countries.

    Parameters
    ----------
    countries : list[str] | None
        Country codes to fetch.  Defaults to all six LATAM countries.
    indicators : list[str] | None
        Indicator codes to fetch.  Defaults to all seven indicators.
    progress_callback : callable | None
        Optional ``(current, total, message)`` callback for progress reporting
        (used by the Streamlit UI).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``country_code, country_name, indicator, indicator_name, year, value``.
    """
    countries = countries or COUNTRY_CODES
    indicators = indicators or INDICATOR_CODES

    all_records: List[Dict] = []
    total = len(countries) * len(indicators)
    current = 0

    for cc in countries:
        for ic in indicators:
            current += 1
            label = f"{COUNTRIES.get(cc, cc)} — {INDICATORS.get(ic, ic)}"
            logger.info("Fetching %s  [%d/%d]", label, current, total)

            if progress_callback:
                progress_callback(current, total, f"Fetching {label}")

            records = fetch_indicator(cc, ic)
            all_records.extend(records)

            # Be polite to the API
            time.sleep(0.25)

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("Extraction returned zero records.")
        return df

    logger.info(
        "Extraction complete: %d records for %d countries and %d indicators.",
        len(df),
        df["country_code"].nunique(),
        df["indicator"].nunique(),
    )
    return df
