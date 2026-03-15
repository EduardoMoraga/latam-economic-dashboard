"""
Utility functions and constants for the LATAM Economic Dashboard.

Centralizes configuration (country codes, indicator definitions, color palettes)
and provides shared helpers used across extraction, transformation, and visualization.
"""

from __future__ import annotations

from typing import Dict, List

# ── Country configuration ────────────────────────────────────────────────────

COUNTRIES: Dict[str, str] = {
    "CHL": "Chile",
    "ARG": "Argentina",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "COL": "Colombia",
    "PER": "Peru",
}

COUNTRY_CODES: List[str] = list(COUNTRIES.keys())
COUNTRY_NAMES: List[str] = list(COUNTRIES.values())

# ── Indicator configuration ──────────────────────────────────────────────────

INDICATORS: Dict[str, str] = {
    "NY.GDP.PCAP.CD": "GDP per Capita (USD)",
    "NY.GDP.MKTP.KD.ZG": "GDP Growth (%)",
    "FP.CPI.TOTL.ZG": "Inflation (%)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "TX.VAL.MRCH.CD.WT": "Merchandise Exports (USD)",
    "SL.UEM.TOTL.ZS": "Unemployment (%)",
    "IT.NET.USER.ZS": "Internet Users (%)",
}

INDICATOR_CODES: List[str] = list(INDICATORS.keys())

# Short names for charts where space is limited
INDICATOR_SHORT: Dict[str, str] = {
    "NY.GDP.PCAP.CD": "GDP/Capita",
    "NY.GDP.MKTP.KD.ZG": "GDP Growth",
    "FP.CPI.TOTL.ZG": "Inflation",
    "NE.TRD.GNFS.ZS": "Trade",
    "TX.VAL.MRCH.CD.WT": "Exports",
    "SL.UEM.TOTL.ZS": "Unemployment",
    "IT.NET.USER.ZS": "Internet",
}

# ── World Bank API ───────────────────────────────────────────────────────────

WB_API_BASE = "https://api.worldbank.org/v2"
WB_DATE_RANGE = "2000:2024"
WB_PER_PAGE = 500

# ── Color palette ────────────────────────────────────────────────────────────

BRAND_COLORS = {
    "background": "#0f1419",
    "surface": "#1a1f25",
    "card": "#1e252d",
    "accent": "#00d4aa",
    "accent_light": "#33debb",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "border": "#30363d",
    "success": "#3fb950",
    "warning": "#d29922",
    "error": "#f85149",
}

COUNTRY_COLORS: Dict[str, str] = {
    "Chile": "#00d4aa",
    "Argentina": "#58a6ff",
    "Brazil": "#3fb950",
    "Mexico": "#f0883e",
    "Colombia": "#d29922",
    "Peru": "#f778ba",
}

# ── Database ─────────────────────────────────────────────────────────────────

DB_PATH = "data/latam_economic.db"

# ── Helpers ──────────────────────────────────────────────────────────────────


def code_to_name(code: str) -> str:
    """Convert a 3-letter ISO country code to its display name."""
    return COUNTRIES.get(code, code)


def name_to_code(name: str) -> str:
    """Convert a country display name back to its ISO code."""
    inverse = {v: k for k, v in COUNTRIES.items()}
    return inverse.get(name, name)


def indicator_label(code: str) -> str:
    """Return the human-readable label for an indicator code."""
    return INDICATORS.get(code, code)


def indicator_short_label(code: str) -> str:
    """Return a short label suitable for chart axes."""
    return INDICATOR_SHORT.get(code, code)
