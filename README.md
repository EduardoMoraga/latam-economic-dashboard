# LATAM Economic Dashboard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end **ETL + Machine Learning + Interactive Dashboard** project that extracts economic indicators from the [World Bank API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392), transforms and stores them in SQLite, generates forecasts with ML models, and presents everything through a polished Streamlit application.

**6 countries** &bull; **7 indicators** &bull; **25 years of data** &bull; **5-year forecasts**

---

## Architecture

```
                          World Bank API v2
                               │
                    ┌──────────▼──────────┐
                    │      EXTRACT        │
                    │  requests + retry   │
                    │  6 countries        │
                    │  7 indicators       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     TRANSFORM       │
                    │  Clean & pivot      │
                    │  YoY growth rates   │
                    │  Normalization      │
                    │  Lag features       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐ ┌────▼─────┐ ┌────────▼────────┐
    │   SQLite DB      │ │  ML      │ │   Streamlit     │
    │  raw_indicators  │ │  Models  │ │   Dashboard     │
    │  transformed     │ │  RF, ES  │ │   6 pages       │
    │  forecasts       │ │  LR     │  │   Plotly charts │
    └──────────────────┘ └────┬─────┘ └─────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    Forecasts       │
                    │  5-year horizon    │
                    │  Confidence bands  │
                    └───────────────────┘
```

## Countries & Indicators

| Country   | Code | Indicators                                                        |
|-----------|------|-------------------------------------------------------------------|
| Chile     | CHL  | GDP per Capita, GDP Growth, Inflation, Trade (% GDP),             |
| Argentina | ARG  | Merchandise Exports, Unemployment, Internet Users                 |
| Brazil    | BRA  |                                                                   |
| Mexico    | MEX  | All indicators sourced from World Bank Open Data.                 |
| Colombia  | COL  | Date range: **2000 - 2024**                                       |
| Peru      | PER  |                                                                   |

## Dashboard Pages

| Page                 | Description                                                    |
|----------------------|----------------------------------------------------------------|
| **Overview**         | Country cards with latest GDP, inflation, growth, unemployment |
| **ETL Pipeline**     | Visual pipeline with status indicators; one-click data refresh |
| **Explorer**         | Interactive line charts — select indicators and countries      |
| **ML Forecasts**     | Historical + 5-year forecast with confidence intervals         |
| **Feature Importance** | Random Forest analysis of GDP growth predictors              |
| **Country Comparison** | Radar chart comparing two countries on normalized indicators |

## Screenshots

> Screenshots will be added after first deployment.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/eduardomoraga/latam-economic-dashboard.git
cd latam-economic-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

Then navigate to **ETL Pipeline** and click **Run Full Pipeline** to fetch data.

## Tech Stack

| Component       | Technology                                      |
|-----------------|-------------------------------------------------|
| Data Source      | World Bank API v2 (free, no token required)     |
| ETL             | Python, requests, pandas                        |
| Storage         | SQLite (zero-config, file-based)                |
| ML              | scikit-learn, statsmodels                       |
| Visualization   | Plotly, Streamlit                               |
| Forecasting     | Exponential Smoothing, Linear Regression        |
| Feature Analysis| Random Forest with TimeSeriesSplit CV           |

## ML Models

- **Linear Regression** — baseline trend extrapolation for each indicator
- **Exponential Smoothing** (Holt) — captures level + trend in time series
- **Random Forest Regressor** — multivariate GDP growth prediction using all other indicators as features, with proper time-series cross-validation

## Project Structure

```
latam-economic-dashboard/
├── app.py                  # Streamlit application (main entry point)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── extract.py          # World Bank API data extraction
│   ├── transform.py        # Data cleaning & feature engineering
│   ├── load.py             # SQLite persistence layer
│   ├── models.py           # ML models & forecasting
│   └── utils.py            # Constants, configuration, helpers
├── data/
│   └── .gitkeep            # Database created at runtime
├── notebooks/
│   └── exploration.ipynb   # Exploratory data analysis
└── outputs/
    └── .gitkeep            # Generated artifacts
```

## Key Findings

> Findings are populated after running the pipeline. Example insights:
>
> - Chile consistently leads LATAM in GDP per capita
> - Argentina exhibits the highest inflation volatility
> - Internet adoption shows the strongest convergence trend across all six countries
> - Trade openness and GDP per capita are the strongest predictors of GDP growth

## License

MIT License - Copyright (c) 2025 Eduardo Moraga

---

<p align="center">
  Built by <a href="https://eduardomoraga.github.io">Eduardo Moraga</a>
</p>
