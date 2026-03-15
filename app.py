"""
LATAM Economic Dashboard — Streamlit Application

An interactive dashboard for exploring economic indicators, ML forecasts,
and feature-importance analysis across six Latin American economies.

Run with:
    streamlit run app.py

Author: Eduardo Moraga
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root is on the path so ``src`` is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.extract import extract_all
from src.load import (
    DB_PATH,
    get_meta,
    load_table,
    save_forecasts,
    save_raw,
    save_transformed,
)
from src.models import generate_all_forecasts, train_gdp_growth_rf
from src.transform import clean_raw, run_transform
from src.utils import (
    BRAND_COLORS,
    COUNTRIES,
    COUNTRY_COLORS,
    COUNTRY_NAMES,
    INDICATORS,
    INDICATOR_SHORT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LATAM Economic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {BRAND_COLORS['background']};
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {BRAND_COLORS['surface']};
    }}
    /* Cards */
    .metric-card {{
        background: {BRAND_COLORS['card']};
        border: 1px solid {BRAND_COLORS['border']};
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
    }}
    .metric-card h3 {{
        color: {BRAND_COLORS['accent']};
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
    }}
    .metric-card .value {{
        color: {BRAND_COLORS['text']};
        font-size: 1.6rem;
        font-weight: 700;
    }}
    .metric-card .label {{
        color: {BRAND_COLORS['text_muted']};
        font-size: 0.8rem;
    }}
    /* Pipeline step */
    .pipeline-step {{
        background: {BRAND_COLORS['card']};
        border: 1px solid {BRAND_COLORS['border']};
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
    .pipeline-step.active {{
        border-color: {BRAND_COLORS['accent']};
    }}
    .pipeline-arrow {{
        font-size: 2rem;
        color: {BRAND_COLORS['accent']};
        text-align: center;
        line-height: 3rem;
    }}
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: {BRAND_COLORS['text_muted']};
        border-top: 1px solid {BRAND_COLORS['border']};
        margin-top: 3rem;
    }}
    .footer a {{
        color: {BRAND_COLORS['accent']};
        text-decoration: none;
    }}
    /* Header accent bar */
    .accent-bar {{
        height: 4px;
        background: linear-gradient(90deg, {BRAND_COLORS['accent']}, {BRAND_COLORS['accent_light']});
        border-radius: 2px;
        margin-bottom: 1.5rem;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Plotly dark template ─────────────────────────────────────────────────────

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BRAND_COLORS["background"],
        plot_bgcolor=BRAND_COLORS["surface"],
        font=dict(color=BRAND_COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=BRAND_COLORS["border"], zerolinecolor=BRAND_COLORS["border"]),
        yaxis=dict(gridcolor=BRAND_COLORS["border"], zerolinecolor=BRAND_COLORS["border"]),
        colorway=[
            BRAND_COLORS["accent"],
            "#58a6ff",
            "#3fb950",
            "#f0883e",
            "#d29922",
            "#f778ba",
            "#a371f7",
        ],
    )
)

# ── Data loading with caching ────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load raw indicators from the database, or return empty DataFrame."""
    return load_table("raw_indicators")


@st.cache_data(ttl=3600, show_spinner=False)
def load_transformed_data() -> pd.DataFrame:
    """Load transformed data from the database."""
    return load_table("transformed")


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast_data() -> pd.DataFrame:
    """Load forecasts from the database."""
    return load_table("forecasts")


def data_is_available() -> bool:
    """Check whether the database has been populated."""
    return os.path.exists(DB_PATH) and not load_raw_data().empty


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## LATAM Economic Dashboard")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "ETL Pipeline",
            "Explorer",
            "ML Forecasts",
            "Feature Importance",
            "Country Comparison",
        ],
        index=0,
    )

    st.markdown("---")

    # Data freshness
    last_updated = get_meta("raw_last_updated")
    if last_updated:
        st.caption(f"Data last refreshed: {last_updated[:19]} UTC")
    else:
        st.caption("No data yet — run the ETL Pipeline.")

    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;color:{BRAND_COLORS["text_muted"]};font-size:0.75rem;">'
        f"v1.0.0 &bull; MIT License</div>",
        unsafe_allow_html=True,
    )


# ── Helper: render footer ───────────────────────────────────────────────────


def render_footer() -> None:
    """Render the page footer."""
    st.markdown(
        '<div class="footer">'
        'Built by <a href="https://eduardomoraga.github.io" target="_blank">Eduardo Moraga</a>'
        " &bull; LATAM Economic Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════


def page_overview() -> None:
    """Render the Overview page with country summary cards."""
    st.markdown("# Overview")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    if not data_is_available():
        st.info("No data available. Go to **ETL Pipeline** to fetch data first.")
        return

    raw = load_raw_data()
    latest_year = int(raw["year"].max())

    st.markdown(f"**Latest data year: {latest_year}** &mdash; 6 LATAM economies, 7 indicators")
    st.markdown("")

    cols = st.columns(3)

    for idx, (code, name) in enumerate(COUNTRIES.items()):
        col = cols[idx % 3]
        country_data = raw[(raw["country_code"] == code) & (raw["year"] == latest_year)]

        with col:
            color = COUNTRY_COLORS.get(name, BRAND_COLORS["accent"])

            gdp_pc = country_data[country_data["indicator"] == "NY.GDP.PCAP.CD"]["value"]
            gdp_pc_val = f"${float(gdp_pc.iloc[0]):,.0f}" if not gdp_pc.empty and pd.notna(gdp_pc.iloc[0]) else "N/A"

            gdp_g = country_data[country_data["indicator"] == "NY.GDP.MKTP.KD.ZG"]["value"]
            gdp_g_val = f"{float(gdp_g.iloc[0]):.1f}%" if not gdp_g.empty and pd.notna(gdp_g.iloc[0]) else "N/A"

            infl = country_data[country_data["indicator"] == "FP.CPI.TOTL.ZG"]["value"]
            infl_val = f"{float(infl.iloc[0]):.1f}%" if not infl.empty and pd.notna(infl.iloc[0]) else "N/A"

            unemp = country_data[country_data["indicator"] == "SL.UEM.TOTL.ZS"]["value"]
            unemp_val = f"{float(unemp.iloc[0]):.1f}%" if not unemp.empty and pd.notna(unemp.iloc[0]) else "N/A"

            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <h3>{name}</h3>
                    <div class="value">{gdp_pc_val}</div>
                    <div class="label">GDP per Capita ({latest_year})</div>
                    <hr style="border-color:{BRAND_COLORS['border']};margin:0.5rem 0;">
                    <div class="label">
                        Growth: <strong>{gdp_g_val}</strong> &bull;
                        Inflation: <strong>{infl_val}</strong> &bull;
                        Unemployment: <strong>{unemp_val}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Summary chart: GDP per capita bar chart
    st.markdown("### GDP per Capita Comparison")
    latest = raw[(raw["indicator"] == "NY.GDP.PCAP.CD") & (raw["year"] == latest_year)].copy()
    latest["value"] = pd.to_numeric(latest["value"], errors="coerce")
    latest = latest.dropna(subset=["value"]).sort_values("value", ascending=True)

    if not latest.empty:
        fig = px.bar(
            latest,
            y="country_name",
            x="value",
            orientation="h",
            labels={"value": "USD", "country_name": ""},
            color="country_name",
            color_discrete_map=COUNTRY_COLORS,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=350, margin=dict(l=0, r=20, t=10, b=0))
        fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


def page_etl_pipeline() -> None:
    """Render the ETL Pipeline page with controls to run the pipeline."""
    st.markdown("# ETL Pipeline")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    st.markdown("Extract data from the **World Bank API**, transform it, and load into SQLite.")

    # Visual pipeline
    p1, arrow1, p2, arrow2, p3 = st.columns([3, 1, 3, 1, 3])

    with p1:
        st.markdown(
            '<div class="pipeline-step">'
            "<h4>Extract</h4>"
            "<p>World Bank API v2<br>6 countries &bull; 7 indicators<br>2000-2024</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with arrow1:
        st.markdown('<div class="pipeline-arrow">&#10140;</div>', unsafe_allow_html=True)
    with p2:
        st.markdown(
            '<div class="pipeline-step">'
            "<h4>Transform</h4>"
            "<p>Clean &bull; Pivot<br>YoY Growth &bull; Normalize<br>Lag Features</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with arrow2:
        st.markdown('<div class="pipeline-arrow">&#10140;</div>', unsafe_allow_html=True)
    with p3:
        st.markdown(
            '<div class="pipeline-step">'
            "<h4>Load</h4>"
            "<p>SQLite Database<br>3 tables<br>+ ML Forecasts</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Status indicators
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        ts = get_meta("raw_last_updated")
        st.metric("Raw Data", "Ready" if ts else "Empty", delta=ts[:19] if ts else None)
    with col_s2:
        ts = get_meta("transformed_last_updated")
        st.metric("Transformed", "Ready" if ts else "Empty", delta=ts[:19] if ts else None)
    with col_s3:
        ts = get_meta("forecasts_last_updated")
        st.metric("Forecasts", "Ready" if ts else "Empty", delta=ts[:19] if ts else None)

    st.markdown("---")

    if st.button("Run Full Pipeline", type="primary", use_container_width=True):
        _run_pipeline()


def _run_pipeline() -> None:
    """Execute the full ETL + forecasting pipeline with progress reporting."""
    progress = st.progress(0, text="Starting pipeline...")

    # Step 1: Extract
    progress.progress(5, text="Extracting data from World Bank API...")
    status_extract = st.status("Extracting data...", expanded=True)

    def extract_progress(current: int, total: int, msg: str) -> None:
        pct = int(5 + (current / total) * 40)
        progress.progress(pct, text=msg)
        status_extract.write(msg)

    raw_df = extract_all(progress_callback=extract_progress)
    status_extract.update(label=f"Extraction complete: {len(raw_df)} records", state="complete")

    if raw_df.empty:
        st.error("Extraction returned no data. Check your internet connection.")
        progress.empty()
        return

    # Step 2: Transform
    progress.progress(50, text="Transforming data...")
    status_transform = st.status("Transforming data...", expanded=True)
    cleaned = clean_raw(raw_df)
    transformed = run_transform(raw_df)
    status_transform.write(f"Cleaned: {len(cleaned)} records")
    status_transform.write(f"Transformed: {len(transformed)} rows x {len(transformed.columns)} columns")
    status_transform.update(label="Transformation complete", state="complete")

    # Step 3: Load
    progress.progress(60, text="Loading into database...")
    save_raw(cleaned)
    save_transformed(transformed)

    # Step 4: Forecasts
    progress.progress(65, text="Generating ML forecasts...")
    status_fc = st.status("Generating forecasts...", expanded=True)

    def fc_progress(current: int, total: int, msg: str) -> None:
        pct = int(65 + (current / total) * 30)
        progress.progress(pct, text=msg)
        status_fc.write(msg)

    forecasts = generate_all_forecasts(cleaned, progress_callback=fc_progress)
    save_forecasts(forecasts)
    status_fc.update(label=f"Forecasting complete: {len(forecasts)} predictions", state="complete")

    progress.progress(100, text="Pipeline complete!")

    # Clear caches so new data is picked up
    load_raw_data.clear()
    load_transformed_data.clear()
    load_forecast_data.clear()

    st.success("Pipeline finished successfully. Navigate to other pages to explore the data.")


def page_explorer() -> None:
    """Interactive indicator explorer with multi-country line charts."""
    st.markdown("# Data Explorer")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    if not data_is_available():
        st.info("No data available. Run the ETL Pipeline first.")
        return

    raw = load_raw_data()

    col1, col2 = st.columns(2)
    with col1:
        selected_indicator = st.selectbox(
            "Indicator",
            list(INDICATORS.keys()),
            format_func=lambda x: INDICATORS[x],
        )
    with col2:
        selected_countries = st.multiselect(
            "Countries",
            COUNTRY_NAMES,
            default=COUNTRY_NAMES,
        )

    if not selected_countries:
        st.warning("Select at least one country.")
        return

    subset = raw[
        (raw["indicator"] == selected_indicator)
        & (raw["country_name"].isin(selected_countries))
    ].copy()
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.dropna(subset=["value"])

    if subset.empty:
        st.warning("No data available for this selection.")
        return

    fig = px.line(
        subset,
        x="year",
        y="value",
        color="country_name",
        markers=True,
        labels={"value": INDICATORS[selected_indicator], "year": "Year", "country_name": "Country"},
        color_discrete_map=COUNTRY_COLORS,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=20, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    with st.expander("View Data Table"):
        pivot = subset.pivot_table(index="year", columns="country_name", values="value")
        st.dataframe(pivot.style.format("{:,.2f}"), use_container_width=True)


def page_ml_forecasts() -> None:
    """Display ML forecasts with confidence intervals and accuracy metrics."""
    st.markdown("# ML Forecasts")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    if not data_is_available():
        st.info("No data available. Run the ETL Pipeline first.")
        return

    raw = load_raw_data()
    forecasts = load_forecast_data()

    if forecasts.empty:
        st.info("No forecasts available. Run the ETL Pipeline to generate them.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country", COUNTRY_NAMES, key="fc_country")
    with col2:
        indicator = st.selectbox(
            "Indicator",
            list(INDICATORS.keys()),
            format_func=lambda x: INDICATORS[x],
            key="fc_indicator",
        )
    with col3:
        model_name = st.selectbox(
            "Model",
            forecasts["model"].unique().tolist(),
            key="fc_model",
        )

    # Historical data
    hist = raw[
        (raw["country_name"] == country) & (raw["indicator"] == indicator)
    ].copy()
    hist["value"] = pd.to_numeric(hist["value"], errors="coerce")
    hist = hist.dropna(subset=["value"]).sort_values("year")

    # Forecast data
    fc = forecasts[
        (forecasts["country_name"] == country)
        & (forecasts["indicator"] == indicator)
        & (forecasts["model"] == model_name)
    ].sort_values("year")

    if hist.empty and fc.empty:
        st.warning("No data for this selection.")
        return

    # Build chart
    fig = go.Figure()

    # Historical line
    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["year"],
                y=hist["value"],
                mode="lines+markers",
                name="Historical",
                line=dict(color=BRAND_COLORS["accent"], width=2),
                marker=dict(size=5),
            )
        )

    # Forecast line + CI
    if not fc.empty:
        fig.add_trace(
            go.Scatter(
                x=fc["year"],
                y=fc["predicted_value"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#58a6ff", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fc["year"], fc["year"][::-1]]),
                y=pd.concat([fc["confidence_upper"], fc["confidence_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(88,166,255,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                showlegend=True,
            )
        )

        # Connect historical to forecast
        if not hist.empty:
            bridge_x = [int(hist["year"].iloc[-1]), int(fc["year"].iloc[0])]
            bridge_y = [float(hist["value"].iloc[-1]), float(fc["predicted_value"].iloc[0])]
            fig.add_trace(
                go.Scatter(
                    x=bridge_x,
                    y=bridge_y,
                    mode="lines",
                    line=dict(color="#58a6ff", width=1, dash="dot"),
                    showlegend=False,
                )
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"{country} — {INDICATORS[indicator]}",
        xaxis_title="Year",
        yaxis_title=INDICATORS[indicator],
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=0, r=20, t=60, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy metrics
    if not fc.empty:
        m_col1, m_col2, m_col3 = st.columns(3)
        rmse_val = fc["rmse"].iloc[0]
        mae_val = fc["mae"].iloc[0]
        r2_val = fc["r2"].iloc[0]

        with m_col1:
            st.metric("RMSE", f"{rmse_val:.4f}" if pd.notna(rmse_val) else "N/A")
        with m_col2:
            st.metric("MAE", f"{mae_val:.4f}" if pd.notna(mae_val) else "N/A")
        with m_col3:
            st.metric("R-squared", f"{r2_val:.4f}" if pd.notna(r2_val) else "N/A")

    with st.expander("Forecast Data"):
        if not fc.empty:
            st.dataframe(
                fc[["year", "predicted_value", "confidence_lower", "confidence_upper"]].style.format(
                    {"predicted_value": "{:,.2f}", "confidence_lower": "{:,.2f}", "confidence_upper": "{:,.2f}"}
                ),
                use_container_width=True,
            )


def page_feature_importance() -> None:
    """Show which indicators best predict GDP growth using Random Forest."""
    st.markdown("# Feature Importance")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    st.markdown("Which economic indicators best predict **GDP Growth**? Analysis via Random Forest.")

    if not data_is_available():
        st.info("No data available. Run the ETL Pipeline first.")
        return

    transformed = load_transformed_data()
    if transformed.empty or "GDP Growth" not in transformed.columns:
        st.warning("Transformed data not available or missing GDP Growth column.")
        return

    with st.spinner("Training Random Forest model..."):
        model, metrics, importances = train_gdp_growth_rf(transformed)

    if importances.empty:
        st.warning("Not enough data to compute feature importances.")
        return

    # Display metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("CV RMSE", f"{metrics['rmse']:.4f}" if pd.notna(metrics["rmse"]) else "N/A")
    with m2:
        st.metric("CV MAE", f"{metrics['mae']:.4f}" if pd.notna(metrics["mae"]) else "N/A")
    with m3:
        st.metric("CV R-squared", f"{metrics['r2']:.4f}" if pd.notna(metrics["r2"]) else "N/A")

    # Top 15 features bar chart
    top = importances.head(15).sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            y=top.index,
            x=top.values,
            orientation="h",
            marker_color=BRAND_COLORS["accent"],
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Top 15 Predictors of GDP Growth",
        xaxis_title="Feature Importance",
        yaxis_title="",
        height=500,
        margin=dict(l=0, r=20, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("All Feature Importances"):
        st.dataframe(
            importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}).style.format(
                {"Importance": "{:.4f}"}
            ),
            use_container_width=True,
        )


def page_country_comparison() -> None:
    """Radar chart comparing two countries across normalized indicators."""
    st.markdown("# Country Comparison")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    if not data_is_available():
        st.info("No data available. Run the ETL Pipeline first.")
        return

    transformed = load_transformed_data()
    if transformed.empty:
        st.warning("Transformed data not available.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        country_a = st.selectbox("Country A", COUNTRY_NAMES, index=0, key="cmp_a")
    with col2:
        country_b = st.selectbox("Country B", COUNTRY_NAMES, index=1, key="cmp_b")
    with col3:
        year = st.selectbox(
            "Year",
            sorted(transformed["year"].unique(), reverse=True),
            key="cmp_year",
        )

    norm_cols = [c for c in transformed.columns if c.endswith("_norm") and "_yoy" not in c and "_lag" not in c]
    if not norm_cols:
        st.warning("No normalized indicator columns found.")
        return

    row_a = transformed[(transformed["country_name"] == country_a) & (transformed["year"] == year)]
    row_b = transformed[(transformed["country_name"] == country_b) & (transformed["year"] == year)]

    if row_a.empty or row_b.empty:
        st.warning(f"No data for one or both countries in {year}.")
        return

    # Prepare radar data
    categories = [c.replace("_norm", "") for c in norm_cols]
    values_a = [float(row_a[c].iloc[0]) if pd.notna(row_a[c].iloc[0]) else 0 for c in norm_cols]
    values_b = [float(row_b[c].iloc[0]) if pd.notna(row_b[c].iloc[0]) else 0 for c in norm_cols]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_a + [values_a[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=country_a,
            line=dict(color=COUNTRY_COLORS.get(country_a, BRAND_COLORS["accent"])),
            fillcolor=f"rgba({_hex_to_rgb(COUNTRY_COLORS.get(country_a, BRAND_COLORS['accent']))},0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_b + [values_b[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=country_b,
            line=dict(color=COUNTRY_COLORS.get(country_b, "#58a6ff")),
            fillcolor=f"rgba({_hex_to_rgb(COUNTRY_COLORS.get(country_b, '#58a6ff'))},0.15)",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        polar=dict(
            bgcolor=BRAND_COLORS["surface"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=BRAND_COLORS["border"]),
            angularaxis=dict(gridcolor=BRAND_COLORS["border"]),
        ),
        title=f"{country_a} vs {country_b} ({year})",
        height=550,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Side-by-side table
    st.markdown("### Indicator Values (Normalized 0-1)")
    comparison = pd.DataFrame(
        {"Indicator": categories, country_a: values_a, country_b: values_b}
    )
    comparison["Difference"] = comparison[country_a] - comparison[country_b]
    st.dataframe(
        comparison.style.format({country_a: "{:.3f}", country_b: "{:.3f}", "Difference": "{:+.3f}"}),
        use_container_width=True,
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to ``R,G,B`` string for rgba() CSS."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ── Router ───────────────────────────────────────────────────────────────────

PAGES = {
    "Overview": page_overview,
    "ETL Pipeline": page_etl_pipeline,
    "Explorer": page_explorer,
    "ML Forecasts": page_ml_forecasts,
    "Feature Importance": page_feature_importance,
    "Country Comparison": page_country_comparison,
}

PAGES[page]()
render_footer()
