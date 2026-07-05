"""Streamlit dashboard for JobPulse UK.

Run with: `streamlit run app/dashboard.py`
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from jobpulse.ml import salary as salary_model
from jobpulse.models import Seniority
from jobpulse.storage import JobStore

st.set_page_config(
    page_title="JobPulse UK",
    page_icon="📈",
    layout="wide",
)

MODEL_PATH = Path("artefacts/salary_model.joblib")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_jobs() -> pd.DataFrame:
    df = JobStore().joined_df()
    if df.empty:
        return df
    df["salary_mid"] = df[["salary_min", "salary_max"]].mean(axis=1)
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
    df["posted_date"] = df["posted_at"].dt.date
    for col in ("technologies", "frameworks", "cloud", "domains"):
        df[col] = df[col].apply(lambda v: list(v) if hasattr(v, "__iter__") else [])
    return df


@st.cache_resource
def load_model():  # type: ignore[no-untyped-def]
    if MODEL_PATH.exists():
        return salary_model.load(MODEL_PATH)
    return None


def explode_skills(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.Series(Counter(s for row in df[col] for s in row))


# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

df = load_jobs()

if df.empty:
    st.warning("No jobs in the store yet. Run `jobpulse ingest --source mock` then refresh.")
    st.stop()

st.sidebar.title("Filters")
regions = sorted(df["region"].dropna().unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
workplace_options = sorted(df["workplace_type"].dropna().unique())
selected_workplace = st.sidebar.multiselect(
    "Workplace", workplace_options, default=workplace_options
)
salary_range = st.sidebar.slider(
    "Salary mid-point (£)",
    min_value=int(df["salary_mid"].min(skipna=True) or 0),
    max_value=int(df["salary_mid"].max(skipna=True) or 200_000),
    value=(
        int(df["salary_mid"].min(skipna=True) or 0),
        int(df["salary_mid"].max(skipna=True) or 200_000),
    ),
    step=5_000,
)

mask = (
    df["region"].isin(selected_regions)
    & df["workplace_type"].isin(selected_workplace)
    # keep salary-less postings: real Adzuna data is mostly missing salary,
    # and the filter should not silently hide those jobs from every tab
    & (df["salary_mid"].between(*salary_range, inclusive="both") | df["salary_mid"].isna())
)
fdf = df[mask].copy()

# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------

st.title("JobPulse UK")
st.caption(
    "Tracking UK AI & data jobs — skill demand and salary insights, built "
    "end-to-end. This demo runs on synthetic sample data."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Jobs", f"{len(fdf):,}")
col_b.metric("Median salary", f"£{int(fdf['salary_mid'].median(skipna=True) or 0):,}")
col_c.metric("Companies", f"{fdf['company'].nunique():,}")
top_region = fdf["region"].value_counts().idxmax() if not fdf.empty else "—"
col_d.metric("Top region", str(top_region))

tab_overview, tab_skills, tab_salary, tab_table, tab_about = st.tabs(
    ["Overview", "Skills", "Salary", "Browse", "About"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Roles by seniority")
        sen_counts = (
            fdf["seniority"]
            .fillna("unknown")
            .value_counts()
            .reindex([s.value for s in Seniority], fill_value=0)
        )
        fig = px.bar(
            x=sen_counts.index,
            y=sen_counts.values,
            labels={"x": "Seniority", "y": "Jobs"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Workplace mix")
        wp = fdf["workplace_type"].value_counts()
        fig = px.pie(values=wp.values, names=wp.index, hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top hiring companies")
    top_companies = fdf["company"].value_counts().head(12)
    fig = px.bar(
        x=top_companies.values,
        y=top_companies.index,
        orientation="h",
        labels={"x": "Open roles", "y": "Company"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

with tab_skills:
    st.subheader("Skill demand")
    cat = st.selectbox(
        "Category",
        ["frameworks", "technologies", "cloud", "domains"],
        index=0,
    )
    counts = explode_skills(fdf, cat).sort_values(ascending=False).head(20)
    if counts.empty:
        st.info("No skills extracted yet — run `jobpulse extract-skills`.")
    else:
        fig = px.bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            labels={"x": "Mentions", "y": cat.title()},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Skill × region heatmap")  # noqa: RUF001
    heat_rows: list[dict[str, object]] = []
    for _, row in fdf.iterrows():
        for skill in row[cat]:
            heat_rows.append({"region": row["region"], "skill": skill})
    if heat_rows:
        heat_df = pd.DataFrame(heat_rows)
        pivot = heat_df.groupby(["skill", "region"]).size().unstack(fill_value=0)
        top_skills = pivot.sum(axis=1).sort_values(ascending=False).head(15).index
        pivot = pivot.loc[top_skills]
        fig = px.imshow(
            pivot,
            aspect="auto",
            labels={"color": "Mentions"},
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

with tab_salary:
    st.subheader("Salary distribution")
    sal_df = fdf.dropna(subset=["salary_mid"])
    if sal_df.empty:
        st.info("No salaries in the filtered view.")
    else:
        fig = px.histogram(sal_df, x="salary_mid", nbins=30)
        fig.update_layout(xaxis_title="Salary mid (£)", yaxis_title="Jobs")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("By region")
        order = sal_df.groupby("region")["salary_mid"].median().sort_values(ascending=False).index
        fig = px.box(sal_df, x="region", y="salary_mid", category_orders={"region": list(order)})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Salary predictor")
    pipeline = load_model()
    if pipeline is None:
        st.info(
            "Train the salary model first: `jobpulse train-salary`. The "
            "predictor will appear once `artefacts/salary_model.joblib` exists."
        )
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            p_seniority = st.selectbox("Seniority", [s.value for s in Seniority], index=2)
            p_region = st.selectbox("Region", regions, index=0)
        with c2:
            p_workplace = st.selectbox("Workplace", workplace_options, index=0)
            p_tech = st.slider("Technologies known", 0, 6, 2)
        with c3:
            p_frameworks = st.slider("Frameworks", 0, 8, 2)
            p_cloud = st.slider("Cloud", 0, 5, 1)
            p_domains = st.slider("Domains", 0, 6, 1)
        prediction = salary_model.predict(
            pipeline,
            seniority=p_seniority,
            region=p_region,
            workplace_type=p_workplace,
            n_technologies=p_tech,
            n_frameworks=p_frameworks,
            n_cloud=p_cloud,
            n_domains=p_domains,
        )
        st.metric("Predicted salary mid-point", f"£{int(prediction):,}")

# ---------------------------------------------------------------------------
# Browse table
# ---------------------------------------------------------------------------

with tab_table:
    st.subheader("Browse jobs")
    show = fdf[
        ["title", "company", "region", "workplace_type", "salary_min", "salary_max", "url"]
    ].copy()
    show.columns = ["Title", "Company", "Region", "Workplace", "Min £", "Max £", "URL"]
    st.dataframe(show, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown(
        """
        ### About this project

        **JobPulse UK** is an end-to-end pipeline tracking UK AI and data
        jobs. It ingests postings from the Adzuna UK API (or a synthetic mock
        source), extracts structured skills using a Claude-powered extractor
        with a keyword fallback, trains a salary regression model, and surfaces
        the lot through this dashboard.

        *This public demo runs on synthetic sample data so it works without API keys.*

        **Stack:** Python 3.12 · DuckDB · Adzuna · Anthropic · scikit-learn
        · XGBoost · Streamlit · Plotly · GitHub Actions.

        **Code:** [github.com/abasukanga4/jobpulse-uk](https://github.com/abasukanga4/jobpulse-uk)
        """
    )
