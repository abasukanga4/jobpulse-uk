---
title: JobPulse UK
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app/dashboard.py
python_version: "3.12"
pinned: false
license: mit
---

# JobPulse UK

> End-to-end pipeline tracking UK AI & data jobs: ingests postings, extracts structured skills, predicts salaries, and visualises skill demand and pay through an interactive dashboard.

**Live demo:** _coming soon_ (Hugging Face Spaces) &nbsp;·&nbsp; **Status:** alpha (v0.1)

> The public demo runs on **synthetic sample data** so it works with no API keys. Point it at the Adzuna API (`--source adzuna`, with keys) for real postings.

## Quickstart

```bash
git clone https://github.com/abasukanga4/jobpulse-uk
cd jobpulse-uk
uv sync --extra dev --extra ml --extra dashboard

# Run the pipeline on the included synthetic sample data (no API keys needed)
uv run jobpulse ingest --source mock
uv run jobpulse extract-skills
uv run jobpulse train-salary
uv run streamlit run app/dashboard.py
```

> Live Adzuna ingestion with Claude-powered skill extraction is optional: also run `uv sync --extra llm`, copy `.env.example` to `.env`, and set `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` / `ANTHROPIC_API_KEY`.

## Why this exists

UK AI/data hiring moved fast in 2025–26 and there was no decent public view of what skills were actually in demand, by region, at what salary. This is the tool I wanted while job hunting.

## Tech

| Layer | Choice |
|---|---|
| Ingest | Adzuna UK API (live) · synthetic mock source (demo) |
| Storage | DuckDB + Parquet |
| Extraction | Claude (structured tool use) · regex keyword fallback |
| Modelling | XGBoost salary regression |
| App | Streamlit on Hugging Face Spaces |
| Quality | uv · ruff · mypy strict · pytest · GitHub Actions CI |

## Roadmap (v0.2)

Planned, not yet built — listed separately so the stack above reflects only what's in the code today:

- **Role clustering** — sentence-transformer embeddings → UMAP → HDBSCAN to surface emerging role archetypes.
- **Embedding-based skill normalisation** beyond the current keyword extractor.
- **Scheduled ingestion** — a cron workflow pulling fresh Adzuna data daily.

## Licence

MIT — see [LICENSE](LICENSE).
