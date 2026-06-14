"""Deterministic mock-job generator.

Used for local dev, CI, and the public demo so the pipeline runs end-to-end
without API keys. The data is plausible but synthetic — descriptions are
templated from real-looking skill/role/seniority pools so the downstream
LLM and regex extractors have signal to work with.
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime, timedelta

from jobpulse.models import Job, Source, WorkplaceType

# Role pools — each entry is (title_template, seniority_band, base_salary).
_ROLES: list[tuple[str, str, int]] = [
    ("Data Scientist", "mid", 65_000),
    ("Senior Data Scientist", "senior", 90_000),
    ("Lead Data Scientist", "lead", 115_000),
    ("Junior Data Analyst", "junior", 35_000),
    ("Data Analyst", "mid", 50_000),
    ("Analytics Engineer", "mid", 65_000),
    ("Senior Analytics Engineer", "senior", 85_000),
    ("Data Engineer", "mid", 70_000),
    ("Senior Data Engineer", "senior", 95_000),
    ("Staff Data Engineer", "lead", 125_000),
    ("Machine Learning Engineer", "mid", 75_000),
    ("Senior Machine Learning Engineer", "senior", 105_000),
    ("Principal Machine Learning Engineer", "principal", 140_000),
    ("MLOps Engineer", "mid", 75_000),
    ("AI Engineer", "mid", 80_000),
    ("Senior AI Engineer", "senior", 110_000),
    ("LLM Engineer", "mid", 90_000),
    ("Applied Scientist", "senior", 105_000),
    ("Research Scientist (ML)", "senior", 110_000),
    ("Head of Data", "lead", 145_000),
]

# (region, location string, salary multiplier, weight)
_LOCATIONS: list[tuple[str, str, float, int]] = [
    ("London", "London, UK", 1.20, 40),
    ("Manchester", "Manchester, UK", 0.92, 12),
    ("Cambridge", "Cambridge, UK", 1.10, 8),
    ("Edinburgh", "Edinburgh, UK", 0.95, 8),
    ("Bristol", "Bristol, UK", 0.95, 7),
    ("Birmingham", "Birmingham, UK", 0.88, 6),
    ("Leeds", "Leeds, UK", 0.88, 5),
    ("Oxford", "Oxford, UK", 1.05, 4),
    ("Glasgow", "Glasgow, UK", 0.88, 4),
    ("Sheffield", "Sheffield, UK", 0.85, 3),
    ("Remote", "Remote (UK)", 1.00, 18),
]

_COMPANIES = [
    "Monzo",
    "Revolut",
    "Wise",
    "Octopus Energy",
    "Ocado",
    "Deliveroo",
    "BenevolentAI",
    "DeepMind",
    "Stability AI",
    "Faculty AI",
    "Cohere",
    "Hugging Face",
    "Snowflake",
    "Databricks",
    "BBC",
    "Sky",
    "ITV Digital",
    "Bumble",
    "Babylon Health",
    "ASOS",
    "boohoo",
    "Marks & Spencer Digital",
    "Tesco Labs",
    "Vodafone",
    "BT Group",
    "Sainsbury's Tech",
    "Trainline",
    "Rightmove",
    "Zoopla",
    "Just Eat",
    "Lloyds Banking Group",
    "Barclays",
    "HSBC Tech",
    "NatWest Group",
    "Capital One UK",
    "AstraZeneca",
    "GSK",
    "Roche UK",
    "Ada Health",
    "Improbable",
    "Quantexa",
    "Onfido",
    "Tessian",
    "Multiverse",
]

_SKILL_BLURBS = {
    "python_core": "You're fluent in Python and write tests for the code you ship.",
    "sql": "Strong SQL — you can untangle a 200-line analytics query.",
    "pytorch": "Experience training models in PyTorch, especially transformers.",
    "tensorflow": "TensorFlow or Keras for production model training.",
    "sklearn": "scikit-learn for classical ML — feature engineering, pipelines, evaluation.",
    "xgboost": "Comfortable with gradient-boosted models (XGBoost / LightGBM).",
    "llm_rag": "Hands-on with LLMs, RAG, and embedding-based retrieval (OpenAI, Claude, "
    "Hugging Face).",
    "agents": "Built agentic workflows — tool use, function calling, evals.",
    "spark": "Distributed data processing with Spark / PySpark.",
    "dbt": "Modelling warehouse data with dbt.",
    "airflow": "Authoring DAGs in Airflow, Dagster, or Prefect.",
    "snowflake": "Production experience with Snowflake or BigQuery.",
    "aws": "Production AWS — S3, Lambda, ECS, SageMaker.",
    "gcp": "GCP — BigQuery, Dataflow, Vertex AI.",
    "azure": "Azure — Synapse, Data Factory, ML Studio.",
    "kubernetes": "Kubernetes — deploying ML services as containers.",
    "mlops": "MLOps practice — CI/CD for models, monitoring, drift detection.",
    "experimentation": "Designing and analysing A/B tests.",
    "stats": "Strong statistical grounding — causal inference, Bayesian methods.",
    "viz": "Communicating insights with Looker, Tableau, or Streamlit.",
}

# Which skills cluster naturally with which roles
_ROLE_SKILL_GROUPS: dict[str, list[list[str]]] = {
    "Data Scientist": [
        ["python_core", "sql", "sklearn", "stats", "experimentation"],
        ["python_core", "sql", "xgboost", "stats", "viz"],
    ],
    "Data Analyst": [["sql", "viz", "experimentation", "python_core"]],
    "Analytics Engineer": [["sql", "dbt", "snowflake", "python_core"]],
    "Data Engineer": [
        ["python_core", "sql", "spark", "airflow", "snowflake"],
        ["python_core", "sql", "dbt", "airflow", "aws"],
    ],
    "Machine Learning Engineer": [
        ["python_core", "pytorch", "mlops", "aws", "kubernetes"],
        ["python_core", "sklearn", "xgboost", "mlops", "gcp"],
    ],
    "MLOps Engineer": [["python_core", "mlops", "kubernetes", "aws", "airflow"]],
    "AI Engineer": [["python_core", "llm_rag", "agents", "aws", "mlops"]],
    "LLM Engineer": [["python_core", "llm_rag", "agents", "pytorch", "mlops"]],
    "Applied Scientist": [["python_core", "pytorch", "stats", "llm_rag"]],
    "Research Scientist (ML)": [["python_core", "pytorch", "stats"]],
    "Head of Data": [["sql", "experimentation", "stats", "viz"]],
}

_WORKPLACE_DIST = [
    (WorkplaceType.HYBRID, 50),
    (WorkplaceType.REMOTE, 30),
    (WorkplaceType.ONSITE, 20),
]


def _pick_role(rng: random.Random) -> tuple[str, str, int]:
    return rng.choice(_ROLES)


def _pick_location(rng: random.Random) -> tuple[str, str, float]:
    region, loc, mult, _ = rng.choices(_LOCATIONS, weights=[w for *_, w in _LOCATIONS], k=1)[0]
    return region, loc, mult


def _pick_workplace(rng: random.Random, location: str) -> WorkplaceType:
    if location == "Remote":
        return WorkplaceType.REMOTE
    return rng.choices([w for w, _ in _WORKPLACE_DIST], weights=[w for _, w in _WORKPLACE_DIST])[0]


def _skill_keys_for(title: str, rng: random.Random) -> list[str]:
    # Find the canonical role key inside the title
    for key, groups in _ROLE_SKILL_GROUPS.items():
        if key in title:
            return rng.choice(groups)
    return ["python_core", "sql"]


def _build_description(title: str, company: str, location: str, skills: list[str]) -> str:
    skill_lines = "\n".join(f"- {_SKILL_BLURBS[k]}" for k in skills)
    return (
        f"{company} is hiring a {title} to join the team in {location}.\n\n"
        f"What you'll do:\n"
        f"- Partner with product and engineering to ship data-driven features.\n"
        f"- Own work end-to-end, from problem framing to production.\n\n"
        f"What we're looking for:\n{skill_lines}\n\n"
        f"We offer a competitive package, learning budget, and 28 days holiday."
    )


def _salary_band(base: int, mult: float, rng: random.Random) -> tuple[int, int]:
    centre = int(base * mult)
    spread = rng.randint(8_000, 18_000)
    lo = max(28_000, centre - spread)
    hi = centre + spread
    # Round to nearest 1000 so the numbers look like job ads
    return (round(lo, -3), round(hi, -3))


def generate_jobs(n: int = 200, *, seed: int = 42) -> list[Job]:
    """Generate `n` synthetic UK AI/data job postings."""
    rng = random.Random(seed)
    jobs: list[Job] = []
    now = datetime.now(UTC).replace(tzinfo=None)

    for i in range(n):
        title, _seniority, base = _pick_role(rng)
        region, location, mult = _pick_location(rng)
        company = rng.choice(_COMPANIES)
        workplace = _pick_workplace(rng, region)
        skills = _skill_keys_for(title, rng)
        description = _build_description(title, company, location, skills)
        salary_min, salary_max = _salary_band(base, mult, rng)
        posted_at = now - timedelta(days=rng.randint(0, 30))
        # Stable, wall-clock-independent source_id: with a fixed seed the row
        # index keeps it both deterministic across runs and unique per row, so
        # re-generation dedupes idempotently.
        sid = hashlib.sha1(f"{title}|{company}|{location}|{i}".encode()).hexdigest()[:16]
        url = f"https://example.com/mock/{sid}"

        jobs.append(
            Job(
                source=Source.MOCK,
                source_id=sid,
                title=title,
                company=company,
                location=location,
                region=region,
                workplace_type=workplace,
                description=description,
                url=url,
                salary_min=salary_min,
                salary_max=salary_max,
                salary_currency="GBP",
                posted_at=posted_at,
                ingested_at=now,
            )
        )
    return jobs
