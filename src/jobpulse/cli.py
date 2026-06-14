"""`jobpulse` CLI — the public entrypoint."""

from __future__ import annotations

from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.table import Table

from jobpulse.extract.skills import build_default_extractor, extract_many
from jobpulse.ingest.pipeline import run_ingest
from jobpulse.ml import salary as salary_model
from jobpulse.models import Source
from jobpulse.storage import JobStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

app = typer.Typer(
    name="jobpulse",
    add_completion=False,
    help="End-to-end UK AI/data job tracker.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    source: Source = typer.Option(Source.MOCK, "--source", "-s", help="Where to pull jobs from."),
    fixture: Path | None = typer.Option(
        Path("fixtures/sample_jobs.parquet"),
        "--fixture",
        help="Parquet fixture used when source=mock and the file exists.",
    ),
    mock_n: int = typer.Option(200, "--n", help="Number of jobs to generate when source=mock."),
    adzuna_pages: int = typer.Option(1, "--pages", help="Pages per query for Adzuna."),
) -> None:
    """Ingest jobs into the local store."""
    store = JobStore()
    stats = run_ingest(
        source=source,
        store=store,
        mock_n=mock_n,
        adzuna_pages_per_query=adzuna_pages,
        fixture_path=fixture,
    )
    table = Table(title=f"Ingest · {source.value}")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("fetched", str(stats.fetched))
    table.add_row("inserted", str(stats.inserted))
    table.add_row("duplicates", str(stats.duplicates))
    table.add_row("duration", f"{stats.duration_seconds}s")
    console.print(table)


@app.command("extract-skills")
def extract_skills(
    limit: int | None = typer.Option(
        None, "--limit", help="Only process the first N un-extracted jobs."
    ),
) -> None:
    """Run skill extraction on jobs that don't have skills yet."""
    store = JobStore()
    pending = store.jobs_without_skills(limit=limit)
    if not pending:
        console.print("[green]Nothing to extract — every job already has skills.[/green]")
        return
    extractor = build_default_extractor()
    backend = type(extractor).__name__
    console.print(f"Extracting skills for [bold]{len(pending)}[/bold] jobs using {backend}…")
    skills = extract_many(pending, extractor)
    written = store.upsert_skills(skills)
    console.print(f"[green]wrote {written} skill records[/green]")


@app.command("train-salary")
def train_salary(
    out: Path = typer.Option(
        Path("artefacts/salary_model.joblib"),
        "--out",
        help="Where to write the trained pipeline.",
    ),
) -> None:
    """Train the salary regression model on the joined view in the store."""
    store = JobStore()
    df = store.joined_df()
    pipeline, report = salary_model.train(df)
    salary_model.save(pipeline, out)
    table = Table(title="Salary model")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("rows", str(report.n_train))
    table.add_row("CV MAE", f"£{int(report.cv_mae):,}")
    table.add_row("CV R²", f"{report.cv_r2:.3f}")
    table.add_row("artefact", str(out))
    console.print(table)
    top = list(report.feature_importance.items())[:5]
    console.print("\n[bold]Top features[/bold]")
    for name, score in top:
        console.print(f"  {name:35s} {score:.3f}")


@app.command()
def stats() -> None:
    """Show what's currently in the store."""
    store = JobStore()
    df = store.jobs_df()
    if df.empty:
        console.print("[yellow]No jobs in store. Run `jobpulse ingest` first.[/yellow]")
        return
    console.print(f"[bold]{len(df)}[/bold] jobs in store")
    console.print("\n[bold]Top regions[/bold]")
    console.print(df["region"].value_counts().head(8).to_string())
    console.print("\n[bold]Top titles[/bold]")
    console.print(df["title"].value_counts().head(8).to_string())


if __name__ == "__main__":
    app()
