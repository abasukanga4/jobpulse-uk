"""Tests for the salary model.

These cover the training and predict APIs on a small synthetic frame. They
intentionally avoid checking absolute error thresholds, because the model
is XGBoost-with-noise — flaky thresholds add nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jobpulse.ml import salary as salary_model


def _frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        seniority = rng.choice(["junior", "mid", "senior", "lead"])
        region = rng.choice(["London", "Manchester", "Remote", "Edinburgh"])
        workplace = rng.choice(["hybrid", "remote", "onsite"])
        base = {"junior": 35, "mid": 60, "senior": 90, "lead": 120}[str(seniority)]
        mult = {"London": 1.2, "Manchester": 0.9, "Remote": 1.0, "Edinburgh": 0.95}[str(region)]
        salary = (base + rng.normal(0, 5)) * mult * 1000
        rows.append(
            {
                "seniority": seniority,
                "region": region,
                "workplace_type": workplace,
                "technologies": ["python"],
                "frameworks": ["pytorch", "sklearn"],
                "cloud": ["aws"],
                "domains": ["mlops"],
                "salary_min": salary - 5000,
                "salary_max": salary + 5000,
            }
        )
    return pd.DataFrame(rows)


def test_train_returns_pipeline_and_report() -> None:
    pipeline, report = salary_model.train(_frame(), cv_splits=3)
    assert report.n_train == 60
    assert report.cv_mae > 0
    feat = salary_model._build_features(_frame()).iloc[:3][salary_model._FEATURES]
    assert pipeline.predict(feat).shape == (3,)


def test_train_requires_minimum_rows() -> None:
    df = _frame(10)
    with pytest.raises(ValueError, match="need >="):
        salary_model.train(df)


def test_save_and_load_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pipeline, _ = salary_model.train(_frame(), cv_splits=3)
    path = salary_model.save(pipeline, tmp_path / "m.joblib")
    loaded = salary_model.load(path)
    pred = salary_model.predict(loaded, seniority="senior", region="London")
    assert pred > 0


def test_predict_uses_sensible_defaults() -> None:
    pipeline, _ = salary_model.train(_frame(), cv_splits=3)
    pred = salary_model.predict(pipeline)
    assert pred > 0
