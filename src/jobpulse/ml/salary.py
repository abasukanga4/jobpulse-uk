"""Salary regression model.

Predicts the salary mid-point from the structured signals we already extract:
seniority, workplace type, region, skill counts. No text features yet — those
go in v0.2 once we have enough Adzuna data to make embeddings pay off.

The XGBoost model lives behind a scikit-learn Pipeline so we can swap it for
LightGBM or a baseline LinearRegression without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

from jobpulse.models import Seniority

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()

_SENIORITY_ORDER = [
    Seniority.UNKNOWN.value,
    Seniority.JUNIOR.value,
    Seniority.MID.value,
    Seniority.SENIOR.value,
    Seniority.LEAD.value,
    Seniority.PRINCIPAL.value,
]

_CATEGORICAL = ["workplace_type", "region"]
_ORDINAL = ["seniority"]
_NUMERIC = ["n_technologies", "n_frameworks", "n_cloud", "n_domains"]
_FEATURES = _CATEGORICAL + _ORDINAL + _NUMERIC
_TARGET = "salary_mid"


@dataclass
class SalaryModelReport:
    n_train: int
    cv_mae: float
    cv_r2: float
    feature_importance: dict[str, float]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model features from a joined jobs+skills DataFrame."""
    out = df.copy()
    out["salary_mid"] = out[["salary_min", "salary_max"]].mean(axis=1)
    for col in ("technologies", "frameworks", "cloud", "domains"):
        out[f"n_{col}"] = out[col].apply(
            lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0
        )
    out["seniority"] = out["seniority"].fillna(Seniority.UNKNOWN.value)
    out["workplace_type"] = out["workplace_type"].fillna("unknown")
    out["region"] = out["region"].fillna("unknown")
    return out


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CATEGORICAL,
            ),
            (
                "ord",
                OrdinalEncoder(
                    categories=[_SENIORITY_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                _ORDINAL,
            ),
            ("num", "passthrough", _NUMERIC),
        ]
    )
    return Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=0,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def train(df: pd.DataFrame, *, cv_splits: int = 5) -> tuple[Pipeline, SalaryModelReport]:
    """Train and cross-validate on a joined jobs+skills DataFrame."""
    feat = _build_features(df).dropna(subset=[_TARGET])
    if len(feat) < 20:
        raise ValueError(f"need >=20 jobs with salary to train, got {len(feat)}")
    x = feat[_FEATURES]
    y = feat[_TARGET]

    pipeline = _build_pipeline()
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=0)
    cv_mae = -cross_val_score(pipeline, x, y, cv=kf, scoring="neg_mean_absolute_error").mean()
    cv_r2 = cross_val_score(pipeline, x, y, cv=kf, scoring="r2").mean()

    pipeline.fit(x, y)
    pred = pipeline.predict(x)
    importance = _importance(pipeline)
    logger.info(
        "salary.train.done",
        n=len(feat),
        cv_mae=round(float(cv_mae), 0),
        cv_r2=round(float(cv_r2), 3),
        train_mae=round(float(mean_absolute_error(y, pred)), 0),
        train_r2=round(float(r2_score(y, pred)), 3),
    )
    report = SalaryModelReport(
        n_train=len(feat),
        cv_mae=float(cv_mae),
        cv_r2=float(cv_r2),
        feature_importance=importance,
    )
    return pipeline, report


def _importance(pipeline: Pipeline) -> dict[str, float]:
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out().tolist()
    importances = model.feature_importances_
    pairs = sorted(zip(names, importances, strict=True), key=lambda t: -t[1])
    return {name: float(score) for name, score in pairs[:15]}


def save(pipeline: Pipeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load(path: Path) -> Pipeline:
    return joblib.load(path)


def predict(pipeline: Pipeline, **inputs: object) -> float:
    """Predict a single salary mid-point. Convenience for the dashboard."""
    defaults = {
        "workplace_type": "unknown",
        "region": "London",
        "seniority": Seniority.MID.value,
        "n_technologies": 2,
        "n_frameworks": 2,
        "n_cloud": 1,
        "n_domains": 1,
    }
    defaults.update(inputs)
    row = pd.DataFrame([{k: defaults[k] for k in _FEATURES}])
    return float(pipeline.predict(row)[0])
