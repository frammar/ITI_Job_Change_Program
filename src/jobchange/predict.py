"""Inference helpers shared by the CLI, the tests and the Streamlit app."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap

from . import config as C

_FRIENDLY = {
    "city_development_index": "City development index",
    "education_level": "Education level",
    "experience": "Years of experience",
    "company_size": "Current company size",
    "last_new_job": "Years since last job change",
    "training_hours": "Training hours",
    "log_training_hours": "Training hours (log)",
    "relevent_experience": "Relevant experience",
    "is_unemployed": "No current employer info",
    "n_missing": "Number of missing fields",
}


def friendly(name: str) -> str:
    if name in _FRIENDLY:
        return _FRIENDLY[name]
    for prefix in ("city", "enrolled_university", "major_discipline", "company_type"):
        if name.startswith(prefix + "_"):
            return f"{prefix.replace('_', ' ').capitalize()}: {name[len(prefix) + 1:]}"
    return name


def load_bundle(path=C.MODEL_PATH) -> dict:
    return joblib.load(path)


def score(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Return priority score (probability of looking for a new job) and a recommendation."""
    proba = bundle["model"].predict_proba(df)[:, 1]
    out = pd.DataFrame({
        "priority_score": proba.round(4),
        "recommendation": np.where(proba >= bundle["threshold"], "Advance to next stage", "Lower priority"),
    }, index=df.index)
    if C.ID_COL in df.columns:
        out.insert(0, C.ID_COL, df[C.ID_COL].values)
    return out


def _shap_values(model, df: pd.DataFrame):
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(df)
    values = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(values, list):  # older shap API: [class0, class1]
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return Xt, values


def global_importance(model, df: pd.DataFrame) -> pd.Series:
    Xt, values = _shap_values(model, df)
    s = pd.Series(np.abs(values).mean(axis=0), index=[friendly(c) for c in Xt.columns])
    return s.groupby(level=0).sum().sort_values(ascending=False)


def explain_candidate(bundle: dict, row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Per-feature contribution (in model log-odds) for one candidate."""
    Xt, values = _shap_values(bundle["model"], row)
    contrib = pd.Series(values[0], index=[friendly(c) for c in Xt.columns])
    contrib = contrib.groupby(level=0).sum()
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return top.rename("contribution").to_frame()
