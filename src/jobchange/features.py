"""Cleaning and feature engineering, wrapped in an sklearn pipeline so imputation and
encoding are only fitted on training data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .config import EDU_ORDER, RAW_FEATURES, SIZE_ORDER

NUMERIC_FEATURES = [
    "city_development_index",
    "education_level",
    "experience",
    "company_size",
    "last_new_job",
    "training_hours",
    "log_training_hours",
    "relevent_experience",
    "is_unemployed",
    "n_missing",
]
CATEGORICAL_FEATURES = ["city", "enrolled_university", "major_discipline", "company_type"]

_EDU_MAP = {v: i + 1 for i, v in enumerate(EDU_ORDER)}
_SIZE_MAP = {v: i + 1 for i, v in enumerate(SIZE_ORDER)}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataset columns -> model-ready frame."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df[RAW_FEATURES].copy()

    # both company fields missing -> probably not currently employed
    d["is_unemployed"] = (d["company_size"].isna() & d["company_type"].isna()).astype(int)
    d["n_missing"] = d.isna().sum(axis=1)

    d["experience"] = pd.to_numeric(
        d["experience"].astype("object").replace({"<1": "0", ">20": "21"}), errors="coerce"
    )
    d["last_new_job"] = pd.to_numeric(
        d["last_new_job"].astype("object").replace({"never": "0", ">4": "5"}), errors="coerce"
    )
    d["education_level"] = d["education_level"].map(_EDU_MAP).astype(float)
    # 0 means unknown
    d["company_size"] = d["company_size"].map(_SIZE_MAP).fillna(0).astype(float)
    d["relevent_experience"] = (d["relevent_experience"] == "Has relevent experience").astype(int)
    d["log_training_hours"] = np.log1p(pd.to_numeric(d["training_hours"], errors="coerce"))

    for col in ["enrolled_university", "major_discipline", "company_type", "city"]:
        d[col] = d[col].astype("object").where(d[col].notna(), "Unknown")

    return d[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False
    )
    ct = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("clean", FunctionTransformer(clean, validate=False)), ("encode", ct)])
