import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobchange import config as C
from jobchange.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, clean
from jobchange.predict import explain_candidate, load_bundle, score


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(C.TRAIN_CSV)


def test_clean_parses_special_values():
    df = pd.DataFrame([{
        "city": "city_1", "city_development_index": 0.9, "relevent_experience": "No relevent experience",
        "enrolled_university": None, "education_level": "Phd", "major_discipline": None,
        "experience": ">20", "company_size": None, "company_type": None,
        "last_new_job": "never", "training_hours": 10,
    }])
    out = clean(df).iloc[0]
    assert out["experience"] == 21 and out["last_new_job"] == 0
    assert out["education_level"] == 5 and out["relevent_experience"] == 0
    assert out["is_unemployed"] == 1 and out["company_type"] == "Unknown"
    assert list(clean(df).columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_clean_requires_columns():
    with pytest.raises(ValueError):
        clean(pd.DataFrame({"city": ["x"]}))


def test_preprocessor_has_no_nans_and_ignores_gender(raw):
    prep = build_preprocessor().fit(raw)
    Xt = prep.transform(raw)
    assert not Xt.isna().any().any()
    assert not any("gender" in c for c in Xt.columns)


@pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="model not trained")
def test_trained_model_scores_and_explains(raw):
    bundle = load_bundle()
    res = score(bundle, raw.head(50))
    assert res["priority_score"].between(0, 1).all()
    assert set(res["recommendation"]) <= {"Advance to next stage", "Lower priority"}
    assert not explain_candidate(bundle, raw.head(1)).empty


@pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="model not trained")
def test_model_beats_random_on_training_data(raw):
    from sklearn.metrics import roc_auc_score
    p = load_bundle()["model"].predict_proba(raw)[:, 1]
    assert roc_auc_score(raw[C.TARGET], p) > 0.75
