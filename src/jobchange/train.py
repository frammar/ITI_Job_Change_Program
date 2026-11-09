"""Train, compare, evaluate and persist the candidate-screening model.

Usage:  python -m jobchange.train
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C
from .features import build_preprocessor


def candidate_models(pos_weight: float) -> dict:
    return {
        "Decision Tree (baseline)": DecisionTreeClassifier(
            max_depth=7, class_weight="balanced", random_state=C.RANDOM_STATE
        ),
        "Logistic Regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=C.RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
            scale_pos_weight=pos_weight, eval_metric="logloss", n_jobs=-1,
            random_state=C.RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=30, reg_lambda=2.0,
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=C.RANDOM_STATE, verbose=-1,
        ),
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def precision_at_k(y_true, proba, k: int) -> float:
    order = np.argsort(-np.asarray(proba))[:k]
    return float(np.asarray(y_true)[order].mean())


def scores(y_true, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "threshold": thr,
    }


def main() -> None:
    t0 = time.time()
    C.MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(C.TRAIN_CSV)
    y = df[C.TARGET].astype(int)
    X = df.drop(columns=[C.TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"train={len(X_tr):,} test={len(X_te):,} positive rate={y.mean():.1%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    comparison, oof = {}, {}
    for name, clf in candidate_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        proba = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        thr, _ = best_f1_threshold(y_tr, proba)
        oof[name] = (proba, thr)
        comparison[name] = scores(y_tr, proba, thr)
        print(f"  CV {name:<26} ROC-AUC={comparison[name]['roc_auc']:.4f} "
              f"PR-AUC={comparison[name]['pr_auc']:.4f} F1={comparison[name]['f1']:.4f}")

    # Pick the model by cross-validated PR-AUC (classes are 75/25).
    winner = max(comparison, key=lambda n: comparison[n]["pr_auc"])
    proba_oof, threshold = oof[winner]
    print(f"Selected: {winner} (threshold from CV = {threshold:.3f})")

    # Fit every model on the full training split for an apples-to-apples held-out comparison.
    holdout = {}
    fitted = {}
    for name, clf in candidate_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)]).fit(X_tr, y_tr)
        fitted[name] = pipe
        holdout[name] = scores(y_te, pipe.predict_proba(X_te)[:, 1], oof[name][1])

    model = fitted[winner]
    p_te = model.predict_proba(X_te)[:, 1]
    pred_te = (p_te >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred_te).ravel()
    base_rate = float(y_te.mean())
    ks = [10, 50, 100, 250, 500]
    p_at_k = {str(k): precision_at_k(y_te, p_te, k) for k in ks}
    top_decile = int(len(y_te) * 0.1)
    lift_top_decile = precision_at_k(y_te, p_te, top_decile) / base_rate

    # Score the unlabeled Kaggle test file -> ranked shortlist + submission.
    new = pd.read_csv(C.TEST_CSV)
    new_proba = model.predict_proba(new)[:, 1]
    ranked = pd.DataFrame({
        C.ID_COL: new[C.ID_COL],
        "priority_score": new_proba.round(4),
        "recommendation": np.where(new_proba >= threshold, "Advance to next stage", "Lower priority"),
    }).sort_values("priority_score", ascending=False).reset_index(drop=True)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    ranked.to_csv(C.REPORTS_DIR / "scored_candidates.csv", index=False)
    ranked.head(10).to_csv(C.REPORTS_DIR / "top10_candidates.csv", index=False)
    pd.DataFrame({C.ID_COL: new[C.ID_COL], C.TARGET: new_proba.round(4)}).to_csv(
        C.REPORTS_DIR / "submission.csv", index=False
    )

    bundle = {
        "model": model,
        "threshold": threshold,
        "model_name": winner,
        "feature_columns": C.RAW_FEATURES,
        "train_pos_rate": float(y_tr.mean()),
    }
    joblib.dump(bundle, C.MODEL_PATH, compress=3)

    metrics = {
        "selected_model": winner,
        "threshold": threshold,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "positive_rate": base_rate,
        "cv_comparison": comparison,
        "holdout_comparison": holdout,
        "holdout": scores(y_te, p_te, threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_at_k": p_at_k,
        "lift_top_decile": lift_top_decile,
        "scored_test_candidates": int(len(ranked)),
        "recommended_to_advance": int((ranked["recommendation"] == "Advance to next stage").sum()),
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # figures and feature importance
    from .reporting import make_figures
    make_figures(model, winner, X_tr, X_te, y_te, p_te, threshold, metrics, oof, y_tr)

    h = metrics["holdout"]
    print(f"\nHold-out ({winner}): ROC-AUC={h['roc_auc']:.3f} PR-AUC={h['pr_auc']:.3f} "
          f"F1={h['f1']:.3f} P={h['precision']:.3f} R={h['recall']:.3f} lift@10%={lift_top_decile:.2f}x")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
