"""Static figures for the README and the Streamlit app."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from .predict import global_importance

BLUE, GREY, ORANGE = "#2563eb", "#94a3b8", "#f97316"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


def _save(fig, name: str) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def make_figures(model, winner, X_tr, X_te, y_te, p_te, threshold, metrics, oof, y_tr) -> None:
    # 1. Model comparison (CV PR-AUC and ROC-AUC)
    comp = pd.DataFrame(metrics["cv_comparison"]).T.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(comp))
    ax.barh(y - 0.2, comp["roc_auc"], 0.4, label="ROC-AUC", color=BLUE)
    ax.barh(y + 0.2, comp["pr_auc"], 0.4, label="PR-AUC", color=ORANGE)
    ax.set_yticks(y, comp.index)
    ax.set_xlim(0.3, 0.9)
    ax.set_title("5-fold cross-validation on the training split")
    ax.legend(loc="lower right")
    _save(fig, "model_comparison.png")

    # 2. ROC + PR curves on the held-out test split
    fpr, tpr, _ = roc_curve(y_te, p_te)
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"{winner} (AUC {metrics['holdout']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color=GREY, label="Random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (hold-out)")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, color=ORANGE, lw=2, label=f"{winner} (AP {metrics['holdout']['pr_auc']:.3f})")
    axes[1].axhline(metrics["positive_rate"], ls="--", color=GREY, label="Random (base rate)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (hold-out)")
    axes[1].legend(loc="upper right")
    _save(fig, "roc_pr_curves.png")

    # 3. Confusion matrix at the CV-selected threshold
    cm = metrics["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mat, cmap="Blues")
    ax.grid(False)
    labels = ["Not looking", "Looking"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=13)
    _save(fig, "confusion_matrix.png")

    # 4. Cumulative gain: how many "looking" candidates are found by screening the top x%
    order = np.argsort(-p_te)
    gain = np.cumsum(np.asarray(y_te)[order]) / np.sum(y_te)
    frac = np.arange(1, len(order) + 1) / len(order)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(frac * 100, gain * 100, color=BLUE, lw=2, label="Model ranking")
    ax.plot([0, 100], [0, 100], "--", color=GREY, label="Random ranking")
    ax.set(xlabel="% of candidates screened (highest score first)",
           ylabel="% of true movers found", title="Cumulative gain (hold-out)")
    ax.legend(loc="lower right")
    _save(fig, "cumulative_gain.png")

    # 5. Global feature importance (mean |SHAP|)
    imp = global_importance(model, X_te.sample(min(len(X_te), 1500), random_state=C.RANDOM_STATE))
    imp.to_csv(C.MODELS_DIR / "feature_importance.csv", header=["mean_abs_shap"])
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top.index, top.values, color=BLUE)
    ax.set(xlabel="Mean |SHAP value|", title="What drives the prediction")
    _save(fig, "feature_importance.png")
