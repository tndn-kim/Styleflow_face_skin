"""Evaluation utilities for the Palette-Aware Personal Color Classifier."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score,
)

from config import OUTPUTS_DIR, SEASON_LABELS


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: list[str],
    label: str = "",
) -> dict:
    """
    Compute and print a full evaluation report.

    Parameters
    ----------
    y_true     : integer class labels (ground truth)
    y_pred     : integer class labels (predicted)
    y_proba    : [N, C] probability array, or None
    class_names: ordered list of class name strings
    label      : human-readable model name for printing

    Returns
    -------
    dict with scalar metrics
    """
    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    top2_acc: Optional[float] = None
    if y_proba is not None and y_proba.shape[1] >= 2:
        try:
            top2_acc = top_k_accuracy_score(y_true, y_proba, k=2)
        except Exception:
            pass

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    if label:
        hdr = f"  ─── {label} ───"
        print(f"\n{hdr}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro  F1 : {f1_macro:.4f}")
    print(f"  Wtd    F1 : {f1_weight:.4f}")
    if top2_acc is not None:
        print(f"  Top-2 Acc : {top2_acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("  Confusion matrix (rows=True, cols=Pred):")
    _print_cm(cm, class_names)

    return {
        "accuracy":        acc,
        "f1_macro":        f1_macro,
        "f1_weighted":     f1_weight,
        "top2_accuracy":   top2_acc,
        "report":          report,
        "confusion_matrix": cm.tolist(),
    }


def _print_cm(cm: np.ndarray, class_names: list[str]) -> None:
    w = max(len(c) for c in class_names) + 2
    header = "  " + "".join(f"{c:>{w}}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = "  " + f"{row_name:<{w}}"
        for val in cm[i]:
            row_str += f"{val:>{w}}"
        print(row_str)


# ─── Model comparison ─────────────────────────────────────────────────────────

def compare_models(results: dict[str, dict], output_path: Optional[Path] = None) -> None:
    """Print a ranked comparison table and save to JSON."""
    rows = sorted(
        [(name, r["test"]["f1_macro"], r["test"]["accuracy"])
         for name, r in results.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n" + "=" * 52)
    print(f"  {'Model':<20}  {'Macro F1':>10}  {'Accuracy':>10}")
    print("  " + "-" * 44)
    for name, f1, acc in rows:
        print(f"  {name:<20}  {f1:>10.4f}  {acc:>10.4f}")
    print("=" * 52)

    if output_path is None:
        output_path = OUTPUTS_DIR / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[eval] Comparison saved → {output_path}")


# ─── Misclassified report ─────────────────────────────────────────────────────

def save_misclassified(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: list[str],
    output_path: Optional[Path] = None,
) -> None:
    """
    Save a CSV of misclassified test samples with distances and probabilities.
    """
    wrong = y_true != y_pred
    if not wrong.any():
        print("[eval] No misclassifications on test set.")
        return

    # Discover distance columns dynamically (works for both original and 4-class)
    dist_cols = [c for c in df_test.columns if c.startswith("dist_to_")]

    rows = []
    for i in np.where(wrong)[0]:
        row: dict = {
            "image_path": df_test.iloc[i].get("image_path", ""),
            "true_label": class_names[y_true[i]],
            "pred_label": class_names[y_pred[i]],
        }
        if y_proba is not None:
            row["pred_prob"] = float(y_proba[i, y_pred[i]])
            top2_idx = np.argsort(y_proba[i])[::-1][:2]
            row["top2_label"] = class_names[top2_idx[1]] if len(top2_idx) > 1 else ""
            row["top2_prob"]  = float(y_proba[i, top2_idx[1]]) if len(top2_idx) > 1 else float("nan")

        for col in dist_cols:
            row[col] = float(df_test.iloc[i][col]) if col in df_test.columns else float("nan")

        rows.append(row)

    mis_df = pd.DataFrame(rows)
    if output_path is None:
        output_path = OUTPUTS_DIR / "misclassified.csv"
    mis_df.to_csv(output_path, index=False)
    print(f"[eval] {len(rows)} misclassified samples → {output_path}")


# ─── Save confusion matrix ────────────────────────────────────────────────────

def save_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    output_path: Optional[Path] = None,
) -> None:
    if output_path is None:
        output_path = OUTPUTS_DIR / "confusion_matrix.csv"
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(output_path)
    print(f"[eval] Confusion matrix → {output_path}")


# ─── Classification report text ──────────────────────────────────────────────

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str = "",
    output_path: Optional[Path] = None,
) -> None:
    if output_path is None:
        output_path = OUTPUTS_DIR / "classification_report.txt"
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(output_path, "w") as f:
        if model_name:
            f.write(f"Best model: {model_name}\n\n")
        f.write(report)
    print(f"[eval] Classification report → {output_path}")
