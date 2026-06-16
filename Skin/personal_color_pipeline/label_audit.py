"""Phase 6: High-confidence-wrong audit workflow.

Turns Phase 5's high_confidence_wrong export into something a human can
actually act on:

  1. run_label_audit_workflow() — export high-confidence wrong samples
     (4-class + warm/cool), copy a sample of the actual images, and build
     a review_template CSV with empty review_status/corrected_label/
     remove_image/review_note columns for a human to fill in.

  2. apply_audit_corrections() — once a human has filled in the template,
     turn it into machine-readable artefacts the next training run can
     consume: a manifest of what changed, an excluded_images.txt, and a
     label_overrides.json. The original dataset is never touched directly
     — these are opt-in manifests for the *next* run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from config import OUTPUTS_DIR, HIGH_CONFIDENCE_THRESHOLD, LABEL_AUDIT_COUNT
from audit import export_high_confidence_wrong, export_label_audit_samples

LABEL_AUDIT_OUT_DIR = OUTPUTS_DIR / "label_audit"

_TEMPLATE_COLUMNS = [
    "image_path", "true_label", "pred_label", "pred_prob",
    "top2_label", "top2_prob", "warm_prob", "cool_prob", "error_type",
    "review_status", "corrected_label", "remove_image", "review_note",
]


# ─── Review template ────────────────────────────────────────────────────────

def build_audit_review_template(
    hc_wrong_df: pd.DataFrame,
    top_n: int = LABEL_AUDIT_COUNT,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build outputs/label_audit/audit_review_template.csv from the
    high_confidence_wrong export, keeping only the `top_n` most-confident
    mistakes (highest pred_prob first — those are the most suspicious).
    """
    out_dir = Path(output_dir) if output_dir else LABEL_AUDIT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if hc_wrong_df is None or hc_wrong_df.empty:
        template = pd.DataFrame(columns=_TEMPLATE_COLUMNS)
    else:
        df = hc_wrong_df.sort_values("pred_prob", ascending=False).head(top_n).copy()
        keep = [c for c in ["image_path", "true_label", "pred_label", "pred_prob",
                             "top2_label", "top2_prob", "warm_prob", "cool_prob",
                             "error_type"] if c in df.columns]
        template = df[keep].reset_index(drop=True)
        template["review_status"]   = "pending"
        template["corrected_label"] = ""
        template["remove_image"]    = False
        template["review_note"]     = ""
        template = template[_TEMPLATE_COLUMNS]

    path = out_dir / "audit_review_template.csv"
    template.to_csv(path, index=False)
    print(f"[label_audit] Review template ({len(template)} rows) -> {path}")
    return template


def run_label_audit_workflow(
    base_probs, y_true, class_names, df_test, wc_bundle=None,
    audit_top_n: int = LABEL_AUDIT_COUNT,
    audit_min_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
    boundary_case_log: Optional[list[dict]] = None,
) -> dict:
    """
    Full label-audit export: high-confidence-wrong CSVs (re-saved under
    outputs/label_audit/ to keep this workflow self-contained), a copy of
    the flagged images, and the human review template.
    """
    out_dir = LABEL_AUDIT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    hc_wrong_df, hc_wc_wrong_df = export_high_confidence_wrong(
        base_probs, y_true, class_names, df_test, wc_bundle,
        high_confidence_threshold=audit_min_confidence,
    )
    # Mirror the Phase 5 CSVs into the Phase 6 audit folder so everything
    # relevant to the audit workflow lives in one place.
    hc_wrong_df.to_csv(out_dir / "high_confidence_wrong.csv", index=False)
    hc_wc_wrong_df.to_csv(out_dir / "high_confidence_warm_cool_wrong.csv", index=False)

    template = build_audit_review_template(hc_wrong_df, top_n=audit_top_n, output_dir=out_dir)

    meta_df = export_label_audit_samples(
        hc_wrong_df, boundary_case_log, audit_count=audit_top_n,
        output_dir=out_dir / "audit_samples",
    )

    print(f"[label_audit] Workflow complete: {len(hc_wrong_df)} high-confidence wrong, "
          f"{len(template)} in review template, {len(meta_df)} images copied -> {out_dir}")
    return {
        "high_confidence_wrong": hc_wrong_df,
        "high_confidence_wc_wrong": hc_wc_wrong_df,
        "review_template": template,
        "copied_samples": meta_df,
    }


# ─── Corrections ─────────────────────────────────────────────────────────────

def _is_truthy(val) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes", "y")


def _clean_str_cell(val) -> str:
    """Safely stringify a pandas cell that may be NaN (float) because the
    CSV cell was empty — pd.read_csv round-trips an empty `corrected_label`
    column as float64 NaN, and naive str(val) would yield the literal
    string "nan", which would be treated as a real corrected label."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def apply_audit_corrections(
    template_path: str | Path,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Read a filled-in audit_review_template.csv and turn it into:
      audit_corrections_manifest.csv  — every row's resulting action
      excluded_images.txt             — image_paths with remove_image=true
      label_overrides.json            — {image_path: corrected_label}

    Does NOT touch the original dataset/images — these are manifests for
    the next training run to optionally consume (filter excluded_images,
    remap label_overrides before building feat_cols/label_col).
    """
    out_dir = Path(output_dir) if output_dir else LABEL_AUDIT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(template_path)
    manifest_rows, overrides, excluded = [], {}, []

    for _, r in df.iterrows():
        path      = str(r.get("image_path", ""))
        true_lbl  = r.get("true_label", "")
        corrected = _clean_str_cell(r.get("corrected_label", ""))
        remove    = _is_truthy(r.get("remove_image", False))

        if remove:
            excluded.append(path)
            action = "exclude"
        elif corrected and corrected != true_lbl:
            overrides[path] = corrected
            action = "override"
        else:
            action = "none"

        manifest_rows.append({
            "image_path":       path,
            "action":           action,
            "original_label":   true_lbl,
            "corrected_label":  corrected,
            "review_status":    r.get("review_status", ""),
            "review_note":      r.get("review_note", ""),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "audit_corrections_manifest.csv", index=False)

    with open(out_dir / "excluded_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(excluded))

    with open(out_dir / "label_overrides.json", "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)

    print(f"[label_audit] Corrections applied: {len(overrides)} overrides, "
          f"{len(excluded)} excluded -> {out_dir}")
    return {"overrides": overrides, "excluded": excluded, "manifest": manifest_df}
