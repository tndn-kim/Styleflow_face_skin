"""Phase 6: Final single-image inference API.

Loads the artefacts saved by train.py's --save-final-model-bundle into one
in-memory bundle, then runs the full production pipeline for a single
image: feature extraction -> palette distance features -> base 4-class
model -> margin-gated pairwise reranker -> warm/cool probability ->
confidence-based boundary output -> a single JSON-able result dict.

Usage
-----
    from final_inference import load_final_model_bundle, predict_personal_color
    bundle = load_final_model_bundle("outputs/final_model_bundle")
    result = predict_personal_color("photo.jpg", bundle)

CLI
---
    python final_inference.py --bundle outputs/final_model_bundle --image photo.jpg
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import CLASS_DISPLAY_NAMES
from extract_person_features import (
    extract_features_from_image, _make_detector,
    add_palette_distances, add_axis_distances,
)
from extract_palette_features import (
    load_prototypes, load_axis_prototypes,
    load_prototypes_4class, load_axis_prototypes_4class,
)
from pairwise_specialists import get_specialist, specialist_predict_row
from warm_cool import get_warm_cool_probs
from boundary import classify_boundary_type


# ─── Bundle loading ─────────────────────────────────────────────────────────────

def load_final_model_bundle(bundle_dir: str | Path) -> dict:
    """Load every artefact written by train.py's final-model-bundle builder
    into one dict, ready for predict_personal_color()."""
    bundle_dir = Path(bundle_dir)

    with open(bundle_dir / "base_model.pkl", "rb") as f:
        base = pickle.load(f)
    with open(bundle_dir / "feature_columns.json", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(bundle_dir / "label_mapping.json", encoding="utf-8") as f:
        label_mapping = json.load(f)

    proto  = load_prototypes(bundle_dir / "palette_prototypes.json")
    axis   = load_axis_prototypes(bundle_dir / "palette_axis_prototypes.json")
    proto4 = load_prototypes_4class(bundle_dir / "palette_prototypes_4class.json")
    axis4  = load_axis_prototypes_4class(bundle_dir / "palette_axis_prototypes_4class.json")

    specialists: dict = {}
    spec_dir = bundle_dir / "pairwise_specialists"
    if spec_dir.exists():
        for pkl in sorted(spec_dir.glob("*.pkl")):
            with open(pkl, "rb") as f:
                specialists[pkl.stem] = pickle.load(f)

    wc_bundle = None
    wc_path = bundle_dir / "warm_cool_model.pkl"
    if wc_path.exists():
        with open(wc_path, "rb") as f:
            wc_bundle = pickle.load(f)

    with open(bundle_dir / "selected_thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)
    with open(bundle_dir / "inference_config.json", encoding="utf-8") as f:
        inference_config = json.load(f)

    return {
        "base_model":                    base["model"],
        "label_encoder":                 base["label_encoder"],
        "feature_cols":                  feature_cols,
        "label_mapping":                 label_mapping,
        "palette_prototypes":            proto,
        "palette_axis_prototypes":       axis,
        "palette_prototypes_4class":     proto4,
        "palette_axis_prototypes_4class": axis4,
        "specialists":                   specialists,
        "wc_bundle":                     wc_bundle,
        "thresholds":                    thresholds,
        "inference_config":              inference_config,
    }


# ─── Single-image inference ─────────────────────────────────────────────────────

def predict_personal_color(
    image_path: str | Path,
    bundle: dict,
    return_debug: bool = False,
) -> dict:
    """
    Run the full final-policy inference pipeline on one image.

    Returns a dict matching the schema in outputs/final_inference_schema.json
    (see boundary.py's output_type vocabulary: single / boundary_top2 /
    warm_cool_boundary / low_confidence).
    """
    cfg = bundle["inference_config"]
    wb  = cfg.get("white_balance", "none")

    detector = _make_detector()
    try:
        feats = extract_features_from_image(image_path, detector, wb=wb)
    finally:
        detector.close()

    if feats is None:
        return {"error": "no_face_detected", "image_path": str(image_path)}

    feats["image_path"] = str(image_path)
    df_one = pd.DataFrame([feats])
    # Mirror train.py's exact call order (original season prototypes, then
    # 4-class prototypes) so the same column set the model was trained on
    # — dist_to_spring.. AND dist_to_spring_warm.. — gets reproduced here.
    df_one = add_palette_distances(df_one, bundle["palette_prototypes"])
    df_one = add_axis_distances(df_one, bundle["palette_axis_prototypes"])
    df_one = add_palette_distances(df_one, bundle["palette_prototypes_4class"])
    df_one = add_axis_distances(df_one, bundle["palette_axis_prototypes_4class"])

    feature_cols = bundle["feature_cols"]
    # Any column the base model expects but extraction didn't produce
    # becomes NaN (the model's own imputer handles it the same way it does
    # at training time).
    X = df_one.reindex(columns=feature_cols).values.astype(np.float32)

    model = bundle["base_model"]
    le = bundle["label_encoder"]
    class_names = list(le.classes_)
    proba = model.predict_proba(X)[0]

    order = np.argsort(proba)[::-1]
    top1_idx, top2_idx = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    top1_name, top2_name = class_names[top1_idx], class_names[top2_idx]
    top1_prob, top2_prob = float(proba[top1_idx]), float(proba[top2_idx])
    margin = top1_prob - top2_prob

    thr = bundle["thresholds"]
    final_policy = cfg.get("final_policy", "margin_pairwise")

    final_name = top1_name
    if final_policy == "margin_pairwise" and bundle["specialists"]:
        specialist = get_specialist(bundle["specialists"], top1_name, top2_name)
        if margin < thr["pairwise_margin_threshold"] and specialist is not None:
            sp_pred, sp_prob = specialist_predict_row(specialist, df_one.iloc[0])
            if sp_prob >= thr["pairwise_confidence_threshold"] and sp_pred in class_names:
                final_name = sp_pred

    warm_prob = cool_prob = float("nan")
    wc_confidence = float("nan")
    if bundle.get("wc_bundle") is not None:
        warm_probs, cool_probs = get_warm_cool_probs(df_one, bundle["wc_bundle"])
        warm_prob, cool_prob = float(warm_probs[0]), float(cool_probs[0])
        wc_confidence = max(warm_prob, cool_prob)

    output_type = classify_boundary_type(
        top1_prob, margin, wc_confidence,
        thr.get("boundary_min_confidence", 0.45),
        thr.get("boundary_margin_threshold", 0.08),
        thr.get("warm_cool_boundary_threshold", 0.55),
    )
    is_boundary = output_type != "single"

    display = CLASS_DISPLAY_NAMES
    tone_direction = "warm" if (not np.isnan(warm_prob) and warm_prob >= cool_prob) else "cool"
    if np.isnan(wc_confidence):
        confidence_level = "unknown"
    elif wc_confidence >= 0.65:
        confidence_level = "high"
    elif wc_confidence >= 0.55:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    notes: list[str] = []
    if margin < thr.get("boundary_margin_threshold", 0.08):
        notes.append(f"{display.get(top1_name, top1_name)}와 {display.get(top2_name, top2_name)} 후보가 가까운 편입니다.")
    if not np.isnan(wc_confidence) and wc_confidence < thr.get("warm_cool_boundary_threshold", 0.55):
        side = "쿨" if tone_direction == "cool" else "웜"
        notes.append(f"{side} 쪽으로 약간 기울어져 있습니다.")

    result: dict = {
        "final_label":   None if output_type == "boundary_top2" else final_name,
        "display_name":  None if output_type == "boundary_top2" else display.get(final_name, final_name),
        "output_type":   output_type,
        "top1": {"label": top1_name, "display_name": display.get(top1_name, top1_name), "prob": top1_prob},
        "top2": {"label": top2_name, "display_name": display.get(top2_name, top2_name), "prob": top2_prob},
        "margin": margin,
        "warm_cool": {"warm_prob": warm_prob, "cool_prob": cool_prob, "confidence": wc_confidence},
        "is_boundary": is_boundary,
        "explanation": {
            "tone_direction":   tone_direction,
            "confidence_level": confidence_level,
            "notes":            notes,
        },
    }

    if output_type == "boundary_top2":
        result["candidates"] = [top1_name, top2_name]
        result["message"] = (f"{display.get(top1_name, top1_name)}과 "
                              f"{display.get(top2_name, top2_name)} 경계형으로 보입니다.")

    if return_debug:
        result["debug"] = {
            "raw_features": {k: v for k, v in feats.items() if k != "image_path"},
            "all_probs": {class_names[i]: float(proba[i]) for i in range(len(class_names))},
        }
    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    # Windows consoles/redirected files default to the system codepage
    # (cp949 on Korean Windows), which mangles the Korean display_name /
    # explanation text in our JSON output when piped to a file (`> out.json`).
    # Force UTF-8 on stdout so the JSON is valid regardless of locale.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="Final personal-color inference for a single image")
    p.add_argument("--bundle", required=True, help="Path to outputs/final_model_bundle")
    p.add_argument("--image",  required=True, help="Path to a face image")
    p.add_argument("--debug",  action="store_true", help="Include raw features + all class probabilities")
    args = p.parse_args()

    bundle = load_final_model_bundle(args.bundle)
    result = predict_personal_color(args.image, bundle, return_debug=args.debug)
    print(json.dumps(result, indent=2, ensure_ascii=False))
