"""
Feature importance, correlation analysis, and scatter plots
for the Palette-Aware Personal Color Classifier.

Outputs (saved to outputs/analysis/):
  feature_importance.png      -- LightGBM gain-based importance (top 30)
  feature_importance.csv      -- Full ranked importance table
  correlation_heatmap.png     -- Top-N feature × season Pearson correlation
  scatter_key_pairs.png       -- Key feature pair scatter plots by season
  boxplot_top_features.png    -- Box plots of top features per season
  season_lab_3d_pca.png       -- PCA of all features coloured by season
  dist_to_season.png          -- Palette-distance distributions per season
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUTS_DIR, CACHE_DIR, SEASON_LABELS

# ─── Setup ───────────────────────────────────────────────────────────────────
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

SEASON_COLORS = {
    "Spring": "#F4A460",   # sandy warm
    "Summer": "#87CEFA",   # light blue cool
    "Autumn": "#CD853F",   # peru warm dark
    "Winter": "#6A5ACD",   # slate blue cool dark
}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ─── Load data ────────────────────────────────────────────────────────────────

def load_data():
    cache_path = CACHE_DIR / "person_features_cache.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {cache_path}\n"
            "Run train.py first to extract features."
        )
    df = pd.read_csv(cache_path)

    # Add palette distances if not already present
    dist_col = f"dist_to_{SEASON_LABELS[0].lower()}"
    if dist_col not in df.columns:
        proto_path = OUTPUTS_DIR / "palette_prototypes.json"
        if proto_path.exists():
            from extract_palette_features import load_prototypes
            from extract_person_features import add_palette_distances
            prototypes = load_prototypes(proto_path)
            df = add_palette_distances(df, prototypes)
            print(f"[load] Computed palette distances on the fly")

    model_path = OUTPUTS_DIR / "best_model.pkl"
    model_bundle = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_bundle = pickle.load(f)

    return df, model_bundle


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude metadata and label columns)."""
    skip = {"image_path", "season", "subtype", "label_season", "label_subtype"}
    bool_cols = {c for c in df.columns if c.endswith("_valid")}
    return [
        c for c in df.columns
        if c not in skip and c not in bool_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ─── 1. Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(
    model_bundle: dict | None,
    df: pd.DataFrame,
    feat_cols: list[str],
    top_n: int = 30,
) -> None:
    print("[1] Feature importance …")

    if model_bundle is None:
        print("  No model bundle found — training LightGBM for importance only.")
        importances, names = _fit_lgbm_importance(df, feat_cols)
    else:
        model    = model_bundle["model"]
        feat_cols_saved = model_bundle.get("feature_cols", feat_cols)
        # Extract from the pipeline's last step
        clf = model.named_steps["clf"]
        if hasattr(clf, "estimator"):
            clf = clf.estimator
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            names       = feat_cols_saved
        elif hasattr(clf, "coef_"):
            importances = np.mean(np.abs(clf.coef_), axis=0)
            names       = feat_cols_saved
        else:
            importances, names = _fit_lgbm_importance(df, feat_cols)

    imp_df = pd.DataFrame({"feature": names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(ANALYSIS_DIR / "feature_importance.csv", index=False)

    top = imp_df.head(top_n)

    # colour bars by region
    region_colors = {
        "skin": "#E07B54", "hair": "#7B5E3C",
        "eye":  "#4A90D9", "lip":  "#C0395A",
        "delta": "#6AAB5E", "dist": "#9B59B6",
        "face":  "#F0A500", "clear": "#888888",
        "light": "#888888", "min_":  "#9B59B6",
        "palette": "#9B59B6",
    }

    def _bar_color(name: str) -> str:
        for prefix, color in region_colors.items():
            if name.startswith(prefix):
                return color
        return "#AAAAAA"

    colors = [_bar_color(n) for n in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Importance (LightGBM gain)")
    ax.set_title(f"Top-{top_n} Feature Importances")
    ax.tick_params(axis="y", labelsize=8)

    # legend
    seen = {}
    for n, c in zip(top["feature"], colors):
        for prefix, col in region_colors.items():
            if n.startswith(prefix) and prefix not in seen:
                seen[prefix] = col
    handles = [mpatches.Patch(color=c, label=p) for p, c in seen.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    out = ANALYSIS_DIR / "feature_importance.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")
    print(f"  Top-10: {', '.join(imp_df['feature'].head(10).tolist())}")


def _fit_lgbm_importance(df: pd.DataFrame, feat_cols: list[str]):
    from lightgbm import LGBMClassifier
    le = LabelEncoder()
    y  = le.fit_transform(df["label_season"].values)
    X  = df[feat_cols].values.astype(float)
    X  = SimpleImputer(strategy="median").fit_transform(X)
    X  = StandardScaler().fit_transform(X)
    clf = LGBMClassifier(n_estimators=300, verbosity=-1, random_state=42)
    clf.fit(X, y)
    return clf.feature_importances_, feat_cols


# ─── 2. Correlation heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    feat_cols: list[str],
    top_n: int = 25,
) -> None:
    print("[2] Correlation heatmap …")

    # One-hot encode season → numeric dummy
    season_dummies = pd.get_dummies(df["label_season"])[SEASON_ORDER].astype(float)

    X_num = df[feat_cols].copy()
    corr_rows = []
    for col in feat_cols:
        vals = X_num[col].fillna(X_num[col].median())
        row  = {}
        for season in SEASON_ORDER:
            r = np.corrcoef(vals, season_dummies[season])[0, 1]
            row[season] = round(float(r), 4)
        corr_rows.append({"feature": col, **row})

    corr_df = pd.DataFrame(corr_rows).set_index("feature")

    # Rank features by max |correlation| across any season
    corr_df["max_abs"] = corr_df[SEASON_ORDER].abs().max(axis=1)
    top_feats = corr_df.nlargest(top_n, "max_abs").index.tolist()
    corr_df.drop(columns="max_abs", inplace=True)
    corr_df.to_csv(ANALYSIS_DIR / "feature_season_correlation.csv")

    sub = corr_df.loc[top_feats][SEASON_ORDER]

    fig, ax = plt.subplots(figsize=(7, top_n * 0.38 + 1.5))
    sns.heatmap(
        sub, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        linewidths=0.4, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(f"Feature–Season Pearson Correlation (top-{top_n} by |r|)")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(SEASON_ORDER, rotation=0, fontsize=9)
    plt.tight_layout()
    out = ANALYSIS_DIR / "correlation_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")

    # Print top correlations
    for season in SEASON_ORDER:
        top3 = corr_df[season].abs().nlargest(3).index.tolist()
        vals = [f"{f}({corr_df.loc[f, season]:+.2f})" for f in top3]
        print(f"  {season}: {', '.join(vals)}")


# ─── 3. Scatter plots ─────────────────────────────────────────────────────────

def plot_scatter_pairs(df: pd.DataFrame) -> None:
    print("[3] Scatter plots …")

    PAIRS = [
        ("skin_mean_L",      "skin_mean_b",      "Skin L* vs b* (warm/cool)"),
        ("skin_mean_a",      "skin_mean_b",      "Skin a* vs b* (LAB chromaticity)"),
        ("skin_warm_score",  "skin_mean_L",      "Warm score vs Lightness"),
        ("face_contrast_L",  "clear_muted_score","Contrast vs Chroma (clarity)"),
        ("deltaL_skin_hair", "deltaE_skin_eye",  "Skin-Hair ΔL vs Skin-Eye ΔE"),
        ("dist_to_spring",   "dist_to_summer",   "Palette dist: Spring vs Summer"),
        ("dist_to_autumn",   "dist_to_winter",   "Palette dist: Autumn vs Winter"),
        ("hair_mean_L",      "eye_mean_L",       "Hair L* vs Eye L*"),
    ]

    n_rows = (len(PAIRS) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, n_rows * 4.5))
    axes = axes.ravel()

    for ax, (x_col, y_col, title) in zip(axes, PAIRS):
        for season in SEASON_ORDER:
            sub = df[df["label_season"] == season]
            xv  = pd.to_numeric(sub[x_col], errors="coerce") if x_col in sub else None
            yv  = pd.to_numeric(sub[y_col], errors="coerce") if y_col in sub else None
            if xv is None or yv is None:
                continue
            mask = xv.notna() & yv.notna()
            ax.scatter(
                xv[mask], yv[mask],
                c=SEASON_COLORS[season], label=season,
                alpha=0.30, s=12, edgecolors="none",
            )
            # season centroid
            ax.scatter(
                xv[mask].mean(), yv[mask].mean(),
                c=SEASON_COLORS[season], s=120,
                marker="D", edgecolors="black", linewidths=0.8, zorder=5,
            )

        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(PAIRS):]:
        ax.set_visible(False)

    plt.suptitle("Feature Scatter Plots by Season (◆ = centroid)", fontsize=11, y=1.01)
    plt.tight_layout()
    out = ANALYSIS_DIR / "scatter_key_pairs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─── 4. Box plots ─────────────────────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, feat_cols: list[str], top_n: int = 12) -> None:
    print("[4] Box plots …")

    # Pick top features by between-season variance (ANOVA-style)
    season_dummies = pd.get_dummies(df["label_season"])[SEASON_ORDER].astype(float)
    scores = {}
    for col in feat_cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(
            pd.to_numeric(df[col], errors="coerce").median()
        )
        groups = [
            vals[df["label_season"] == s].dropna().values
            for s in SEASON_ORDER
        ]
        overall_mean = np.mean([np.mean(g) for g in groups if len(g)])
        between_var  = np.mean([
            (np.mean(g) - overall_mean) ** 2
            for g in groups if len(g)
        ])
        within_var   = np.mean([np.var(g) for g in groups if len(g)]) + 1e-6
        scores[col]  = between_var / within_var

    top_feats = sorted(scores, key=scores.get, reverse=True)[:top_n]

    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.0))
    axes = axes.ravel()
    palette = {s: SEASON_COLORS[s] for s in SEASON_ORDER}

    for ax, feat in zip(axes, top_feats):
        plot_df = df[["label_season", feat]].copy()
        plot_df[feat] = pd.to_numeric(plot_df[feat], errors="coerce")
        plot_df = plot_df.dropna()
        # Order categories
        plot_df["label_season"] = pd.Categorical(
            plot_df["label_season"], categories=SEASON_ORDER
        )
        sns.boxplot(
            data=plot_df, x="label_season", y=feat,
            hue="label_season", palette=palette, legend=False, ax=ax,
            order=SEASON_ORDER, width=0.55,
            linewidth=0.8, fliersize=2,
        )
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[top_n:]:
        ax.set_visible(False)

    plt.suptitle(f"Top-{top_n} Features by Between-Season Variance", fontsize=11)
    plt.tight_layout()
    out = ANALYSIS_DIR / "boxplot_top_features.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ─── 5. PCA scatter ───────────────────────────────────────────────────────────

def plot_pca(df: pd.DataFrame, feat_cols: list[str]) -> None:
    print("[5] PCA projection …")

    X = df[feat_cols].copy()
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    pca   = PCA(n_components=3, random_state=42)
    comps = pca.fit_transform(X)
    evr   = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (xi, yi, lx, ly) in zip(
        axes,
        [(0, 1, f"PC1 ({evr[0]:.1%})", f"PC2 ({evr[1]:.1%})"),
         (0, 2, f"PC1 ({evr[0]:.1%})", f"PC3 ({evr[2]:.1%})")],
    ):
        for season in SEASON_ORDER:
            mask = df["label_season"].values == season
            ax.scatter(
                comps[mask, xi], comps[mask, yi],
                c=SEASON_COLORS[season], label=season,
                alpha=0.30, s=10, edgecolors="none",
            )
            ax.scatter(
                comps[mask, xi].mean(), comps[mask, yi].mean(),
                c=SEASON_COLORS[season], s=160,
                marker="D", edgecolors="black", linewidths=0.9, zorder=5,
            )
        ax.set_xlabel(lx); ax.set_ylabel(ly)
        ax.legend(fontsize=8, markerscale=1.5)
        ax.grid(alpha=0.2)

    # PCA loadings arrow overlay on PC1 vs PC2
    loadings = pca.components_.T   # [n_features, 3]
    ax0 = axes[0]
    # Scale arrows to span plot
    scale = (comps[:, 0].max() - comps[:, 0].min()) * 0.4
    top_idx = np.argsort(np.abs(loadings[:, 0]) + np.abs(loadings[:, 1]))[-8:]
    for i in top_idx:
        ax0.annotate(
            "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8, alpha=0.5),
        )
        ax0.text(
            loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
            feat_cols[i], fontsize=6, color="black", alpha=0.7,
            ha="center",
        )

    plt.suptitle("PCA of All Features — Personal Colour Seasons (◆ = centroid)", fontsize=11)
    plt.tight_layout()
    out = ANALYSIS_DIR / "pca_projection.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")
    print(f"  Explained variance: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}, PC3={evr[2]:.1%}")


# ─── 6. Palette distance distributions ───────────────────────────────────────

def plot_palette_distances(df: pd.DataFrame) -> None:
    print("[6] Palette distance distributions …")

    dist_cols = [f"dist_to_{s.lower()}" for s in SEASON_LABELS]
    available = [c for c in dist_cols if c in df.columns]
    if not available:
        print("  No palette distance columns found - skipping.")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, dcol in zip(axes, available):
        target_season = dcol.replace("dist_to_", "").capitalize()
        for season in SEASON_ORDER:
            sub   = df[df["label_season"] == season][dcol].dropna()
            color = SEASON_COLORS[season]
            ax.hist(
                sub, bins=40, alpha=0.55, color=color,
                label=season, density=True, edgecolor="none",
            )
        ax.axvline(
            df[dcol].median(), color="black", lw=1, ls="--", alpha=0.5
        )
        ax.set_title(f"Distance to {target_season} prototype", fontsize=9)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density" if ax == axes[0] else "")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.suptitle("Palette Distance Distributions by True Season", fontsize=11)
    plt.tight_layout()
    out = ANALYSIS_DIR / "dist_to_season.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ─── 7. Per-season Lab summary table ─────────────────────────────────────────

def print_season_summary(df: pd.DataFrame) -> None:
    print("\n[7] Season feature summary (mean ± std)\n")
    cols = [
        "skin_mean_L", "skin_mean_a", "skin_mean_b", "skin_mean_C",
        "skin_warm_score", "hair_mean_L", "eye_mean_L",
        "face_contrast_L", "clear_muted_score",
    ]
    available = [c for c in cols if c in df.columns]
    rows = []
    for season in SEASON_ORDER:
        sub = df[df["label_season"] == season]
        row = {"season": season, "n": len(sub)}
        for c in available:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            row[c] = f"{v.mean():.2f}±{v.std():.2f}"
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("season")
    print(summary.to_string())
    summary.to_csv(ANALYSIS_DIR / "season_summary.csv")
    print(f"\n  Saved → {ANALYSIS_DIR / 'season_summary.csv'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Feature analysis for personal_color_pipeline")
    p.add_argument("--top-n",   type=int, default=25,  help="Features to show in importance/heatmap")
    p.add_argument("--top-box", type=int, default=12,  help="Features for box plots")
    args = p.parse_args()

    df, model_bundle = load_data()
    feat_cols = _feature_cols(df)
    print(f"Loaded {len(df)} samples, {len(feat_cols)} numeric feature columns\n")

    plot_feature_importance(model_bundle, df, feat_cols, top_n=args.top_n)
    plot_correlation_heatmap(df, feat_cols, top_n=args.top_n)
    plot_scatter_pairs(df)
    plot_boxplots(df, feat_cols, top_n=args.top_box)
    plot_pca(df, feat_cols)
    plot_palette_distances(df)
    print_season_summary(df)

    print(f"\n{'='*55}")
    print(f"  All analysis outputs → {ANALYSIS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
