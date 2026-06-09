"""Fairness analysis for the reduced_logistic_elastic_net model (Meeting 6).

Steps:
  1. Load training data (with protected attributes) and the saved model.
  2. create the test split same as when the model was trained
  3. Generate mortality probability predictions on the test set.
  4. Evaluate fairness: FNR/FPR by subgroup, calibration, intersectional analysis.
  5. Save all figures and a narrative summary.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.cohort import apply_stage1_cohort
from src.features import (
    EXCLUDED_MODEL_COLUMNS,
    CATEGORICAL_CODE_FEATURES,
    TARGET_COLUMN,
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = PROJECT_ROOT / "data" / "training_v2.csv"
MODEL_PATH = PROJECT_ROOT / "saved_models_and_results" / "reduced_logistic_elastic_net.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fairness_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

PROTECTED_ATTRS = ["age", "gender", "ethnicity", "hospital_id"]
REDUCED_EXCLUDED = ["hospital_id", "icu_id", "apache_2_diagnosis", "apache_3j_diagnosis"]

THRESHOLD = 0.3  # recommended decision threshold
THRESHOLD = 0.5  # default decision threshold


def _ensure_dirs():
    for d in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_data() -> pd.DataFrame:
    """Load and apply cohort filtering."""
    raw = pd.read_csv(TRAINING_PATH)
    return apply_stage1_cohort(raw)


def _get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the 177 features used by the reduced model."""
    excluded = set(EXCLUDED_MODEL_COLUMNS) | set(REDUCED_EXCLUDED)
    return [c for c in df.columns if c not in excluded]


def _age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    if age < 80:
        return "65-79"
    return "80+"


def _compute_error_rates(y_true, y_pred):
    """Return dict with FNR, FPR, and support counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return {
        "FNR": fnr,
        "FPR": fpr,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
        "n_total": int(tn + fp + fn + tp),
        "mortality_rate": (tp + fn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else np.nan,
    }


def _calibration_stats(y_true, y_prob, n_bins=10):
    """Compute calibration: mean predicted vs observed mortality per bin."""
    try:
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ece = np.mean(np.abs(fraction_pos - mean_pred))
        return {"ECE": ece, "n_bins_used": len(fraction_pos)}
    except Exception:
        return {"ECE": np.nan, "n_bins_used": 0}



def prepare_data_and_predictions():
    print("Loading data and model...")
    df = _load_data()
    model = joblib.load(MODEL_PATH)
    feature_cols = _get_model_features(df)

    # create the test split same as when the model was trained
    # same as in train_models.py
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(df, df[TARGET_COLUMN].astype(int)))
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    test_df["y_true"] = y_test.values
    test_df["y_prob"] = y_prob
    test_df["y_pred"] = y_pred

    # Add grouped protected attributes
    test_df["age_group"] = test_df["age"].apply(_age_group)
    test_df["gender_clean"] = test_df["gender"].fillna("Unknown")
    test_df["ethnicity_clean"] = test_df["ethnicity"].fillna("Unknown")
    test_df["hospital_id_clean"] = test_df["hospital_id"].fillna(-1).astype(int).astype(str)

    print(f"  Test set size: {len(test_df)}, Mortality rate: {y_test.mean():.3f}")
    print(f"  Model features: {len(feature_cols)}")
    return test_df

def compute_subgroup_error_rates(test_df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing subgroup error rates...")
    rows = []
    attr_col_map = {
        "Age": "age_group",
        "Gender": "gender_clean",
        "Ethnicity": "ethnicity_clean",
    }
    for attr_name, col in attr_col_map.items():
        for grp, sub in test_df.groupby(col):
            if len(sub) < 10:
                continue
            rates = _compute_error_rates(sub["y_true"], sub["y_pred"])
            rates["attribute"] = attr_name
            rates["subgroup"] = str(grp)
            rows.append(rates)

    # Hospital: top 15 by volume
    hosp_counts = test_df["hospital_id_clean"].value_counts()
    top_hospitals = hosp_counts.head(15).index.tolist()
    for h in top_hospitals:
        sub = test_df[test_df["hospital_id_clean"] == h]
        rates = _compute_error_rates(sub["y_true"], sub["y_pred"])
        rates["attribute"] = "Hospital ID"
        rates["subgroup"] = str(h)
        rows.append(rates)

    result = pd.DataFrame(rows)
    result = result[["attribute", "subgroup", "FNR", "FPR", "n_positive", "n_negative",
                      "n_total", "mortality_rate", "TP", "FP", "FN", "TN"]]
    return result


def compute_subgroup_calibration(test_df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing subgroup calibration...")
    rows = []
    attr_col_map = {
        "Age": "age_group",
        "Gender": "gender_clean",
        "Ethnicity": "ethnicity_clean",
    }
    for attr_name, col in attr_col_map.items():
        for grp, sub in test_df.groupby(col):
            if len(sub) < 30:
                continue
            stats = _calibration_stats(sub["y_true"], sub["y_prob"])
            stats["attribute"] = attr_name
            stats["subgroup"] = str(grp)
            stats["n"] = len(sub)
            stats["observed_rate"] = sub["y_true"].mean()
            stats["mean_predicted"] = sub["y_prob"].mean()
            stats["calibration_gap"] = stats["mean_predicted"] - stats["observed_rate"]
            rows.append(stats)
    return pd.DataFrame(rows)[["attribute", "subgroup", "n", "observed_rate",
                                "mean_predicted", "calibration_gap", "ECE"]]



def compute_intersectional_analysis(test_df: pd.DataFrame) -> pd.DataFrame:
    
    """
    do intersectional analysis, first take 2 groups combinations and then 3 groups combinations.
    """

    print("\nIntersectional subgroup analysis...")
    rows = []
    # Age x Gender
    for (ag, gn), sub in test_df.groupby(["age_group", "gender_clean"]):
        if len(sub) < 20:
            continue
        rates = _compute_error_rates(sub["y_true"], sub["y_pred"])
        cal = _calibration_stats(sub["y_true"], sub["y_prob"])
        rows.append({
            "intersection": f"{ag} / {gn}",
            "variables": "Age × Gender",
            **rates, **cal,
        })

    # Age x Ethnicity
    for (ag, eth), sub in test_df.groupby(["age_group", "ethnicity_clean"]):
        if len(sub) < 20:
            continue
        rates = _compute_error_rates(sub["y_true"], sub["y_pred"])
        cal = _calibration_stats(sub["y_true"], sub["y_prob"])
        rows.append({
            "intersection": f"{ag} / {eth}",
            "variables": "Age × Ethnicity",
            **rates, **cal,
        })

    # Gender x Ethnicity
    for (gn, eth), sub in test_df.groupby(["gender_clean", "ethnicity_clean"]):
        if len(sub) < 20:
            continue
        rates = _compute_error_rates(sub["y_true"], sub["y_pred"])
        cal = _calibration_stats(sub["y_true"], sub["y_prob"])
        rows.append({
            "intersection": f"{gn} / {eth}",
            "variables": "Gender × Ethnicity",
            **rates, **cal,
        })

    # Age x Gender x Ethnicity (triple intersection)
    for (ag, gn, eth), sub in test_df.groupby(["age_group", "gender_clean", "ethnicity_clean"]):
        if len(sub) < 20:
            continue
        rates = _compute_error_rates(sub["y_true"], sub["y_pred"])
        cal = _calibration_stats(sub["y_true"], sub["y_prob"])
        rows.append({
            "intersection": f"{ag} / {gn} / {eth}",
            "variables": "Age × Gender × Ethnicity",
            **rates, **cal,
        })

    result = pd.DataFrame(rows)
    cols = ["variables", "intersection", "FNR", "FPR", "ECE", "mortality_rate",
            "n_total", "n_positive", "n_negative"]
    return result[[c for c in cols if c in result.columns]]

def plot_error_rates_bar(error_df: pd.DataFrame):
    """Bar charts of FNR and FPR by subgroup for each protected attribute."""
    for attr in error_df["attribute"].unique():
        sub = error_df[error_df["attribute"] == attr].sort_values("subgroup")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Error Rates by {attr}", fontsize=14, fontweight="bold")

        # FNR
        colors_fnr = plt.cm.Reds(np.linspace(0.3, 0.8, len(sub)))
        axes[0].barh(sub["subgroup"], sub["FNR"], color=colors_fnr, edgecolor="black", linewidth=0.5)
        axes[0].set_xlabel("False Negative Rate (Missed Deaths)")
        axes[0].set_title("FNR by Subgroup")
        for i, (v, n) in enumerate(zip(sub["FNR"], sub["n_positive"])):
            if not np.isnan(v):
                axes[0].text(v + 0.005, i, f"{v:.3f} (n={n})", va="center", fontsize=8)

        # FPR
        colors_fpr = plt.cm.Blues(np.linspace(0.3, 0.8, len(sub)))
        axes[1].barh(sub["subgroup"], sub["FPR"], color=colors_fpr, edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_title("FPR by Subgroup")
        for i, (v, n) in enumerate(zip(sub["FPR"], sub["n_negative"])):
            if not np.isnan(v):
                axes[1].text(v + 0.005, i, f"{v:.3f} (n={n})", va="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"error_rates_{attr.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved error rate plot for {attr}")

def plot_calibration_curves(test_df: pd.DataFrame):
    """Calibration curves overlaid by subgroup for each protected attribute."""
    attr_col_map = {
        "Age": "age_group",
        "Gender": "gender_clean",
        "Ethnicity": "ethnicity_clean",
    }
    for attr_name, col in attr_col_map.items():
        groups = sorted(test_df[col].unique())
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

        for grp in groups:
            sub = test_df[test_df[col] == grp]
            if len(sub) < 30 or sub["y_true"].nunique() < 2:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(sub["y_true"], sub["y_prob"],
                                                         n_bins=10, strategy="uniform")
                ax.plot(mean_pred, frac_pos, "o-", label=f"{grp} (n={len(sub)})", markersize=4)
            except Exception:
                continue

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Mortality Rate")
        ax.set_title(f"Calibration Curves by {attr_name}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"calibration_{attr_name.lower()}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved calibration plot for {attr_name}")


def plot_intersectional_summary(inter_df: pd.DataFrame):
    """Horizontal bar chart showing top/bottom intersections by FNR."""
    if inter_df.empty:
        return
    ranked = inter_df.dropna(subset=["FNR"]).sort_values("FNR", ascending=True)

    # Show worst 15 and best 15
    n_show = min(15, len(ranked))
    best = ranked.head(n_show)
    worst = ranked.tail(n_show)
    display_df = pd.concat([best, worst]).drop_duplicates()
    display_df = display_df.sort_values("FNR", ascending=True)

    display_labels = display_df["intersection"] + " (n=" + display_df["n_total"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(12, max(6, len(display_df) * 0.35)))
    colors = ["#2ecc71" if v < ranked["FNR"].median() else "#e74c3c" for v in display_df["FNR"]]
    ax.barh(display_labels, display_df["FNR"], color=colors,
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("False Negative Rate (Missed Deaths)")
    ax.set_title("Intersectional FNR: Best vs Worst Subgroups", fontsize=13, fontweight="bold")
    ax.axvline(ranked["FNR"].median(), color="orange", linestyle="--", label=f"Median FNR={ranked['FNR'].median():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intersectional_fnr_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved intersectional FNR summary plot")


def plot_calibration_gap_heatmap(cal_df: pd.DataFrame):
    """Bar chart of calibration gap (predicted - observed) by subgroup."""
    if cal_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = cal_df["attribute"] + ": " + cal_df["subgroup"]
    gaps = cal_df["calibration_gap"]
    colors = ["#e74c3c" if abs(g) > 0.02 else "#2ecc71" for g in gaps]
    ax.barh(labels, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Calibration Gap (Predicted − Observed)")
    ax.set_title("Calibration Gap by Subgroup", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_gap_by_subgroup.png", dpi=150)
    plt.close(fig)
    print("  Saved calibration gap plot")

def generate_narrative(error_df, cal_df, inter_df, test_df):
    """
    generate automatc report for the analysis
    """
    print("\nGenerating narrative report...")
    lines = []
    lines.append("=" * 80)
    lines.append("FAIRNESS ANALYSIS REPORT — reduced_logistic_elastic_net")
    lines.append("=" * 80)
    lines.append("")

    n = len(test_df)
    mort = test_df["y_true"].mean()
    lines.append(f"Test Set: {n:,} patients | Mortality rate: {mort:.1%}")
    lines.append(f"Decision threshold: {THRESHOLD}")
    lines.append("")

    # --- Error rate findings ---
    lines.append("-" * 60)
    lines.append("1. SUBGROUP ERROR RATES (Equal Opportunity)")
    lines.append("-" * 60)
    for attr in error_df["attribute"].unique():
        sub = error_df[error_df["attribute"] == attr]
        lines.append(f"\n  {attr}:")
        for _, row in sub.iterrows():
            fnr_str = f"{row['FNR']:.3f}" if not np.isnan(row["FNR"]) else "N/A"
            fpr_str = f"{row['FPR']:.3f}" if not np.isnan(row["FPR"]) else "N/A"
            lines.append(f"    {row['subgroup']:>20s}: FNR={fnr_str}  FPR={fpr_str}  "
                         f"(n={row['n_total']}, deaths={row['n_positive']})")

        fnr_range = sub["FNR"].max() - sub["FNR"].min()
        worst_fnr = sub.loc[sub["FNR"].idxmax()]
        best_fnr = sub.loc[sub["FNR"].idxmin()]
        lines.append(f"    → FNR range: {fnr_range:.3f} "
                     f"(worst: {worst_fnr['subgroup']}, best: {best_fnr['subgroup']})")

    # --- Calibration findings ---
    lines.append("")
    lines.append("-" * 60)
    lines.append("2. CALIBRATION ANALYSIS")
    lines.append("-" * 60)
    for _, row in cal_df.iterrows():
        lines.append(f"  {row['attribute']:>10s} / {row['subgroup']:>20s}: "
                     f"Observed={row['observed_rate']:.3f}  Predicted={row['mean_predicted']:.3f}  "
                     f"Gap={row['calibration_gap']:+.3f}  ECE={row['ECE']:.3f}  (n={row['n']})")

    # Identify worst calibration gaps
    worst_cal = cal_df.reindex(cal_df["calibration_gap"].abs().nlargest(3).index)
    lines.append("\n  Largest calibration gaps:")
    for _, row in worst_cal.iterrows():
        lines.append(f"    {row['attribute']}/{row['subgroup']}: "
                     f"gap={row['calibration_gap']:+.3f}")

    # --- Intersectional findings ---
    lines.append("")
    lines.append("-" * 60)
    lines.append("3. INTERSECTIONAL ANALYSIS")
    lines.append("-" * 60)
    if not inter_df.empty:
        ranked = inter_df.dropna(subset=["FNR"]).sort_values("FNR")
        n_show = min(5, len(ranked))
        lines.append("\n  Best-performing intersections (lowest FNR):")
        for _, row in ranked.head(n_show).iterrows():
            lines.append(f"    {row['intersection']:>40s}: FNR={row['FNR']:.3f}  "
                         f"(n={row['n_total']}, deaths={row['n_positive']})")
        lines.append("\n  Worst-performing intersections (highest FNR):")
        for _, row in ranked.tail(n_show).iterrows():
            lines.append(f"    {row['intersection']:>40s}: FNR={row['FNR']:.3f}  "
                         f"(n={row['n_total']}, deaths={row['n_positive']})")

    # --- Clinical implications ---
    lines.append("")
    lines.append("-" * 60)
    lines.append("4. CLINICAL IMPLICATIONS")
    lines.append("-" * 60)
    lines.append("""
  A high False Negative Rate (FNR) in a subgroup means the model is more likely
  to MISS deaths in that population — patients who will die are predicted to
  survive. This is the most dangerous type of error in a mortality prediction
  system because it can lead to under-triage and inadequate care.

  A high False Positive Rate (FPR) means unnecessary alarms, which can lead to
  resource waste and alarm fatigue, but is generally less harmful than missed
  deaths.

  Calibration gaps indicate that a predicted risk (e.g., 30%) does not translate
  to the same real-world mortality rate across groups. For example, if the model
  predicts 30% risk for a group whose actual mortality is 45%, clinicians relying
  on those scores would systematically underestimate risk for that group.

  RECOMMENDATIONS:
  - Subgroups with FNR significantly above the population average should be
    flagged for additional clinical review.
  - Consider threshold adjustment for poorly calibrated subgroups.
  - Monitor model performance by subgroup in deployment to detect drift.
  - Collect more data for underrepresented subgroups to improve calibration.
""")

    report = "\n".join(lines)
    report_path = REPORTS_DIR / "fairness_narrative_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {report_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _ensure_dirs()

    # Load data, model, generate predictions
    test_df = prepare_data_and_predictions() # test df includes y pred col

    # Subgroup error rates
    error_df = compute_subgroup_error_rates(test_df)
    error_df.to_csv(REPORTS_DIR / "subgroup_error_rates.csv", index=False)

    # Calibration
    cal_df = compute_subgroup_calibration(test_df)
    cal_df.to_csv(REPORTS_DIR / "subgroup_calibration.csv", index=False)

    # Intersectional analysis
    inter_df = compute_intersectional_analysis(test_df)
    inter_df.to_csv(REPORTS_DIR / "intersectional_analysis.csv", index=False)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_error_rates_bar(error_df)
    plot_calibration_curves(test_df)
    plot_intersectional_summary(inter_df)
    plot_calibration_gap_heatmap(cal_df)

    # Narrative
    report = generate_narrative(error_df, cal_df, inter_df, test_df)
    print("\n" + report)
    print("\n✓ Fairness analysis complete. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
