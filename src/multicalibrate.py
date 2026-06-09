"""Multicalibration post-processing (HKRR) evaluation."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import HKRR from the installed package
from multicalibration.HKRR.hkrr import HKRRAlgorithm

# Import helpers from the fairness analysis script
from src.fairness_analysis import (
    prepare_data_and_predictions,
    _calibration_stats,
    _compute_error_rates,
    THRESHOLD
)

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multicalibration"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

PROTECTED_ATTRS = ["age_group", "gender_clean", "ethnicity_clean", "hospital_id_clean"]

def _ensure_dirs():
    for d in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def extract_subgroups(df: pd.DataFrame) -> tuple[list[list[int]], list[str]]:
    """Generate list of indices for marginal and intersectional subgroups."""
    subgroups = []
    labels = []
    
    # Marginal subgroups
    for attr in PROTECTED_ATTRS:
        if attr not in df.columns: continue
        for val in df[attr].dropna().unique():
            idx = np.where(df[attr] == val)[0].tolist()
            if len(idx) > 20:
                subgroups.append(idx)
                labels.append(f"{val}")
    
    # Pairwise Intersections
    attr_subset = ["age_group", "gender_clean", "ethnicity_clean"]
    for i, attr1 in enumerate(attr_subset):
        for attr2 in attr_subset[i+1:]:
            for val1 in df[attr1].dropna().unique():
                for val2 in df[attr2].dropna().unique():
                    idx = np.where((df[attr1] == val1) & (df[attr2] == val2))[0].tolist()
                    if len(idx) > 20:
                        subgroups.append(idx)
                        labels.append(f"{val1} / {val2}")
                        
    return subgroups, labels

def evaluate_calibration(df: pd.DataFrame, prob_col: str, labels: list[str], subgroups: list[list[int]]) -> pd.DataFrame:
    """Evaluate calibration gap for all provided subgroups."""
    results = []
    for idx_list, label in zip(subgroups, labels):
        if len(idx_list) < 20: continue
        
        y_t = df.iloc[idx_list]["y_true"].values
        y_p = df.iloc[idx_list][prob_col].values
        
        stats = _calibration_stats(y_t, y_p)
        obs_rate = y_t.mean()
        mean_pred = y_p.mean()
        
        results.append({
            "subgroup": label,
            "n": len(idx_list),
            "observed_rate": obs_rate,
            "mean_predicted": mean_pred,
            "calibration_gap": mean_pred - obs_rate,
            "ECE": stats["ECE"]
        })
    return pd.DataFrame(results)

def evaluate_error_rates(df: pd.DataFrame, prob_col: str, labels: list[str], subgroups: list[list[int]]) -> pd.DataFrame:
    """Evaluate FNR/FPR for all provided subgroups."""
    results = []
    y_pred_binary = (df[prob_col] >= THRESHOLD).astype(int)
    
    for idx_list, label in zip(subgroups, labels):
        if len(idx_list) < 20: continue
        
        y_t = df.iloc[idx_list]["y_true"].values
        y_p_bin = y_pred_binary.iloc[idx_list].values
        
        errs = _compute_error_rates(y_t, y_p_bin)
        errs["subgroup"] = label
        errs["n"] = len(idx_list)
        results.append(errs)
    return pd.DataFrame(results)

def plot_before_after_gap(before_df: pd.DataFrame, after_df: pd.DataFrame):
    """Plot calibration gaps before and after HKRR."""
    df = pd.merge(before_df, after_df, on=["subgroup", "n"], suffixes=("_before", "_after"))
    df["abs_gap_before"] = df["calibration_gap_before"].abs()
    
    # Sort by absolute gap before
    df = df.sort_values("abs_gap_before", ascending=False).head(25) # Top 25 worst
    
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(df))
    height = 0.35
    
    ax.barh(y_pos + height/2, df["calibration_gap_before"], height, label="Before (Raw)", color="#e74c3c")
    ax.barh(y_pos - height/2, df["calibration_gap_after"], height, label="After (HKRR)", color="#2ecc71")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r.subgroup} (n={r.n})" for _, r in df.iterrows()])
    ax.set_xlabel("Calibration Gap (Predicted - Observed)")
    ax.set_title("Effect of Multicalibration on Calibration Gap (Top 25 Subgroups)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multicalibration_gap_comparison.png", dpi=150)
    plt.close(fig)

def plot_before_after_fnr(before_df: pd.DataFrame, after_df: pd.DataFrame):
    """Plot FNR before and after HKRR."""
    df = pd.merge(before_df, after_df, on=["subgroup", "n"], suffixes=("_before", "_after"))
    df = df.dropna(subset=["FNR_before", "FNR_after"])
    df = df.sort_values("FNR_before", ascending=False).head(25)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(len(df))
    height = 0.35
    
    ax.barh(y_pos + height/2, df["FNR_before"], height, label="Before (Raw)", color="#e74c3c")
    ax.barh(y_pos - height/2, df["FNR_after"], height, label="After (HKRR)", color="#2ecc71")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r.subgroup} (n={r.n})" for _, r in df.iterrows()])
    ax.set_xlabel(f"False Negative Rate (Threshold={THRESHOLD})")
    ax.set_title("Effect of Multicalibration on Missed Deaths (FNR)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multicalibration_fnr_comparison.png", dpi=150)
    plt.close(fig)

def plot_mean_gap_bar(before_df: pd.DataFrame, after_df: pd.DataFrame):
    """Plot mean absolute calibration gap before and after as a simple bar chart."""
    df = pd.merge(before_df, after_df, on=["subgroup", "n"], suffixes=("_before", "_after"))
    mean_before = df["calibration_gap_before"].abs().mean()
    mean_after = df["calibration_gap_after"].abs().mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Before (Raw)", "After (HKRR)"], [mean_before, mean_after], color=["#e74c3c", "#2ecc71"], edgecolor="black")
    ax.set_ylabel("Mean Absolute Calibration Gap")
    ax.set_title("Overall Mean Calibration Gap (Across All 159 Subgroups)")
    for i, v in enumerate([mean_before, mean_after]):
        ax.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multicalibration_mean_gap_bar.png", dpi=150)
    plt.close(fig)

def plot_mean_fnr_bar(before_df: pd.DataFrame, after_df: pd.DataFrame):
    """Plot mean FNR before and after as a simple bar chart."""
    df = pd.merge(before_df, after_df, on=["subgroup", "n"], suffixes=("_before", "_after"))
    df = df.dropna(subset=["FNR_before", "FNR_after"])
    mean_before = df["FNR_before"].mean()
    mean_after = df["FNR_after"].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Before (Raw)", "After (HKRR)"], [mean_before, mean_after], color=["#e74c3c", "#2ecc71"], edgecolor="black")
    ax.set_ylabel(f"Mean FNR (Threshold={THRESHOLD})")
    ax.set_title("Overall Mean False Negative Rate (Across All Valid Subgroups)")
    for i, v in enumerate([mean_before, mean_after]):
        ax.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "multicalibration_mean_fnr_bar.png", dpi=150)
    plt.close(fig)


def main():
    _ensure_dirs()
    print("Step 1: Loading data...")
    test_df = prepare_data_and_predictions()
    
    print("Step 2: Splitting into Calibration and Evaluation sets...")
    calib_df, eval_df = train_test_split(
        test_df, test_size=0.5, stratify=test_df["y_true"], random_state=42
    )
    calib_df = calib_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    
    print(f"Calibration Set: {len(calib_df)} samples")
    print(f"Evaluation Set: {len(eval_df)} samples")
    
    calib_subgroups, _ = extract_subgroups(calib_df)
    eval_subgroups, eval_labels = extract_subgroups(eval_df)
    
    print(f"Identified {len(calib_subgroups)} overlapping subgroups.")
    
    print("Step 3: Fitting HKRR Multicalibration Algorithm...")
    hkrr = HKRRAlgorithm(verbose=False)
    params = {
        'lambda': 0.05,
        'alpha': 0.1,
        'max_iter': 50,
        'randomized': True,
        'use_oracle': False
    }
    
    hkrr.fit(
        confs=calib_df["y_prob"].values,
        labels=calib_df["y_true"].values,
        subgroups=calib_subgroups,
        params=params
    )
    
    print(f"HKRR completed in {len(hkrr.delta_iters)} iterations.")
    
    print("Step 4: Predicting and Evaluating on Hold-out Set...")
    eval_df["y_prob_hkrr"] = hkrr.predict(
        f_xs=eval_df["y_prob"].values,
        groups=eval_subgroups
    )
    
    # 5. Evaluate Metrics
    calib_before = evaluate_calibration(eval_df, "y_prob", eval_labels, eval_subgroups)
    calib_after = evaluate_calibration(eval_df, "y_prob_hkrr", eval_labels, eval_subgroups)
    
    err_before = evaluate_error_rates(eval_df, "y_prob", eval_labels, eval_subgroups)
    err_after = evaluate_error_rates(eval_df, "y_prob_hkrr", eval_labels, eval_subgroups)
    
    # 6. Generate Plots
    print("Step 5: Generating visual comparisons...")
    plot_before_after_gap(calib_before, calib_after)
    plot_before_after_fnr(err_before, err_after)
    plot_mean_gap_bar(calib_before, calib_after)
    plot_mean_fnr_bar(err_before, err_after)
    
    # 7. Write Report
    report_path = REPORTS_DIR / "multicalibration_report.txt"
    with open(report_path, "w") as f:
        f.write("MULTICALIBRATION EFFECT SUMMARY (Evaluation Set)\n")
        f.write("================================================\n\n")
        
        merged = pd.merge(calib_before, calib_after, on=["subgroup", "n"], suffixes=("_before", "_after"))
        merged["abs_gap_before"] = merged["calibration_gap_before"].abs()
        merged["abs_gap_after"] = merged["calibration_gap_after"].abs()
        
        avg_gap_before = merged["abs_gap_before"].mean()
        avg_gap_after = merged["abs_gap_after"].mean()
        
        f.write(f"Average Absolute Calibration Gap (Before): {avg_gap_before:.4f}\n")
        f.write(f"Average Absolute Calibration Gap (After):  {avg_gap_after:.4f}\n\n")
        
        f.write("Top 10 Subgroups by Original Gap:\n")
        top_10 = merged.sort_values("abs_gap_before", ascending=False).head(10)
        for _, r in top_10.iterrows():
            f.write(f"  {r.subgroup} (n={r.n}):\n")
            f.write(f"      Before -> Gap: {r.calibration_gap_before:+.3f} | ECE: {r.ECE_before:.3f}\n")
            f.write(f"      After  -> Gap: {r.calibration_gap_after:+.3f} | ECE: {r.ECE_after:.3f}\n")
            
    print(f"✓ Multicalibration complete. Outputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
