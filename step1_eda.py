#!/usr/bin/env python3
# step1_eda.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "num_frames",
        "empty_frame_count",
        "nonempty_frame_count",
        "lumen_pixel_count",
        "plaque_pixel_count",
        "background_pixel_count",
        "lumen_frame_count",
        "plaque_frame_count",
        "min_lumen_area",
        "mean_lumen_area",
        "max_lumen_area",
        "min_plaque_area",
        "mean_plaque_area",
        "max_plaque_area",
    ]
    df = ensure_numeric(df, numeric_cols)

    df["empty_frame_ratio"] = safe_divide(df["empty_frame_count"], df["num_frames"])
    df["nonempty_frame_ratio"] = safe_divide(df["nonempty_frame_count"], df["num_frames"])

    df["lumen_frame_ratio"] = safe_divide(df["lumen_frame_count"], df["num_frames"])
    df["plaque_frame_ratio"] = safe_divide(df["plaque_frame_count"], df["num_frames"])

    df["total_vessel_pixel_count"] = df["lumen_pixel_count"] + df["plaque_pixel_count"]
    df["global_lumen_ratio"] = safe_divide(df["lumen_pixel_count"], df["total_vessel_pixel_count"])
    df["global_plaque_ratio"] = safe_divide(df["plaque_pixel_count"], df["total_vessel_pixel_count"])

    df["plaque_to_lumen_pixel_ratio"] = safe_divide(df["plaque_pixel_count"], df["lumen_pixel_count"])
    df["mean_plaque_to_lumen_area_ratio"] = safe_divide(df["mean_plaque_area"], df["mean_lumen_area"])
    df["max_plaque_to_min_lumen_ratio"] = safe_divide(df["max_plaque_area"], df["min_lumen_area"])

    df["min_to_mean_lumen_ratio"] = safe_divide(df["min_lumen_area"], df["mean_lumen_area"])
    df["min_to_max_lumen_ratio"] = safe_divide(df["min_lumen_area"], df["max_lumen_area"])
    df["mean_to_max_lumen_ratio"] = safe_divide(df["mean_lumen_area"], df["max_lumen_area"])

    df["lumen_area_range"] = df["max_lumen_area"] - df["min_lumen_area"]
    df["plaque_area_range"] = df["max_plaque_area"] - df["min_plaque_area"]

    df["lumen_range_to_mean_ratio"] = safe_divide(df["lumen_area_range"], df["mean_lumen_area"])
    df["plaque_range_to_mean_ratio"] = safe_divide(df["plaque_area_range"], df["mean_plaque_area"])

    return df


def manual_ks_statistic(x, y):
    """
    Simple two-sample KS statistic without scipy.
    Returns maximum absolute difference between empirical CDFs.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    values = np.sort(np.unique(np.concatenate([x, y])))
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    x_cdf = np.searchsorted(x_sorted, values, side="right") / len(x_sorted)
    y_cdf = np.searchsorted(y_sorted, values, side="right") / len(y_sorted)

    return float(np.max(np.abs(x_cdf - y_cdf)))


def summarize_label_distribution(train_df: pd.DataFrame, out_dir: Path):
    lines = []
    lines.append("Step 1 EDA Summary")
    lines.append("=" * 50)
    lines.append("")

    lines.append("[Label distribution]")
    lines.append(f"num_samples: {len(train_df)}")
    lines.append(f"FFR_min: {train_df['FFR'].min():.4f}")
    lines.append(f"FFR_mean: {train_df['FFR'].mean():.4f}")
    lines.append(f"FFR_std: {train_df['FFR'].std():.4f}")
    lines.append(f"FFR_median: {train_df['FFR'].median():.4f}")
    lines.append(f"FFR_max: {train_df['FFR'].max():.4f}")

    class_counts = train_df["ffr_class"].value_counts().sort_index()
    class_ratios = train_df["ffr_class"].value_counts(normalize=True).sort_index()

    lines.append("")
    lines.append("[Class distribution]")
    for cls in [0, 1]:
        count = int(class_counts.get(cls, 0))
        ratio = float(class_ratios.get(cls, 0.0))
        meaning = "FFR < 0.8" if cls == 0 else "FFR >= 0.8"
        lines.append(f"class {cls} ({meaning}): {count} ({ratio:.4%})")

    near = train_df[(train_df["FFR"] >= 0.75) & (train_df["FFR"] <= 0.85)].copy()
    lines.append("")
    lines.append("[Near-threshold samples]")
    lines.append(f"0.75 <= FFR <= 0.85: {len(near)} ({len(near) / len(train_df):.4%})")

    near.to_csv(out_dir / "near_threshold_samples.csv", index=False)

    return "\n".join(lines)


def classwise_summary(train_df: pd.DataFrame, feature_cols: list[str], out_dir: Path):
    rows = []

    for col in feature_cols:
        if col not in train_df.columns:
            continue

        class0 = train_df.loc[train_df["ffr_class"] == 0, col].dropna()
        class1 = train_df.loc[train_df["ffr_class"] == 1, col].dropna()

        row = {
            "feature": col,
            "class0_mean": class0.mean(),
            "class0_std": class0.std(),
            "class0_median": class0.median(),
            "class1_mean": class1.mean(),
            "class1_std": class1.std(),
            "class1_median": class1.median(),
            "mean_diff_class0_minus_class1": class0.mean() - class1.mean(),
            "median_diff_class0_minus_class1": class0.median() - class1.median(),
            "ks_stat_class0_vs_class1": manual_ks_statistic(class0.values, class1.values),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("ks_stat_class0_vs_class1", ascending=False)
    summary_df.to_csv(out_dir / "classwise_feature_summary.csv", index=False)
    return summary_df


def correlation_summary(train_df: pd.DataFrame, feature_cols: list[str], out_dir: Path):
    rows = []

    for col in feature_cols:
        if col not in train_df.columns:
            continue

        x = train_df[col]
        y_ffr = train_df["FFR"]
        y_class = train_df["ffr_class"]

        if x.notna().sum() < 3:
            continue

        rows.append({
            "feature": col,
            "corr_with_FFR_pearson": x.corr(y_ffr, method="pearson"),
            "corr_with_FFR_spearman": x.corr(y_ffr, method="spearman"),
            "corr_with_ffr_class_pearson": x.corr(y_class, method="pearson"),
            "corr_with_ffr_class_spearman": x.corr(y_class, method="spearman"),
        })

    corr_df = pd.DataFrame(rows)
    corr_df["abs_corr_with_FFR_spearman"] = corr_df["corr_with_FFR_spearman"].abs()
    corr_df = corr_df.sort_values("abs_corr_with_FFR_spearman", ascending=False)
    corr_df.to_csv(out_dir / "feature_correlation_summary.csv", index=False)
    return corr_df


def train_test_distribution_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
):
    rows = []

    for col in feature_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        train_x = train_df[col].dropna()
        test_x = test_df[col].dropna()

        row = {
            "feature": col,
            "train_mean": train_x.mean(),
            "train_std": train_x.std(),
            "train_median": train_x.median(),
            "test_mean": test_x.mean(),
            "test_std": test_x.std(),
            "test_median": test_x.median(),
            "mean_diff_test_minus_train": test_x.mean() - train_x.mean(),
            "median_diff_test_minus_train": test_x.median() - train_x.median(),
            "ks_stat_train_vs_test": manual_ks_statistic(train_x.values, test_x.values),
        }
        rows.append(row)

    dist_df = pd.DataFrame(rows)
    dist_df = dist_df.sort_values("ks_stat_train_vs_test", ascending=False)
    dist_df.to_csv(out_dir / "train_test_feature_distribution_summary.csv", index=False)
    return dist_df


def save_histogram_bins(train_df: pd.DataFrame, out_dir: Path):
    bins = np.arange(0.35, 1.05 + 1e-8, 0.05)
    counts, edges = np.histogram(train_df["FFR"].values, bins=bins)

    hist_df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    })
    hist_df.to_csv(out_dir / "ffr_histogram_bins.csv", index=False)

    class_counts = train_df["ffr_class"].value_counts().sort_index()
    class_df = pd.DataFrame({
        "ffr_class": [0, 1],
        "meaning": ["FFR < 0.8", "FFR >= 0.8"],
        "count": [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))],
    })
    class_df["ratio"] = class_df["count"] / class_df["count"].sum()
    class_df.to_csv(out_dir / "class_distribution.csv", index=False)


def make_optional_plots(train_df: pd.DataFrame, test_df: pd.DataFrame | None, out_dir: Path):
    plt = try_import_matplotlib()
    if plt is None:
        print("[WARN] matplotlib is not available. Skipping plots.")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # FFR histogram
    plt.figure()
    plt.hist(train_df["FFR"], bins=20)
    plt.axvline(0.8, linestyle="--")
    plt.xlabel("FFR")
    plt.ylabel("Count")
    plt.title("FFR distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "ffr_distribution.png", dpi=150)
    plt.close()

    # Class distribution
    plt.figure()
    counts = train_df["ffr_class"].value_counts().sort_index()
    plt.bar(["0: FFR<0.8", "1: FFR>=0.8"], [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Count")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "class_distribution.png", dpi=150)
    plt.close()

    # num_frames train/test
    plt.figure()
    plt.hist(train_df["num_frames"], bins=30, alpha=0.6, label="train")
    if test_df is not None:
        plt.hist(test_df["num_frames"], bins=30, alpha=0.6, label="test_public")
        plt.legend()
    plt.xlabel("num_frames")
    plt.ylabel("Count")
    plt.title("num_frames distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "num_frames_train_test.png", dpi=150)
    plt.close()

    # Classwise important features
    important_features = [
        "num_frames",
        "min_lumen_area",
        "mean_lumen_area",
        "max_lumen_area",
        "mean_plaque_area",
        "max_plaque_area",
        "global_plaque_ratio",
        "plaque_to_lumen_pixel_ratio",
        "min_to_mean_lumen_ratio",
    ]

    for col in important_features:
        if col not in train_df.columns:
            continue

        plt.figure()
        data0 = train_df.loc[train_df["ffr_class"] == 0, col].dropna()
        data1 = train_df.loc[train_df["ffr_class"] == 1, col].dropna()
        plt.boxplot([data0, data1], tick_labels=["0: FFR<0.8", "1: FFR>=0.8"])
        plt.ylabel(col)
        plt.title(f"Classwise {col}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"classwise_{col}.png", dpi=150)
        plt.close()

    # Scatter against FFR
    scatter_features = [
        "num_frames",
        "min_lumen_area",
        "mean_lumen_area",
        "global_plaque_ratio",
        "plaque_to_lumen_pixel_ratio",
        "min_to_mean_lumen_ratio",
    ]

    for col in scatter_features:
        if col not in train_df.columns:
            continue

        plt.figure()
        plt.scatter(train_df[col], train_df["FFR"], s=10, alpha=0.5)
        plt.axhline(0.8, linestyle="--")
        plt.xlabel(col)
        plt.ylabel("FFR")
        plt.title(f"FFR vs {col}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"scatter_FFR_vs_{col}.png", dpi=150)
        plt.close()

    print(f"[SAVE] plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0-dir", type=str, default="step0_outputs")
    parser.add_argument("--out-dir", type=str, default="step1_outputs")
    args = parser.parse_args()

    step0_dir = Path(args.step0_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = step0_dir / "data_check_train.csv"
    test_path = step0_dir / "data_check_test_public.csv"
    labels_path = step0_dir / "labels_with_class.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing file: {labels_path}")

    train_check = pd.read_csv(train_path)
    labels = pd.read_csv(labels_path)

    train_check["serial_no"] = train_check["serial_no"].astype(str)
    labels["serial_no"] = labels["serial_no"].astype(str)

    if "ffr_class" not in labels.columns:
        labels["ffr_class"] = (labels["FFR"] >= 0.8).astype(int)

    train_df = train_check.merge(labels[["serial_no", "FFR", "ffr_class"]], on="serial_no", how="left")
    train_df = add_derived_features(train_df)

    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df["serial_no"] = test_df["serial_no"].astype(str)
        test_df = add_derived_features(test_df)

    # Save merged datasets
    train_df.to_csv(out_dir / "train_eda_table.csv", index=False)
    if test_df is not None:
        test_df.to_csv(out_dir / "test_public_eda_table.csv", index=False)

    # Feature columns for analysis
    exclude_cols = {
        "serial_no",
        "file_path",
        "exists",
        "read_ok",
        "error",
        "dtype",
        "unique_values",
        "only_0_1_2",
        "has_lumen",
        "has_plaque",
        "FFR",
        "ffr_class",
    }

    feature_cols = [
        c for c in train_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # Save basic summaries
    label_summary_text = summarize_label_distribution(train_df, out_dir)
    class_summary_df = classwise_summary(train_df, feature_cols, out_dir)
    corr_df = correlation_summary(train_df, feature_cols, out_dir)
    save_histogram_bins(train_df, out_dir)

    dist_df = None
    if test_df is not None:
        dist_df = train_test_distribution_summary(train_df, test_df, feature_cols, out_dir)

    # Optional plots
    make_optional_plots(train_df, test_df, out_dir)

    # Write final text summary
    lines = []
    lines.append(label_summary_text)
    lines.append("")
    lines.append("[Top class-separating features by KS statistic]")
    lines.append(class_summary_df.head(15).to_string(index=False))
    lines.append("")
    lines.append("[Top FFR-correlated features by Spearman correlation]")
    lines.append(corr_df.head(15).to_string(index=False))

    if dist_df is not None:
        lines.append("")
        lines.append("[Top train-test shifted features by KS statistic]")
        lines.append(dist_df.head(15).to_string(index=False))

    summary_path = out_dir / "step1_eda_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[SAVE] {out_dir / 'train_eda_table.csv'}")
    if test_df is not None:
        print(f"[SAVE] {out_dir / 'test_public_eda_table.csv'}")
    print(f"[SAVE] {out_dir / 'near_threshold_samples.csv'}")
    print(f"[SAVE] {out_dir / 'class_distribution.csv'}")
    print(f"[SAVE] {out_dir / 'ffr_histogram_bins.csv'}")
    print(f"[SAVE] {out_dir / 'classwise_feature_summary.csv'}")
    print(f"[SAVE] {out_dir / 'feature_correlation_summary.csv'}")
    if dist_df is not None:
        print(f"[SAVE] {out_dir / 'train_test_feature_distribution_summary.csv'}")
    print(f"[SAVE] {summary_path}")

    print("")
    print("=" * 80)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()