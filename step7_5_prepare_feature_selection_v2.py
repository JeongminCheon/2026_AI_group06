#!/usr/bin/env python3
# step7_5_prepare_feature_selection_v2.py
#
# Purpose:
#   Prepare selected feature CSVs for additional Step 7 experiments.
#
# This script creates fixed feature subsets from:
#   1) merged v1+v2 features
#   2) v2-only features
#
# Selection criterion:
#   absolute Spearman correlation with FFR on train set.
#
# Output examples:
#   step7_5_feature_selection_outputs/train_merged_top200_corr.csv
#   step7_5_feature_selection_outputs/test_merged_top200_corr.csv
#   step7_5_feature_selection_outputs/train_merged_top300_corr.csv
#   step7_5_feature_selection_outputs/test_merged_top300_corr.csv
#   step7_5_feature_selection_outputs/train_v2only_top200_corr.csv
#   step7_5_feature_selection_outputs/test_v2only_top200_corr.csv
#   step7_5_feature_selection_outputs/run_step7_5_training_commands.sh
#
# After this script, train with step3_3_advanced_mlp.py using the generated CSVs.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["serial_no"]
LABEL_COLS = ["FFR", "ffr_class"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], 0.0)
    out = out.fillna(0.0)
    return out


def get_numeric_feature_cols(df: pd.DataFrame, v2_only: bool = False):
    exclude = set(ID_COLS + LABEL_COLS)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if v2_only and not c.startswith("v2_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # Exclude status columns if they accidentally became numeric.
            if c.endswith("_read_ok"):
                continue
            cols.append(c)
    return cols


def compute_corr_ranking(train: pd.DataFrame, feature_cols):
    if "FFR" not in train.columns:
        raise ValueError("train features must contain FFR for correlation-based selection.")

    y = train["FFR"].astype(float)
    rows = []
    for c in feature_cols:
        x = train[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Constant feature has undefined corr; set to 0.
        if float(x.std()) < 1e-12:
            sp = 0.0
            pe = 0.0
        else:
            try:
                sp = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
                if not np.isfinite(sp):
                    sp = 0.0
            except Exception:
                sp = 0.0

            try:
                pe = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
                if not np.isfinite(pe):
                    pe = 0.0
            except Exception:
                pe = 0.0

        rows.append({
            "feature": c,
            "spearman_ffr": sp,
            "abs_spearman_ffr": abs(sp),
            "pearson_ffr": pe,
            "abs_pearson_ffr": abs(pe),
            "std": float(x.std()),
            "mean": float(x.mean()),
        })

    rank = pd.DataFrame(rows)
    rank = rank.sort_values(["abs_spearman_ffr", "abs_pearson_ffr"], ascending=False).reset_index(drop=True)
    return rank


def write_selected_csvs(train, test, selected_features, out_train, out_test):
    train_cols = ["serial_no"] + [c for c in ["FFR", "ffr_class"] if c in train.columns] + selected_features
    test_cols = ["serial_no"] + selected_features

    missing_train = [c for c in train_cols if c not in train.columns]
    missing_test = [c for c in test_cols if c not in test.columns]
    if missing_train:
        raise ValueError(f"Missing train columns: {missing_train[:10]}")
    if missing_test:
        raise ValueError(f"Missing test columns: {missing_test[:10]}")

    train[train_cols].to_csv(out_train, index=False)
    test[test_cols].to_csv(out_test, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-merged", default="step7_0_feature_v2_outputs/train_features_v2_merged.csv")
    parser.add_argument("--test-merged", default="step7_0_feature_v2_outputs/test_public_features_v2_merged.csv")
    parser.add_argument("--train-v2-only", default="step7_0_feature_v2_outputs/train_features_v2_only.csv")
    parser.add_argument("--test-v2-only", default="step7_0_feature_v2_outputs/test_public_features_v2_only.csv")
    parser.add_argument("--out-dir", default="step7_5_feature_selection_outputs")
    parser.add_argument("--merged-top-k", nargs="+", type=int, default=[200, 300, 400])
    parser.add_argument("--v2-only-top-k", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2025, 123, 777, 3407, 1004])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_merged = clean_dataframe(pd.read_csv(args.train_merged))
    test_merged = clean_dataframe(pd.read_csv(args.test_merged))
    train_v2 = clean_dataframe(pd.read_csv(args.train_v2_only))
    test_v2 = clean_dataframe(pd.read_csv(args.test_v2_only))

    # Ensure serial_no is str.
    for df in [train_merged, test_merged, train_v2, test_v2]:
        df["serial_no"] = df["serial_no"].astype(str)

    # Ranking for merged features.
    merged_cols = get_numeric_feature_cols(train_merged, v2_only=False)
    merged_rank = compute_corr_ranking(train_merged, merged_cols)
    merged_rank.to_csv(out_dir / "merged_feature_corr_ranking.csv", index=False)

    # Ranking for v2-only features.
    v2_cols = get_numeric_feature_cols(train_v2, v2_only=True)
    v2_rank = compute_corr_ranking(train_v2, v2_cols)
    v2_rank.to_csv(out_dir / "v2only_feature_corr_ranking.csv", index=False)

    generated = []

    for k in args.merged_top_k:
        selected = merged_rank.head(k)["feature"].tolist()
        train_path = out_dir / f"train_merged_top{k}_corr.csv"
        test_path = out_dir / f"test_merged_top{k}_corr.csv"
        write_selected_csvs(train_merged, test_merged, selected, train_path, test_path)
        generated.append(("merged", k, train_path, test_path))

    for k in args.v2_only_top_k:
        selected = v2_rank.head(k)["feature"].tolist()
        train_path = out_dir / f"train_v2only_top{k}_corr.csv"
        test_path = out_dir / f"test_v2only_top{k}_corr.csv"
        write_selected_csvs(train_v2, test_v2, selected, train_path, test_path)
        generated.append(("v2only", k, train_path, test_path))

    # Write training commands.
    seed_str = " ".join(str(s) for s in args.seeds)
    cmd_lines = []
    cmd_lines.append("#!/usr/bin/env bash")
    cmd_lines.append("set -e")
    cmd_lines.append("")
    cmd_lines.append("# Generated training commands for Step 7-5 feature-selection experiments.")
    cmd_lines.append("")

    for kind, k, train_path, test_path in generated:
        out_name = f"step7_5_mlp_{kind}_top{k}_7seed_outputs"
        cmd_lines.append(f"# {kind} top{k}")
        cmd_lines.append("python step3_3_advanced_mlp.py \\")
        cmd_lines.append(f"  --train-features {train_path} \\")
        cmd_lines.append(f"  --test-features {test_path} \\")
        cmd_lines.append(f"  --out-dir {out_name} \\")
        cmd_lines.append("  --hidden-dims 512 256 128 \\")
        cmd_lines.append("  --dropout 0.25 \\")
        cmd_lines.append("  --lr 0.0007 \\")
        cmd_lines.append("  --weight-decay 0.0001 \\")
        cmd_lines.append(f"  --seeds {seed_str}")
        cmd_lines.append("")

    run_script = out_dir / "run_step7_5_training_commands.sh"
    run_script.write_text("\n".join(cmd_lines), encoding="utf-8")
    run_script.chmod(0o755)

    # Summary.
    lines = []
    lines.append("Step 7-5 Feature Selection Preparation Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"merged train shape: {train_merged.shape}")
    lines.append(f"merged test shape: {test_merged.shape}")
    lines.append(f"v2-only train shape: {train_v2.shape}")
    lines.append(f"v2-only test shape: {test_v2.shape}")
    lines.append("")
    lines.append(f"numeric merged feature candidates: {len(merged_cols)}")
    lines.append(f"numeric v2-only feature candidates: {len(v2_cols)}")
    lines.append("")
    lines.append("[Top 30 merged features]")
    lines.append(merged_rank.head(30).to_string(index=False))
    lines.append("")
    lines.append("[Top 30 v2-only features]")
    lines.append(v2_rank.head(30).to_string(index=False))
    lines.append("")
    lines.append("[Generated datasets]")
    for kind, k, train_path, test_path in generated:
        lines.append(f"{kind} top{k}: {train_path}, {test_path}")
    lines.append("")
    lines.append(f"Training script: {run_script}")

    summary_path = out_dir / "step7_5_feature_selection_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] generated selected feature datasets in {out_dir}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {run_script}")
    print("")
    print("Recommended first runs:")
    print(f"bash {run_script}")


if __name__ == "__main__":
    main()
