#!/usr/bin/env python3
# phase3_probability_ensemble.py
#
# Combine test_public probability CSVs from multiple models.
#
# Each input probability CSV must contain:
#   serial_no, p_risk
#
# Example:
#   python phase3_probability_ensemble.py \
#     --inputs mlp.csv extra_trees.csv random_forest.csv \
#     --weights 0.8 0.15 0.05 \
#     --out-dir phase3_ensemble_outputs \
#     --prefix mlp080_et015_rf005 \
#     --thresholds 0.45 0.48 0.51 0.54 0.57
#
# Output:
#   ensemble_probabilities.csv
#   quickcheck_*.csv files
#   ensemble_summary.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_thresholds(args):
    if args.thresholds:
        return [float(x) for x in args.thresholds]
    values = np.arange(args.start, args.end + 1e-12, args.step)
    return [round(float(x), 6) for x in values]


def load_prob(path):
    df = pd.read_csv(path)
    required = {"serial_no", "p_risk"}
    if not required.issubset(df.columns):
        # Accept some common alternatives.
        p_cols = [c for c in df.columns if c.lower() in {"risk_prob", "prob_risk", "p0", "p_class0"}]
        if "serial_no" in df.columns and p_cols:
            df = df.rename(columns={p_cols[0]: "p_risk"})
        else:
            raise ValueError(f"{path} must contain serial_no,p_risk. Columns={df.columns.tolist()}")
    return df[["serial_no", "p_risk"]].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Probability CSVs with serial_no,p_risk")
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="ensemble")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--start", type=float, default=0.40)
    parser.add_argument("--end", type=float, default=0.65)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    if len(args.inputs) != len(args.weights):
        raise ValueError("--inputs and --weights must have the same length")

    weights = np.asarray(args.weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weights = weights / weights.sum()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = [load_prob(p) for p in args.inputs]
    merged = dfs[0].rename(columns={"p_risk": "p_risk_0"})
    for i, df in enumerate(dfs[1:], start=1):
        merged = merged.merge(
            df.rename(columns={"p_risk": f"p_risk_{i}"}),
            on="serial_no",
            how="inner",
        )

    if len(merged) != len(dfs[0]):
        raise RuntimeError("Some serial_no values did not match across probability files.")

    probs = np.stack([merged[f"p_risk_{i}"].values for i in range(len(dfs))], axis=1)
    p_ens = (probs * weights.reshape(1, -1)).sum(axis=1)

    prob_df = pd.DataFrame({
        "serial_no": merged["serial_no"].astype(str).values,
        "p_risk": p_ens,
    })
    prob_path = out_dir / f"{args.prefix}_probabilities.csv"
    prob_df.to_csv(prob_path, index=False)

    thresholds = parse_thresholds(args)
    summary_rows = []
    for thr in thresholds:
        pred = np.where(p_ens >= thr, 0, 1).astype(int)
        tag = f"{thr:.3f}".replace(".", "p")
        out_path = out_dir / f"quickcheck_{args.prefix}_thr{tag}.csv"
        pd.DataFrame({
            "serial_no": merged["serial_no"].astype(str).values,
            "ffr_class": pred,
        }).to_csv(out_path, index=False)

        summary_rows.append({
            "threshold": thr,
            "filename": out_path.name,
            "pred_class0_count": int((pred == 0).sum()),
            "pred_class1_count": int((pred == 1).sum()),
            "pred_class0_ratio": float((pred == 0).mean()),
            "quickcheck_score": "",
            "weights": ",".join(f"{w:.6f}" for w in weights),
            "inputs": "|".join(args.inputs),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{args.prefix}_ensemble_summary.csv", index=False)

    print(f"[DONE] saved ensemble probabilities: {prob_path}")
    print(f"[DONE] generated {len(thresholds)} quickcheck files")
    print(summary[["threshold", "filename", "pred_class0_count", "pred_class1_count", "pred_class0_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
