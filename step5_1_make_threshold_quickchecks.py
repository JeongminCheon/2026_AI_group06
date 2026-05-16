#!/usr/bin/env python3
# phase1_make_threshold_quickchecks.py
#
# Make multiple quickcheck CSV files from a probability file.
#
# Input CSV must contain:
#   serial_no, p_risk
#
# Output CSV format:
#   serial_no, ffr_class
#
# Definition:
#   p_risk = P(FFR < 0.8)
#   if p_risk >= threshold -> ffr_class = 0
#   else                   -> ffr_class = 1

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_thresholds(args):
    if args.thresholds:
        return [float(x) for x in args.thresholds]

    values = np.arange(args.start, args.end + 1e-12, args.step)
    return [round(float(x), 6) for x in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-csv", required=True, help="CSV with serial_no,p_risk")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="quickcheck_mlp")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--start", type=float, default=0.40)
    parser.add_argument("--end", type=float, default=0.65)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--score-note", action="store_true", help="Create a blank score note table.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.prob_csv)
    required = {"serial_no", "p_risk"}
    if not required.issubset(df.columns):
        raise ValueError(f"{args.prob_csv} must contain columns {required}, got {df.columns.tolist()}")

    thresholds = parse_thresholds(args)

    summary_rows = []
    for thr in thresholds:
        pred = np.where(df["p_risk"].values >= thr, 0, 1).astype(int)
        class0_ratio = float((pred == 0).mean())
        class1_ratio = float((pred == 1).mean())

        tag = f"{thr:.3f}".replace(".", "p")
        out_path = out_dir / f"{args.prefix}_thr{tag}.csv"

        out_df = pd.DataFrame({
            "serial_no": df["serial_no"].astype(str).values,
            "ffr_class": pred,
        })
        out_df.to_csv(out_path, index=False)

        summary_rows.append({
            "threshold": thr,
            "filename": out_path.name,
            "num_samples": len(out_df),
            "pred_class0_count": int((pred == 0).sum()),
            "pred_class1_count": int((pred == 1).sum()),
            "pred_class0_ratio": class0_ratio,
            "pred_class1_ratio": class1_ratio,
            "quickcheck_score": "",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{args.prefix}_threshold_summary.csv", index=False)

    print(f"[DONE] generated {len(thresholds)} quickcheck files in {out_dir}")
    print(f"[CHECK] {out_dir / (args.prefix + '_threshold_summary.csv')}")
    print(summary[["threshold", "filename", "pred_class0_count", "pred_class1_count", "pred_class0_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
