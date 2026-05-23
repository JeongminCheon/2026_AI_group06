#!/usr/bin/env python3
# step7_3_make_a5_v2_ensemble_quickchecks.py
#
# Step 7-3:
#   Make quickcheck CSVs for A5 + v2 probability ensemble.
#
# Current best:
#   A5 = 0.5 * p_3seed + 0.5 * p_7seed
#   threshold = 0.50
#   public score = 68.44
#
# New candidate:
#   p_final = w_a5 * p_A5 + w_v2 * p_v2
#
# where:
#   p_A5 = 0.5 * p_3seed + 0.5 * p_7seed
#
# Input CSVs must contain:
#   serial_no,p_risk
#
# Output CSV format:
#   serial_no,ffr_class
#
# Rule:
#   p_risk = P(FFR < 0.8)
#   if p_risk >= threshold -> ffr_class = 0
#   else                   -> ffr_class = 1
#
# Recommended run:
#
# python step7_3_make_a5_v2_ensemble_quickchecks.py \
#   --prob-3seed step3_6_seed_ensemble_mlp_outputs/test_public_probabilities.csv \
#   --prob-7seed step5_0_seed_ensemble_7seeds_outputs/test_public_probabilities.csv \
#   --prob-v2 step7_2_mlp_v2_7seed_outputs/test_public_probabilities.csv \
#   --out-dir step7_3_a5_v2_ensemble_outputs

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_prob_csv(path: str, name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} probability file not found: {path}")

    df = pd.read_csv(path)

    if "serial_no" not in df.columns:
        raise ValueError(f"{path} must contain serial_no. Columns={df.columns.tolist()}")

    if "p_risk" not in df.columns:
        candidates = [
            "risk_prob",
            "prob_risk",
            "p_class0",
            "p0",
            "prob_class0",
            "probability",
        ]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise ValueError(f"{path} must contain p_risk. Columns={df.columns.tolist()}")
        df = df.rename(columns={found: "p_risk"})

    out = df[["serial_no", "p_risk"]].copy()
    out["serial_no"] = out["serial_no"].astype(str)
    out["p_risk"] = out["p_risk"].astype(float)
    return out


def make_prediction(p_risk: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p_risk >= threshold, 0, 1).astype(int)


def save_quickcheck(serial_no: pd.Series, pred: np.ndarray, out_path: Path):
    pd.DataFrame({
        "serial_no": serial_no.astype(str).values,
        "ffr_class": pred.astype(int),
    }).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prob-3seed",
        default="step3_6_seed_ensemble_mlp_outputs/test_public_probabilities.csv",
        help="3-seed MLP probability CSV with serial_no,p_risk",
    )
    parser.add_argument(
        "--prob-7seed",
        default="step5_0_seed_ensemble_7seeds_outputs/test_public_probabilities.csv",
        help="7-seed MLP probability CSV with serial_no,p_risk",
    )
    parser.add_argument(
        "--prob-v2",
        default="step7_2_mlp_v2_7seed_outputs/test_public_probabilities.csv",
        help="v2 merged-feature 7-seed MLP probability CSV with serial_no,p_risk",
    )
    parser.add_argument(
        "--out-dir",
        default="step7_3_a5_v2_ensemble_outputs",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional A5 reference quickcheck CSV for diff count. If omitted, generated BASE_A5 is used.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p3 = load_prob_csv(args.prob_3seed, "3-seed")
    p7 = load_prob_csv(args.prob_7seed, "7-seed")
    pv2 = load_prob_csv(args.prob_v2, "v2")

    df = p3.rename(columns={"p_risk": "p3"}).merge(
        p7.rename(columns={"p_risk": "p7"}),
        on="serial_no",
        how="inner",
    ).merge(
        pv2.rename(columns={"p_risk": "pv2"}),
        on="serial_no",
        how="inner",
    )

    if len(df) != len(p3):
        raise RuntimeError(f"serial_no mismatch: 3seed rows={len(p3)}, merged rows={len(df)}")

    df["p_a5"] = 0.5 * df["p3"] + 0.5 * df["p7"]

    # Candidate set:
    # 1) A5 + v2 small weights at threshold 0.50
    # 2) A5 + v2 small weights at threshold 0.52 and 0.54, because v2 OOF thresholds shifted higher
    # 3) v2-only threshold sweep for reference
    candidates = [
        # A5 baseline
        {
            "id": "BASE_A5",
            "name": "a5_thr050",
            "wa5": 1.00,
            "wv2": 0.00,
            "threshold": 0.50,
            "description": "A5 baseline: A5 1.00 + v2 0.00, threshold 0.50",
        },

        # A5 + v2, threshold 0.50
        {
            "id": "D1",
            "name": "a5_090_v2_010_thr050",
            "wa5": 0.90,
            "wv2": 0.10,
            "threshold": 0.50,
            "description": "A5 0.90 + v2 0.10, threshold 0.50",
        },
        {
            "id": "D2",
            "name": "a5_080_v2_020_thr050",
            "wa5": 0.80,
            "wv2": 0.20,
            "threshold": 0.50,
            "description": "A5 0.80 + v2 0.20, threshold 0.50",
        },
        {
            "id": "D3",
            "name": "a5_070_v2_030_thr050",
            "wa5": 0.70,
            "wv2": 0.30,
            "threshold": 0.50,
            "description": "A5 0.70 + v2 0.30, threshold 0.50",
        },

        # A5 + v2, threshold 0.52
        {
            "id": "D4",
            "name": "a5_090_v2_010_thr052",
            "wa5": 0.90,
            "wv2": 0.10,
            "threshold": 0.52,
            "description": "A5 0.90 + v2 0.10, threshold 0.52",
        },
        {
            "id": "D5",
            "name": "a5_080_v2_020_thr052",
            "wa5": 0.80,
            "wv2": 0.20,
            "threshold": 0.52,
            "description": "A5 0.80 + v2 0.20, threshold 0.52",
        },
        {
            "id": "D6",
            "name": "a5_070_v2_030_thr052",
            "wa5": 0.70,
            "wv2": 0.30,
            "threshold": 0.52,
            "description": "A5 0.70 + v2 0.30, threshold 0.52",
        },

        # A5 + v2, threshold 0.54
        {
            "id": "D7",
            "name": "a5_090_v2_010_thr054",
            "wa5": 0.90,
            "wv2": 0.10,
            "threshold": 0.54,
            "description": "A5 0.90 + v2 0.10, threshold 0.54",
        },
        {
            "id": "D8",
            "name": "a5_080_v2_020_thr054",
            "wa5": 0.80,
            "wv2": 0.20,
            "threshold": 0.54,
            "description": "A5 0.80 + v2 0.20, threshold 0.54",
        },
        {
            "id": "D9",
            "name": "a5_070_v2_030_thr054",
            "wa5": 0.70,
            "wv2": 0.30,
            "threshold": 0.54,
            "description": "A5 0.70 + v2 0.30, threshold 0.54",
        },

        # v2-only references
        {
            "id": "V1",
            "name": "v2_only_thr060",
            "wa5": 0.00,
            "wv2": 1.00,
            "threshold": 0.60,
            "description": "v2 only, threshold 0.60",
        },
        {
            "id": "V2",
            "name": "v2_only_thr065",
            "wa5": 0.00,
            "wv2": 1.00,
            "threshold": 0.65,
            "description": "v2 only, threshold 0.65",
        },
        {
            "id": "V3",
            "name": "v2_only_thr070",
            "wa5": 0.00,
            "wv2": 1.00,
            "threshold": 0.70,
            "description": "v2 only, threshold 0.70",
        },
        {
            "id": "V4",
            "name": "v2_only_thr075",
            "wa5": 0.00,
            "wv2": 1.00,
            "threshold": 0.75,
            "description": "v2 only, threshold 0.75",
        },
    ]

    pred_by_id = {}
    summary_rows = []

    for c in candidates:
        p = c["wa5"] * df["p_a5"].values + c["wv2"] * df["pv2"].values
        pred = make_prediction(p, c["threshold"])
        pred_by_id[c["id"]] = pred

        filename = f"quickcheck_step7_3_{c['id']}_{c['name']}.csv"
        save_quickcheck(df["serial_no"], pred, out_dir / filename)

        summary_rows.append({
            "candidate_id": c["id"],
            "filename": filename,
            "description": c["description"],
            "wa5": c["wa5"],
            "wv2": c["wv2"],
            "threshold": c["threshold"],
            "pred_class0_count": int((pred == 0).sum()),
            "pred_class1_count": int((pred == 1).sum()),
            "pred_class0_ratio": float((pred == 0).mean()),
            "quickcheck_score": "",
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "step7_3_a5_v2_ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Difference summary versus A5.
    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv)
        if not {"serial_no", "ffr_class"}.issubset(ref.columns):
            raise ValueError("reference_csv must contain serial_no,ffr_class")
        ref["serial_no"] = ref["serial_no"].astype(str)
        ref = df[["serial_no"]].merge(ref[["serial_no", "ffr_class"]], on="serial_no", how="left")
        if ref["ffr_class"].isna().any():
            raise RuntimeError("reference_csv is missing some serial_no values.")
        ref_pred = ref["ffr_class"].astype(int).values
        ref_name = Path(args.reference_csv).name
    else:
        ref_pred = pred_by_id["BASE_A5"]
        ref_name = "BASE_A5"

    diff_rows = []
    for c in candidates:
        pred = pred_by_id[c["id"]]
        diff = pred != ref_pred
        diff_rows.append({
            "candidate_id": c["id"],
            "filename": f"quickcheck_step7_3_{c['id']}_{c['name']}.csv",
            "reference": ref_name,
            "num_diff_from_reference": int(diff.sum()),
            "diff_serial_no": ",".join(df.loc[diff, "serial_no"].astype(str).tolist()),
        })

    diff_df = pd.DataFrame(diff_rows)
    diff_path = out_dir / "step7_3_prediction_diff_summary.csv"
    diff_df.to_csv(diff_path, index=False)

    print(f"[DONE] generated {len(candidates)} quickcheck files in {out_dir}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {diff_path}")
    print("")
    print("[Suggested first submissions]")
    first_ids = ["D1", "D2", "D3", "D4", "D5", "D6", "V2", "V3"]
    print(summary[summary["candidate_id"].isin(first_ids)][[
        "candidate_id",
        "filename",
        "pred_class0_count",
        "pred_class1_count",
        "pred_class0_ratio",
        "description",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
