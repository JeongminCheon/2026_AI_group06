#!/usr/bin/env python3
# step6_2_make_a5_finetune_quickchecks.py
#
# Step 6-2:
#   Generate fine-tuning quickcheck CSVs around the best Step 6-1 candidate:
#
#   A5 = 0.50 * p_3seed + 0.50 * p_7seed, threshold = 0.50
#   public quickcheck score = 68.44
#
# Candidates generated:
#   C1: 3seed 0.50 + 7seed 0.50, threshold 0.495
#   C2: 3seed 0.50 + 7seed 0.50, threshold 0.505
#   C3: 3seed 0.50 + 7seed 0.50, threshold 0.490
#   C4: 3seed 0.50 + 7seed 0.50, threshold 0.510
#   C5: 3seed 0.60 + 7seed 0.40, threshold 0.500

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
        candidates = ["risk_prob", "prob_risk", "p_class0", "p0", "prob_class0", "probability"]
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
    parser.add_argument("--out-dir", default="step6_2_a5_finetune_outputs")
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional reference quickcheck CSV, e.g. Step6 A5, for diff count.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p3 = load_prob_csv(args.prob_3seed, "3-seed")
    p7 = load_prob_csv(args.prob_7seed, "7-seed")

    df = p3.rename(columns={"p_risk": "p3"}).merge(
        p7.rename(columns={"p_risk": "p7"}),
        on="serial_no",
        how="inner",
    )

    if len(df) != len(p3):
        raise RuntimeError(f"serial_no mismatch: 3seed rows={len(p3)}, merged rows={len(df)}")

    candidates = [
        {
            "id": "C1",
            "name": "3seed050_7seed050_thr0495",
            "w3": 0.50,
            "w7": 0.50,
            "threshold": 0.495,
            "description": "3seed 0.50 + 7seed 0.50, threshold 0.495",
        },
        {
            "id": "C2",
            "name": "3seed050_7seed050_thr0505",
            "w3": 0.50,
            "w7": 0.50,
            "threshold": 0.505,
            "description": "3seed 0.50 + 7seed 0.50, threshold 0.505",
        },
        {
            "id": "C3",
            "name": "3seed050_7seed050_thr0490",
            "w3": 0.50,
            "w7": 0.50,
            "threshold": 0.490,
            "description": "3seed 0.50 + 7seed 0.50, threshold 0.490",
        },
        {
            "id": "C4",
            "name": "3seed050_7seed050_thr0510",
            "w3": 0.50,
            "w7": 0.50,
            "threshold": 0.510,
            "description": "3seed 0.50 + 7seed 0.50, threshold 0.510",
        },
        {
            "id": "C5",
            "name": "3seed060_7seed040_thr0500",
            "w3": 0.60,
            "w7": 0.40,
            "threshold": 0.500,
            "description": "3seed 0.60 + 7seed 0.40, threshold 0.500",
        },
        {
            "id": "BASE_A5",
            "name": "3seed050_7seed050_thr0500",
            "w3": 0.50,
            "w7": 0.50,
            "threshold": 0.500,
            "description": "A5 baseline: 3seed 0.50 + 7seed 0.50, threshold 0.500",
        },
    ]

    pred_by_id = {}
    summary_rows = []

    for c in candidates:
        p = c["w3"] * df["p3"].values + c["w7"] * df["p7"].values
        pred = make_prediction(p, c["threshold"])
        pred_by_id[c["id"]] = pred

        filename = f"quickcheck_step6_2_{c['id']}_{c['name']}.csv"
        save_quickcheck(df["serial_no"], pred, out_dir / filename)

        summary_rows.append({
            "candidate_id": c["id"],
            "filename": filename,
            "description": c["description"],
            "w3seed": c["w3"],
            "w7seed": c["w7"],
            "threshold": c["threshold"],
            "pred_class0_count": int((pred == 0).sum()),
            "pred_class1_count": int((pred == 1).sum()),
            "pred_class0_ratio": float((pred == 0).mean()),
            "quickcheck_score": "",
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "step6_2_a5_finetune_summary.csv"
    summary.to_csv(summary_path, index=False)

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
        ref_name = "BASE_A5_3seed050_7seed050_thr0500"

    diff_rows = []
    for c in candidates:
        pred = pred_by_id[c["id"]]
        diff = pred != ref_pred
        diff_rows.append({
            "candidate_id": c["id"],
            "filename": f"quickcheck_step6_2_{c['id']}_{c['name']}.csv",
            "reference": ref_name,
            "num_diff_from_reference": int(diff.sum()),
            "diff_serial_no": ",".join(df.loc[diff, "serial_no"].astype(str).tolist()),
        })

    diff_df = pd.DataFrame(diff_rows)
    diff_path = out_dir / "step6_2_prediction_diff_summary.csv"
    diff_df.to_csv(diff_path, index=False)

    print(f"[DONE] generated {len(candidates)} quickcheck files in {out_dir}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {diff_path}")
    print("")
    print(summary[[
        "candidate_id",
        "filename",
        "pred_class0_count",
        "pred_class1_count",
        "pred_class0_ratio",
        "description",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
