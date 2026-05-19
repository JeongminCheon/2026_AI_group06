#!/usr/bin/env python3
# step6_1_make_soft_ensemble_quickchecks.py
#
# Step 6-1:
#   Make quickcheck CSVs for Batch A and Batch B soft-ensemble candidates.
#
# Required input probability CSVs:
#   - 3-seed MLP probabilities: serial_no,p_risk
#   - 7-seed MLP probabilities: serial_no,p_risk
#   - Attention MIL s64 probabilities: serial_no,p_risk
#
# Output:
#   - quickcheck_step6_*.csv
#   - step6_1_ensemble_summary.csv
#   - step6_1_prediction_diff_summary.csv
#
# Label rule:
#   p_risk = P(FFR < 0.8)
#   if p_risk >= threshold -> ffr_class = 0
#   else                   -> ffr_class = 1

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
        raise ValueError(f"{path} must contain serial_no column. Got: {df.columns.tolist()}")

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
            raise ValueError(
                f"{path} must contain p_risk column. Got: {df.columns.tolist()}"
            )
        df = df.rename(columns={found: "p_risk"})

    out = df[["serial_no", "p_risk"]].copy()
    out["serial_no"] = out["serial_no"].astype(str)
    out["p_risk"] = out["p_risk"].astype(float)
    return out


def merge_probabilities(p3: pd.DataFrame, p7: pd.DataFrame, pattn: pd.DataFrame) -> pd.DataFrame:
    merged = p3.rename(columns={"p_risk": "p3"})
    merged = merged.merge(
        p7.rename(columns={"p_risk": "p7"}),
        on="serial_no",
        how="inner",
    )
    merged = merged.merge(
        pattn.rename(columns={"p_risk": "p_attn"}),
        on="serial_no",
        how="inner",
    )

    if len(merged) != len(p3):
        raise RuntimeError(
            f"serial_no mismatch: 3seed has {len(p3)} rows, merged has {len(merged)} rows."
        )

    return merged


def make_prediction(p_risk: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p_risk >= threshold, 0, 1).astype(int)


def save_quickcheck(serial_no: pd.Series, pred: np.ndarray, out_path: Path):
    out_df = pd.DataFrame({
        "serial_no": serial_no.astype(str).values,
        "ffr_class": pred.astype(int),
    })
    out_df.to_csv(out_path, index=False)


def candidate_probability(df: pd.DataFrame, recipe: dict) -> np.ndarray:
    return (
        recipe.get("w3", 0.0) * df["p3"].values
        + recipe.get("w7", 0.0) * df["p7"].values
        + recipe.get("wa", 0.0) * df["p_attn"].values
    )


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
        "--prob-attn",
        default="step5_3_attention_s64_outputs/test_public_probabilities.csv",
        help="Attention MIL s64 probability CSV with serial_no,p_risk",
    )
    parser.add_argument(
        "--out-dir",
        default="step6_1_soft_ensemble_outputs",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional reference quickcheck CSV for diff count. If omitted, use generated A2 candidate as reference.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p3 = load_prob_csv(args.prob_3seed, "3-seed")
    p7 = load_prob_csv(args.prob_7seed, "7-seed")
    pattn = load_prob_csv(args.prob_attn, "attention")

    df = merge_probabilities(p3, p7, pattn)

    recipes = [
        # Batch A: 3seed + 7seed
        {"id": "A1", "name": "3seed075_7seed025_thr048", "w3": 0.75, "w7": 0.25, "wa": 0.00, "threshold": 0.48, "description": "3seed 0.75 + 7seed 0.25, threshold 0.48"},
        {"id": "A2", "name": "3seed050_7seed050_thr048", "w3": 0.50, "w7": 0.50, "wa": 0.00, "threshold": 0.48, "description": "3seed 0.50 + 7seed 0.50, threshold 0.48"},
        {"id": "A3", "name": "3seed025_7seed075_thr048", "w3": 0.25, "w7": 0.75, "wa": 0.00, "threshold": 0.48, "description": "3seed 0.25 + 7seed 0.75, threshold 0.48"},
        {"id": "A4", "name": "3seed050_7seed050_thr046", "w3": 0.50, "w7": 0.50, "wa": 0.00, "threshold": 0.46, "description": "3seed 0.50 + 7seed 0.50, threshold 0.46"},
        {"id": "A5", "name": "3seed050_7seed050_thr050", "w3": 0.50, "w7": 0.50, "wa": 0.00, "threshold": 0.50, "description": "3seed 0.50 + 7seed 0.50, threshold 0.50"},
        {"id": "A6", "name": "3seed050_7seed050_thr052", "w3": 0.50, "w7": 0.50, "wa": 0.00, "threshold": 0.52, "description": "3seed 0.50 + 7seed 0.50, threshold 0.52"},

        # Batch B: MLP + Attention s64
        {"id": "B1", "name": "7seed095_attn005_thr048", "w3": 0.00, "w7": 0.95, "wa": 0.05, "threshold": 0.48, "description": "7seed 0.95 + attention 0.05, threshold 0.48"},
        {"id": "B2", "name": "7seed090_attn010_thr048", "w3": 0.00, "w7": 0.90, "wa": 0.10, "threshold": 0.48, "description": "7seed 0.90 + attention 0.10, threshold 0.48"},
        {"id": "B3", "name": "3seed095_attn005_thr048", "w3": 0.95, "w7": 0.00, "wa": 0.05, "threshold": 0.48, "description": "3seed 0.95 + attention 0.05, threshold 0.48"},
        {"id": "B4", "name": "3seed090_attn010_thr048", "w3": 0.90, "w7": 0.00, "wa": 0.10, "threshold": 0.48, "description": "3seed 0.90 + attention 0.10, threshold 0.48"},
    ]

    baseline_recipes = [
        {"id": "BASE_3SEED_048", "name": "baseline_3seed_thr048", "w3": 1.0, "w7": 0.0, "wa": 0.0, "threshold": 0.48, "description": "3seed only, threshold 0.48"},
        {"id": "BASE_7SEED_048", "name": "baseline_7seed_thr048", "w3": 0.0, "w7": 1.0, "wa": 0.0, "threshold": 0.48, "description": "7seed only, threshold 0.48"},
    ]

    all_recipes = recipes + baseline_recipes

    pred_by_id = {}
    summary_rows = []

    for recipe in all_recipes:
        p = candidate_probability(df, recipe)
        pred = make_prediction(p, recipe["threshold"])

        filename = f"quickcheck_step6_{recipe['id']}_{recipe['name']}.csv"
        out_path = out_dir / filename
        save_quickcheck(df["serial_no"], pred, out_path)
        pred_by_id[recipe["id"]] = pred

        summary_rows.append({
            "candidate_id": recipe["id"],
            "filename": filename,
            "description": recipe["description"],
            "w3seed": recipe["w3"],
            "w7seed": recipe["w7"],
            "w_attention": recipe["wa"],
            "threshold": recipe["threshold"],
            "pred_class0_count": int((pred == 0).sum()),
            "pred_class1_count": int((pred == 1).sum()),
            "pred_class0_ratio": float((pred == 0).mean()),
            "quickcheck_score": "",
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "step6_1_ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)

    if args.reference_csv is not None:
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
        ref_pred = pred_by_id["A2"]
        ref_name = "A2_3seed050_7seed050_thr048"

    diff_rows = []
    for recipe in all_recipes:
        pred = pred_by_id[recipe["id"]]
        diff = pred != ref_pred
        diff_rows.append({
            "candidate_id": recipe["id"],
            "filename": f"quickcheck_step6_{recipe['id']}_{recipe['name']}.csv",
            "reference": ref_name,
            "num_diff_from_reference": int(diff.sum()),
            "diff_serial_no": ",".join(df.loc[diff, "serial_no"].astype(str).tolist()),
        })

    diff_summary = pd.DataFrame(diff_rows)
    diff_path = out_dir / "step6_1_prediction_diff_summary.csv"
    diff_summary.to_csv(diff_path, index=False)

    print(f"[DONE] generated {len(all_recipes)} quickcheck files in {out_dir}")
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {diff_path}")
    print("")
    print("[Main candidates to submit]")
    keep_ids = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 5)]
    print(summary[summary["candidate_id"].isin(keep_ids)][
        ["candidate_id", "filename", "pred_class0_count", "pred_class1_count", "pred_class0_ratio", "description"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
