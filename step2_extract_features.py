#!/usr/bin/env python3
# step2_extract_features.py

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


EPS = 1e-8


def read_mha(path: Path) -> np.ndarray:
    """
    Read .mha mask volume.

    Expected:
        shape: (num_frames, 256, 256)
        values: 0 background, 1 lumen, 2 plaque
    """
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return arr


def safe_divide(a, b, eps: float = EPS):
    return a / (b + eps)


def longest_true_run(mask: np.ndarray) -> int:
    """
    Return the longest consecutive True segment length in a 1D boolean array.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0

    max_run = 0
    current = 0

    for v in mask:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0

    return int(max_run)


def add_basic_stats(prefix: str, x: np.ndarray, out: dict):
    """
    Add common statistics for a 1D array.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        stats = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    else:
        stats = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "median": float(np.median(x)),
            "p01": float(np.percentile(x, 1)),
            "p05": float(np.percentile(x, 5)),
            "p10": float(np.percentile(x, 10)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
        }

    for k, v in stats.items():
        out[f"{prefix}_{k}"] = v


def segment_indices(n: int, start_ratio: float, end_ratio: float) -> slice:
    """
    Convert ratio range into slice indices.
    """
    start = int(round(n * start_ratio))
    end = int(round(n * end_ratio))
    start = max(0, min(n, start))
    end = max(start + 1, min(n, end))
    return slice(start, end)


def add_region_stats(
    region_name: str,
    lumen_area: np.ndarray,
    plaque_area: np.ndarray,
    vessel_area: np.ndarray,
    plaque_burden: np.ndarray,
    out: dict,
):
    """
    Add proximal/mid/distal or local-window summary features.
    """
    add_basic_stats(f"{region_name}_lumen", lumen_area, out)
    add_basic_stats(f"{region_name}_plaque", plaque_area, out)
    add_basic_stats(f"{region_name}_vessel", vessel_area, out)
    add_basic_stats(f"{region_name}_burden", plaque_burden, out)

    out[f"{region_name}_lumen_sum"] = float(np.sum(lumen_area))
    out[f"{region_name}_plaque_sum"] = float(np.sum(plaque_area))
    out[f"{region_name}_vessel_sum"] = float(np.sum(vessel_area))
    out[f"{region_name}_global_burden"] = float(
        safe_divide(np.sum(plaque_area), np.sum(vessel_area))
    )


def extract_features_from_array(arr: np.ndarray, serial_no: str = "") -> dict:
    """
    Extract volume-level and slice-wise summary features from a mask volume.
    """
    out = {
        "serial_no": serial_no,
        "read_ok": True,
        "error": "",
    }

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape}")

    num_frames, height, width = arr.shape

    out["num_frames"] = int(num_frames)
    out["height"] = int(height)
    out["width"] = int(width)
    out["dtype_is_uint8"] = int(arr.dtype == np.uint8)

    unique_values = np.unique(arr)
    out["unique_values"] = ",".join(map(str, unique_values.tolist()))
    out["only_0_1_2"] = int(set(unique_values.tolist()).issubset({0, 1, 2}))

    lumen = arr == 1
    plaque = arr == 2
    vessel = lumen | plaque

    lumen_area = lumen.sum(axis=(1, 2)).astype(np.float64)
    plaque_area = plaque.sum(axis=(1, 2)).astype(np.float64)
    vessel_area = vessel.sum(axis=(1, 2)).astype(np.float64)

    nonempty = vessel_area > 0
    out["nonempty_frame_count"] = int(nonempty.sum())
    out["empty_frame_count"] = int((~nonempty).sum())
    out["nonempty_frame_ratio"] = float(safe_divide(nonempty.sum(), num_frames))
    out["empty_frame_ratio"] = float(safe_divide((~nonempty).sum(), num_frames))

    # If some frames are empty, feature statistics should be computed on nonempty frames.
    # In Step 0, all uploaded data had lumen/plaque in every case, but this makes the code robust.
    lumen_area_ne = lumen_area[nonempty]
    plaque_area_ne = plaque_area[nonempty]
    vessel_area_ne = vessel_area[nonempty]

    plaque_burden = safe_divide(plaque_area_ne, vessel_area_ne)
    lumen_ratio = safe_divide(lumen_area_ne, vessel_area_ne)
    plaque_to_lumen = safe_divide(plaque_area_ne, lumen_area_ne)

    out["lumen_pixel_count"] = float(lumen_area_ne.sum())
    out["plaque_pixel_count"] = float(plaque_area_ne.sum())
    out["vessel_pixel_count"] = float(vessel_area_ne.sum())

    out["global_lumen_ratio"] = float(
        safe_divide(out["lumen_pixel_count"], out["vessel_pixel_count"])
    )
    out["global_plaque_burden"] = float(
        safe_divide(out["plaque_pixel_count"], out["vessel_pixel_count"])
    )
    out["global_plaque_to_lumen_ratio"] = float(
        safe_divide(out["plaque_pixel_count"], out["lumen_pixel_count"])
    )

    # Basic distribution features
    add_basic_stats("lumen_area", lumen_area_ne, out)
    add_basic_stats("plaque_area", plaque_area_ne, out)
    add_basic_stats("vessel_area", vessel_area_ne, out)
    add_basic_stats("plaque_burden", plaque_burden, out)
    add_basic_stats("lumen_ratio", lumen_ratio, out)
    add_basic_stats("plaque_to_lumen", plaque_to_lumen, out)

    # Important ratio features from Step 1
    out["max_plaque_to_min_lumen_ratio"] = float(
        safe_divide(out["plaque_area_max"], out["lumen_area_min"])
    )
    out["mean_plaque_to_mean_lumen_ratio"] = float(
        safe_divide(out["plaque_area_mean"], out["lumen_area_mean"])
    )
    out["min_to_mean_lumen_ratio"] = float(
        safe_divide(out["lumen_area_min"], out["lumen_area_mean"])
    )
    out["min_to_max_lumen_ratio"] = float(
        safe_divide(out["lumen_area_min"], out["lumen_area_max"])
    )
    out["mean_to_max_lumen_ratio"] = float(
        safe_divide(out["lumen_area_mean"], out["lumen_area_max"])
    )
    out["lumen_area_range"] = float(out["lumen_area_max"] - out["lumen_area_min"])
    out["plaque_area_range"] = float(out["plaque_area_max"] - out["plaque_area_min"])
    out["lumen_range_to_mean_ratio"] = float(
        safe_divide(out["lumen_area_range"], out["lumen_area_mean"])
    )
    out["plaque_range_to_mean_ratio"] = float(
        safe_divide(out["plaque_area_range"], out["plaque_area_mean"])
    )

    # Plaque burden threshold features
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        high = plaque_burden >= thr
        out[f"burden_ge_{int(thr * 100)}_frame_count"] = int(high.sum())
        out[f"burden_ge_{int(thr * 100)}_frame_ratio"] = float(
            safe_divide(high.sum(), plaque_burden.size)
        )
        out[f"burden_ge_{int(thr * 100)}_longest_run"] = longest_true_run(high)
        out[f"burden_ge_{int(thr * 100)}_longest_run_ratio"] = float(
            safe_divide(longest_true_run(high), plaque_burden.size)
        )

    # Narrow lumen features based on relative thresholds
    lumen_median = out["lumen_area_median"]
    lumen_mean = out["lumen_area_mean"]
    lumen_max = out["lumen_area_max"]

    for base_name, base_value in [
        ("median", lumen_median),
        ("mean", lumen_mean),
        ("max", lumen_max),
    ]:
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            narrow = lumen_area_ne <= ratio * base_value
            key = f"lumen_le_{int(ratio * 100)}pct_{base_name}"
            out[f"{key}_frame_count"] = int(narrow.sum())
            out[f"{key}_frame_ratio"] = float(safe_divide(narrow.sum(), lumen_area_ne.size))
            out[f"{key}_longest_run"] = longest_true_run(narrow)
            out[f"{key}_longest_run_ratio"] = float(
                safe_divide(longest_true_run(narrow), lumen_area_ne.size)
            )

    # Lowest lumen percentile segment features
    for pct in [5, 10, 20]:
        threshold = np.percentile(lumen_area_ne, pct)
        narrow = lumen_area_ne <= threshold
        out[f"lumen_bottom_{pct}pct_threshold"] = float(threshold)
        out[f"lumen_bottom_{pct}pct_frame_ratio"] = float(
            safe_divide(narrow.sum(), lumen_area_ne.size)
        )
        out[f"lumen_bottom_{pct}pct_longest_run"] = longest_true_run(narrow)
        out[f"lumen_bottom_{pct}pct_longest_run_ratio"] = float(
            safe_divide(longest_true_run(narrow), lumen_area_ne.size)
        )

    # Location of minimum lumen
    min_lumen_idx = int(np.argmin(lumen_area_ne))
    out["min_lumen_idx_nonempty"] = min_lumen_idx
    out["min_lumen_relative_position"] = float(safe_divide(min_lumen_idx, max(1, lumen_area_ne.size - 1)))
    out["plaque_burden_at_min_lumen"] = float(plaque_burden[min_lumen_idx])
    out["plaque_area_at_min_lumen"] = float(plaque_area_ne[min_lumen_idx])
    out["vessel_area_at_min_lumen"] = float(vessel_area_ne[min_lumen_idx])

    # Proximal / mid / distal features
    n = lumen_area_ne.size
    regions = {
        "proximal": segment_indices(n, 0.0, 1.0 / 3.0),
        "mid": segment_indices(n, 1.0 / 3.0, 2.0 / 3.0),
        "distal": segment_indices(n, 2.0 / 3.0, 1.0),
    }

    for name, sl in regions.items():
        add_region_stats(
            name,
            lumen_area_ne[sl],
            plaque_area_ne[sl],
            vessel_area_ne[sl],
            plaque_burden[sl],
            out,
        )

    # Region ratio features
    out["mid_min_lumen_to_proximal_mean_lumen_ratio"] = float(
        safe_divide(out["mid_lumen_min"], out["proximal_lumen_mean"])
    )
    out["distal_mean_lumen_to_proximal_mean_lumen_ratio"] = float(
        safe_divide(out["distal_lumen_mean"], out["proximal_lumen_mean"])
    )
    out["mid_burden_mean_to_proximal_burden_mean_ratio"] = float(
        safe_divide(out["mid_burden_mean"], out["proximal_burden_mean"])
    )
    out["distal_burden_mean_to_proximal_burden_mean_ratio"] = float(
        safe_divide(out["distal_burden_mean"], out["proximal_burden_mean"])
    )
    out["max_region_burden_mean"] = float(
        np.nanmax([
            out["proximal_burden_mean"],
            out["mid_burden_mean"],
            out["distal_burden_mean"],
        ])
    )
    out["min_region_lumen_mean"] = float(
        np.nanmin([
            out["proximal_lumen_mean"],
            out["mid_lumen_mean"],
            out["distal_lumen_mean"],
        ])
    )

    # Local window features around minimum lumen
    # Ratios are based on nonempty frame count.
    for win_ratio in [0.01, 0.025, 0.05, 0.10]:
        half = max(1, int(round(n * win_ratio)))
        start = max(0, min_lumen_idx - half)
        end = min(n, min_lumen_idx + half + 1)

        local_lumen = lumen_area_ne[start:end]
        local_plaque = plaque_area_ne[start:end]
        local_vessel = vessel_area_ne[start:end]
        local_burden = plaque_burden[start:end]

        name = f"local_min_lumen_win_{str(win_ratio).replace('.', 'p')}"
        out[f"{name}_frame_count"] = int(end - start)
        add_region_stats(name, local_lumen, local_plaque, local_vessel, local_burden, out)

        out[f"{name}_mean_lumen_to_global_mean_ratio"] = float(
            safe_divide(out[f"{name}_lumen_mean"], out["lumen_area_mean"])
        )
        out[f"{name}_mean_burden_to_global_mean_ratio"] = float(
            safe_divide(out[f"{name}_burden_mean"], out["plaque_burden_mean"])
        )

    # Simple derivative / variation features along vessel direction
    if n >= 2:
        d_lumen = np.diff(lumen_area_ne)
        d_burden = np.diff(plaque_burden)
        add_basic_stats("diff_lumen_area", d_lumen, out)
        add_basic_stats("abs_diff_lumen_area", np.abs(d_lumen), out)
        add_basic_stats("diff_plaque_burden", d_burden, out)
        add_basic_stats("abs_diff_plaque_burden", np.abs(d_burden), out)

        out["max_lumen_drop"] = float(np.max(-d_lumen))
        out["max_lumen_rise"] = float(np.max(d_lumen))
        out["max_burden_increase"] = float(np.max(d_burden))
        out["max_burden_decrease"] = float(np.max(-d_burden))
    else:
        for prefix in [
            "diff_lumen_area",
            "abs_diff_lumen_area",
            "diff_plaque_burden",
            "abs_diff_plaque_burden",
        ]:
            add_basic_stats(prefix, np.array([], dtype=np.float64), out)
        out["max_lumen_drop"] = np.nan
        out["max_lumen_rise"] = np.nan
        out["max_burden_increase"] = np.nan
        out["max_burden_decrease"] = np.nan

    return out


def extract_features_from_file(path: Path) -> dict:
    serial_no = path.stem

    try:
        arr = read_mha(path)
        return extract_features_from_array(arr, serial_no=serial_no)
    except Exception:
        return {
            "serial_no": serial_no,
            "read_ok": False,
            "error": traceback.format_exc().replace("\n", " | "),
        }


def extract_split(split_dir: Path, out_csv: Path) -> pd.DataFrame:
    mha_paths = sorted(split_dir.glob("*.mha"))

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] found {len(mha_paths)} .mha files")

    rows = []
    for i, path in enumerate(mha_paths, start=1):
        if i == 1 or i % 25 == 0 or i == len(mha_paths):
            print(f"[INFO] {i}/{len(mha_paths)} {path.name}")

        rows.append(extract_features_from_file(path))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")
    return df


def merge_labels(train_features: pd.DataFrame, labels_path: Path, out_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    labels["serial_no"] = labels["serial_no"].astype(str)

    if "ffr_class" not in labels.columns:
        labels["ffr_class"] = (labels["FFR"] >= 0.8).astype(int)

    train_features = train_features.copy()
    train_features["serial_no"] = train_features["serial_no"].astype(str)

    merged = train_features.merge(labels[["serial_no", "FFR", "ffr_class"]], on="serial_no", how="left")

    missing = merged["FFR"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} train feature rows do not have labels.")

    merged.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")
    return merged


def write_summary(
    train_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    out_path: Path,
):
    lines = []
    lines.append("Step 2 Feature Extraction Summary")
    lines.append("=" * 50)
    lines.append("")

    if train_df is not None:
        lines.append("[Train]")
        lines.append(f"num_rows: {len(train_df)}")
        lines.append(f"read_ok: {int(train_df['read_ok'].sum())}/{len(train_df)}")
        if "FFR" in train_df.columns:
            lines.append(f"FFR_missing: {int(train_df['FFR'].isna().sum())}")
        if "ffr_class" in train_df.columns:
            counts = train_df["ffr_class"].value_counts().sort_index()
            lines.append(f"class_0_FFR_lt_0.8: {int(counts.get(0, 0))}")
            lines.append(f"class_1_FFR_ge_0.8: {int(counts.get(1, 0))}")
        lines.append(f"num_columns: {train_df.shape[1]}")
        lines.append("")

    if test_df is not None:
        lines.append("[Test public]")
        lines.append(f"num_rows: {len(test_df)}")
        lines.append(f"read_ok: {int(test_df['read_ok'].sum())}/{len(test_df)}")
        lines.append(f"num_columns: {test_df.shape[1]}")
        lines.append("")

    if train_df is not None:
        numeric_cols = [
            c for c in train_df.columns
            if pd.api.types.is_numeric_dtype(train_df[c])
        ]
        lines.append("[Feature table]")
        lines.append(f"numeric_columns: {len(numeric_cols)}")
        lines.append("example_numeric_columns:")
        for c in numeric_cols[:30]:
            lines.append(f"  - {c}")
        lines.append("")

        nan_ratio = train_df[numeric_cols].isna().mean().sort_values(ascending=False)
        high_nan = nan_ratio[nan_ratio > 0]
        lines.append("[NaN check]")
        if len(high_nan) == 0:
            lines.append("No NaN values in numeric feature columns.")
        else:
            lines.append("Columns with NaN:")
            for col, ratio in high_nan.head(30).items():
                lines.append(f"  - {col}: {ratio:.4%}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=".", help="Root directory containing train/ and test_public/")
    parser.add_argument("--train-dir", type=str, default="train")
    parser.add_argument("--test-dir", type=str, default="test_public")
    parser.add_argument("--labels", type=str, default=None, help="Default: train/labels.csv")
    parser.add_argument("--out-dir", type=str, default="step2_outputs")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not train_dir.is_absolute():
        train_dir = data_root / train_dir
    if not test_dir.is_absolute():
        test_dir = data_root / test_dir

    labels_path = Path(args.labels) if args.labels is not None else train_dir / "labels.csv"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = None
    test_df = None

    if not args.skip_train:
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        train_features_path = out_dir / "train_features_raw.csv"
        train_df = extract_split(train_dir, train_features_path)

        if labels_path.exists():
            train_df = merge_labels(
                train_df,
                labels_path,
                out_dir / "train_features.csv",
            )
        else:
            print(f"[WARN] labels.csv not found: {labels_path}")
            train_df.to_csv(out_dir / "train_features.csv", index=False)

    if not args.skip_test:
        if test_dir.exists():
            test_df = extract_split(test_dir, out_dir / "test_public_features.csv")
        else:
            print(f"[WARN] Test directory not found: {test_dir}")

    write_summary(
        train_df=train_df,
        test_df=test_df,
        out_path=out_dir / "feature_extraction_summary.txt",
    )

    print("")
    print("[DONE] Step 2 feature extraction completed.")


if __name__ == "__main__":
    main()