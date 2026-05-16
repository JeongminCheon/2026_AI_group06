#!/usr/bin/env python3
# step0_data_check.py

import argparse
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import SimpleITK as sitk


def read_mha(path: Path) -> np.ndarray:
    """
    Read .mha mask volume.
    Expected output shape: (num_frames, 256, 256)
    Expected values: 0 background, 1 lumen, 2 plaque
    """
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return arr


def check_one_mha(path: Path) -> dict:
    serial_no = path.stem

    row = {
        "serial_no": serial_no,
        "file_path": str(path),
        "exists": path.exists(),
        "read_ok": False,
        "error": "",
        "num_frames": np.nan,
        "height": np.nan,
        "width": np.nan,
        "dtype": "",
        "unique_values": "",
        "only_0_1_2": False,
        "has_lumen": False,
        "has_plaque": False,
        "empty_frame_count": np.nan,
        "nonempty_frame_count": np.nan,
        "lumen_pixel_count": np.nan,
        "plaque_pixel_count": np.nan,
        "background_pixel_count": np.nan,
        "lumen_frame_count": np.nan,
        "plaque_frame_count": np.nan,
        "min_lumen_area": np.nan,
        "mean_lumen_area": np.nan,
        "max_lumen_area": np.nan,
        "min_plaque_area": np.nan,
        "mean_plaque_area": np.nan,
        "max_plaque_area": np.nan,
    }

    try:
        arr = read_mha(path)
        row["read_ok"] = True

        row["dtype"] = str(arr.dtype)

        if arr.ndim == 3:
            num_frames, height, width = arr.shape
            row["num_frames"] = int(num_frames)
            row["height"] = int(height)
            row["width"] = int(width)
        else:
            row["error"] = f"Unexpected ndim: {arr.ndim}"
            return row

        unique_values = np.unique(arr)
        row["unique_values"] = ",".join(map(str, unique_values.tolist()))
        row["only_0_1_2"] = bool(set(unique_values.tolist()).issubset({0, 1, 2}))

        background = arr == 0
        lumen = arr == 1
        plaque = arr == 2
        vessel = lumen | plaque

        row["has_lumen"] = bool(lumen.any())
        row["has_plaque"] = bool(plaque.any())

        vessel_area = vessel.sum(axis=(1, 2))
        lumen_area = lumen.sum(axis=(1, 2))
        plaque_area = plaque.sum(axis=(1, 2))

        row["empty_frame_count"] = int((vessel_area == 0).sum())
        row["nonempty_frame_count"] = int((vessel_area > 0).sum())

        row["background_pixel_count"] = int(background.sum())
        row["lumen_pixel_count"] = int(lumen.sum())
        row["plaque_pixel_count"] = int(plaque.sum())

        row["lumen_frame_count"] = int((lumen_area > 0).sum())
        row["plaque_frame_count"] = int((plaque_area > 0).sum())

        if row["lumen_frame_count"] > 0:
            valid_lumen_area = lumen_area[lumen_area > 0]
            row["min_lumen_area"] = float(valid_lumen_area.min())
            row["mean_lumen_area"] = float(valid_lumen_area.mean())
            row["max_lumen_area"] = float(valid_lumen_area.max())

        if row["plaque_frame_count"] > 0:
            valid_plaque_area = plaque_area[plaque_area > 0]
            row["min_plaque_area"] = float(valid_plaque_area.min())
            row["mean_plaque_area"] = float(valid_plaque_area.mean())
            row["max_plaque_area"] = float(valid_plaque_area.max())

    except Exception:
        row["error"] = traceback.format_exc().replace("\n", " | ")

    return row


def check_split(split_dir: Path) -> pd.DataFrame:
    mha_paths = sorted(split_dir.glob("*.mha"))
    rows = []

    print(f"[INFO] Checking {split_dir} ...")
    print(f"[INFO] Found {len(mha_paths)} .mha files")

    for i, path in enumerate(mha_paths, start=1):
        if i % 50 == 0 or i == len(mha_paths):
            print(f"[INFO] {i}/{len(mha_paths)}: {path.name}")
        rows.append(check_one_mha(path))

    return pd.DataFrame(rows)


def make_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    labels_df: pd.DataFrame | None,
) -> str:
    lines = []

    lines.append("Step 0 Data Check Summary")
    lines.append("=" * 40)
    lines.append("")

    lines.append("[Train files]")
    lines.append(f"num_train_mha: {len(train_df)}")
    if len(train_df) > 0:
        lines.append(f"read_ok: {int(train_df['read_ok'].sum())}/{len(train_df)}")
        lines.append(f"bad_read: {int((~train_df['read_ok']).sum())}")
        lines.append(f"only_0_1_2: {int(train_df['only_0_1_2'].sum())}/{len(train_df)}")
        lines.append(
            "shape_256x256: "
            f"{int(((train_df['height'] == 256) & (train_df['width'] == 256)).sum())}/{len(train_df)}"
        )
        lines.append(
            "num_frames stats: "
            f"min={train_df['num_frames'].min()}, "
            f"median={train_df['num_frames'].median()}, "
            f"max={train_df['num_frames'].max()}"
        )
        lines.append(f"no_lumen_count: {int((~train_df['has_lumen']).sum())}")
        lines.append(f"no_plaque_count: {int((~train_df['has_plaque']).sum())}")
    lines.append("")

    if labels_df is not None:
        lines.append("[Labels]")
        lines.append(f"num_labels: {len(labels_df)}")
        lines.append(f"label_columns: {labels_df.columns.tolist()}")
        if "FFR" in labels_df.columns:
            lines.append(f"FFR_missing: {int(labels_df['FFR'].isna().sum())}")
            lines.append(f"FFR_min: {labels_df['FFR'].min():.4f}")
            lines.append(f"FFR_mean: {labels_df['FFR'].mean():.4f}")
            lines.append(f"FFR_median: {labels_df['FFR'].median():.4f}")
            lines.append(f"FFR_max: {labels_df['FFR'].max():.4f}")

            ffr_class = (labels_df["FFR"] >= 0.8).astype(int)
            class_counts = ffr_class.value_counts().sort_index()
            lines.append(f"class_0_FFR_lt_0.8: {int(class_counts.get(0, 0))}")
            lines.append(f"class_1_FFR_ge_0.8: {int(class_counts.get(1, 0))}")

        if "serial_no" in labels_df.columns and len(train_df) > 0:
            train_serials = set(train_df["serial_no"].astype(str))
            label_serials = set(labels_df["serial_no"].astype(str))

            missing_label_for_file = sorted(train_serials - label_serials)
            missing_file_for_label = sorted(label_serials - train_serials)

            lines.append(f"train_files_without_label: {len(missing_label_for_file)}")
            lines.append(f"labels_without_train_file: {len(missing_file_for_label)}")

            if missing_label_for_file:
                lines.append("examples_train_files_without_label: " + ", ".join(missing_label_for_file[:10]))
            if missing_file_for_label:
                lines.append("examples_labels_without_train_file: " + ", ".join(missing_file_for_label[:10]))
        lines.append("")

    if test_df is not None:
        lines.append("[Test public files]")
        lines.append(f"num_test_mha: {len(test_df)}")
        if len(test_df) > 0:
            lines.append(f"read_ok: {int(test_df['read_ok'].sum())}/{len(test_df)}")
            lines.append(f"bad_read: {int((~test_df['read_ok']).sum())}")
            lines.append(f"only_0_1_2: {int(test_df['only_0_1_2'].sum())}/{len(test_df)}")
            lines.append(
                "shape_256x256: "
                f"{int(((test_df['height'] == 256) & (test_df['width'] == 256)).sum())}/{len(test_df)}"
            )
            lines.append(
                "num_frames stats: "
                f"min={test_df['num_frames'].min()}, "
                f"median={test_df['num_frames'].median()}, "
                f"max={test_df['num_frames'].max()}"
            )
            lines.append(f"no_lumen_count: {int((~test_df['has_lumen']).sum())}")
            lines.append(f"no_plaque_count: {int((~test_df['has_plaque']).sum())}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=".", help="Root directory containing train/ and test_public/")
    parser.add_argument("--train-dir", type=str, default="train", help="Train directory name or path")
    parser.add_argument("--test-dir", type=str, default="test_public", help="Test directory name or path")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels.csv. Default: train/labels.csv")
    parser.add_argument("--out-dir", type=str, default="step0_outputs", help="Output directory")
    parser.add_argument("--skip-test", action="store_true", help="Skip test_public check")
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

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    train_df = check_split(train_dir)
    train_out = out_dir / "data_check_train.csv"
    train_df.to_csv(train_out, index=False)
    print(f"[SAVE] {train_out}")

    labels_df = None
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        labels_out = out_dir / "labels_with_class.csv"
        labels_df = labels_df.copy()
        labels_df["ffr_class"] = (labels_df["FFR"] >= 0.8).astype(int)
        labels_df.to_csv(labels_out, index=False)
        print(f"[SAVE] {labels_out}")
    else:
        print(f"[WARN] labels.csv not found: {labels_path}")

    test_df = None
    if not args.skip_test:
        if test_dir.exists():
            test_df = check_split(test_dir)
            test_out = out_dir / "data_check_test_public.csv"
            test_df.to_csv(test_out, index=False)
            print(f"[SAVE] {test_out}")
        else:
            print(f"[WARN] Test directory not found: {test_dir}")

    summary = make_summary(train_df, test_df, labels_df)
    summary_out = out_dir / "data_check_summary.txt"
    summary_out.write_text(summary, encoding="utf-8")
    print(f"[SAVE] {summary_out}")

    print("")
    print(summary)


if __name__ == "__main__":
    main()