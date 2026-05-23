#!/usr/bin/env python3
# step7_0_extract_feature_v2.py
#
# Step 7-0 / Step 7-1:
#   Extract additional morphology features from IVUS segmentation masks.
#
# Recommended:
# python step7_0_extract_feature_v2.py \
#   --data-root ./26S_AI536_NE450 \
#   --base-train-features step2_outputs/train_features.csv \
#   --base-test-features step2_outputs/test_public_features.csv \
#   --out-dir step7_0_feature_v2_outputs \
#   --num-workers 8

import argparse
import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

EPS = 1e-8


def safe_divide(a, b, eps=EPS):
    return a / (b + eps)


def finite_array(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def add_stats(out, prefix, x):
    x = finite_array(x)
    if x.size == 0:
        vals = {k: 0.0 for k in [
            "mean", "std", "min", "max", "median", "p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99"
        ]}
    else:
        vals = {
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
    for k, v in vals.items():
        out[f"{prefix}_{k}"] = v


def longest_true_run(mask):
    best = 0
    cur = 0
    for v in np.asarray(mask, dtype=bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def top_fraction_mean(x, frac):
    x = finite_array(x)
    if x.size == 0:
        return 0.0
    k = max(1, int(math.ceil(x.size * frac)))
    return float(np.mean(np.sort(x)[-k:]))


def bbox_stats_from_mask(mask):
    ys, xs = np.nonzero(mask)
    area = int(len(xs))
    if area == 0:
        return {"area": 0.0, "bbox_w": 0.0, "bbox_h": 0.0, "bbox_area": 0.0, "aspect": 0.0, "compactness": 0.0, "cy": 0.0, "cx": 0.0}

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = float(y1 - y0 + 1)
    w = float(x1 - x0 + 1)
    bbox_area = float(h * w)
    aspect = float(min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0
    compactness = float(area / (bbox_area + EPS))
    return {
        "area": float(area),
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": bbox_area,
        "aspect": aspect,
        "compactness": compactness,
        "cy": float(ys.mean()),
        "cx": float(xs.mean()),
    }


def read_mha(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.uint8, copy=False)


def segment_slices(n, segments=5):
    edges = np.linspace(0, n, segments + 1).round().astype(int)
    out = []
    for i in range(segments):
        s = int(edges[i])
        e = int(edges[i + 1])
        if e <= s:
            e = min(n, s + 1)
        out.append(slice(s, e))
    return out


def add_reference_lumen_features(out, lumen_area):
    if len(lumen_area) == 0:
        lumen_area = np.asarray([0.0], dtype=np.float64)

    min_lumen = float(np.min(lumen_area))
    mean_lumen = float(np.mean(lumen_area))
    p05_lumen = float(np.percentile(lumen_area, 5))
    p10_lumen = float(np.percentile(lumen_area, 10))

    for pct, frac in [(10, 0.10), (20, 0.20), (30, 0.30)]:
        ref = top_fraction_mean(lumen_area, frac)
        out[f"v2_ref_lumen_top{pct}_mean"] = ref
        out[f"v2_min_lumen_to_ref_top{pct}_ratio"] = float(safe_divide(min_lumen, ref))
        out[f"v2_p05_lumen_to_ref_top{pct}_ratio"] = float(safe_divide(p05_lumen, ref))
        out[f"v2_p10_lumen_to_ref_top{pct}_ratio"] = float(safe_divide(p10_lumen, ref))
        out[f"v2_mean_lumen_to_ref_top{pct}_ratio"] = float(safe_divide(mean_lumen, ref))
        out[f"v2_stenosis_severity_top{pct}"] = float(1.0 - safe_divide(min_lumen, ref))

    n = len(lumen_area)
    nseg = max(1, n // 5)
    prox_ref = top_fraction_mean(lumen_area[:nseg], 0.50)
    dist_ref = top_fraction_mean(lumen_area[-nseg:], 0.50)
    pd_ref = max(prox_ref, dist_ref)

    out["v2_ref_lumen_prox_top50_mean"] = prox_ref
    out["v2_ref_lumen_dist_top50_mean"] = dist_ref
    out["v2_ref_lumen_prox_dist_max"] = pd_ref
    out["v2_min_lumen_to_prox_ref_ratio"] = float(safe_divide(min_lumen, prox_ref))
    out["v2_min_lumen_to_dist_ref_ratio"] = float(safe_divide(min_lumen, dist_ref))
    out["v2_min_lumen_to_prox_dist_max_ref_ratio"] = float(safe_divide(min_lumen, pd_ref))
    out["v2_stenosis_severity_prox_dist_max"] = float(1.0 - safe_divide(min_lumen, pd_ref))


def add_stenosis_length_features(out, lumen_area):
    if len(lumen_area) == 0:
        lumen_area = np.asarray([0.0], dtype=np.float64)

    ref = max(top_fraction_mean(lumen_area, 0.20), top_fraction_mean(lumen_area, 0.30), EPS)
    stenosis = 1.0 - safe_divide(lumen_area, ref)
    stenosis = np.clip(stenosis, -2.0, 1.0)
    positive = np.clip(stenosis, 0.0, None)

    add_stats(out, "v2_stenosis_ratio_ref", stenosis)
    add_stats(out, "v2_positive_stenosis_ratio_ref", positive)

    out["v2_stenosis_integral"] = float(np.sum(positive))
    out["v2_stenosis_integral_norm"] = float(np.mean(positive))
    out["v2_stenosis_energy"] = float(np.sum(positive ** 2))
    out["v2_stenosis_energy_norm"] = float(np.mean(positive ** 2))

    for thr in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        m = stenosis >= thr
        key = int(round(thr * 100))
        run = longest_true_run(m)
        excess = np.clip(stenosis - thr, 0.0, None)
        out[f"v2_stenosis_ge_{key}_count"] = int(m.sum())
        out[f"v2_stenosis_ge_{key}_ratio"] = float(safe_divide(m.sum(), len(stenosis)))
        out[f"v2_stenosis_ge_{key}_longest_run"] = int(run)
        out[f"v2_stenosis_ge_{key}_longest_run_ratio"] = float(safe_divide(run, len(stenosis)))
        out[f"v2_stenosis_excess_{key}_sum"] = float(np.sum(excess))
        out[f"v2_stenosis_excess_{key}_mean"] = float(np.mean(excess))

    positions = np.linspace(0.0, 1.0, len(stenosis))
    out["v2_stenosis_center_of_mass"] = float(safe_divide(np.sum(positions * positive), np.sum(positive)))
    out["v2_stenosis_position_at_max"] = float(positions[int(np.argmax(stenosis))])


def add_diameter_features(out, lumen_area, vessel_area):
    lumen_d = np.sqrt(np.clip(4.0 * lumen_area / np.pi, 0.0, None))
    vessel_d = np.sqrt(np.clip(4.0 * vessel_area / np.pi, 0.0, None))
    plaque_thick = np.clip(vessel_d - lumen_d, 0.0, None)

    add_stats(out, "v2_equiv_lumen_diameter", lumen_d)
    add_stats(out, "v2_equiv_vessel_diameter", vessel_d)
    add_stats(out, "v2_equiv_plaque_thickness_proxy", plaque_thick)

    ref_d = top_fraction_mean(lumen_d, 0.20)
    min_d = float(np.min(lumen_d)) if len(lumen_d) else 0.0
    p05_d = float(np.percentile(lumen_d, 5)) if len(lumen_d) else 0.0
    out["v2_ref_lumen_diameter_top20_mean"] = float(ref_d)
    out["v2_min_lumen_diameter_to_ref_ratio"] = float(safe_divide(min_d, ref_d))
    out["v2_p05_lumen_diameter_to_ref_ratio"] = float(safe_divide(p05_d, ref_d))
    out["v2_diameter_stenosis_severity"] = float(1.0 - safe_divide(min_d, ref_d))
    out["v2_max_plaque_thick_to_ref_diameter_ratio"] = float(safe_divide(np.max(plaque_thick), ref_d))


def add_shape_features(out, arr, nonempty_idx):
    lumen_aspect, lumen_compact, lumen_bbox_ratio = [], [], []
    plaque_aspect, plaque_compact, plaque_bbox_ratio = [], [], []
    centroid_dist, centroid_dist_norm, dx_norm, dy_norm = [], [], [], []

    for z in nonempty_idx:
        sl = arr[int(z)]
        lum = sl == 1
        pla = sl == 2
        ves = lum | pla
        ls = bbox_stats_from_mask(lum)
        ps = bbox_stats_from_mask(pla)
        vs = bbox_stats_from_mask(ves)

        if ls["area"] > 0:
            lumen_aspect.append(ls["aspect"])
            lumen_compact.append(ls["compactness"])
            lumen_bbox_ratio.append(safe_divide(ls["bbox_area"], vs["bbox_area"]))
        if ps["area"] > 0:
            plaque_aspect.append(ps["aspect"])
            plaque_compact.append(ps["compactness"])
            plaque_bbox_ratio.append(safe_divide(ps["bbox_area"], vs["bbox_area"]))
        if ls["area"] > 0 and ps["area"] > 0:
            dx = ps["cx"] - ls["cx"]
            dy = ps["cy"] - ls["cy"]
            d = math.sqrt(dx * dx + dy * dy)
            norm = math.sqrt(max(vs["area"], EPS))
            centroid_dist.append(d)
            centroid_dist_norm.append(d / (norm + EPS))
            dx_norm.append(dx / (norm + EPS))
            dy_norm.append(dy / (norm + EPS))

    add_stats(out, "v2_lumen_bbox_aspect", lumen_aspect)
    add_stats(out, "v2_lumen_compactness", lumen_compact)
    add_stats(out, "v2_lumen_bbox_area_to_vessel_bbox_area", lumen_bbox_ratio)
    add_stats(out, "v2_plaque_bbox_aspect", plaque_aspect)
    add_stats(out, "v2_plaque_compactness", plaque_compact)
    add_stats(out, "v2_plaque_bbox_area_to_vessel_bbox_area", plaque_bbox_ratio)
    add_stats(out, "v2_plaque_lumen_centroid_distance", centroid_dist)
    add_stats(out, "v2_plaque_lumen_centroid_distance_norm", centroid_dist_norm)
    add_stats(out, "v2_plaque_lumen_centroid_dx_norm", dx_norm)
    add_stats(out, "v2_plaque_lumen_centroid_dy_norm", dy_norm)


def add_segment5_features(out, lumen_area, plaque_area, vessel_area):
    if len(lumen_area) == 0:
        return
    burden = safe_divide(plaque_area, vessel_area)
    ref = max(top_fraction_mean(lumen_area, 0.20), EPS)
    stenosis = np.clip(1.0 - safe_divide(lumen_area, ref), -2.0, 1.0)
    positive = np.clip(stenosis, 0.0, None)

    min_lumen_values, burden_mean_values, stenosis_max_values = [], [], []
    for i, sl in enumerate(segment_slices(len(lumen_area), 5)):
        la, pa, va = lumen_area[sl], plaque_area[sl], vessel_area[sl]
        bu, st, pos = burden[sl], stenosis[sl], positive[sl]
        prefix = f"v2_seg5_{i}"
        add_stats(out, f"{prefix}_lumen", la)
        add_stats(out, f"{prefix}_burden", bu)
        add_stats(out, f"{prefix}_stenosis", st)
        out[f"{prefix}_lumen_sum"] = float(np.sum(la))
        out[f"{prefix}_plaque_sum"] = float(np.sum(pa))
        out[f"{prefix}_vessel_sum"] = float(np.sum(va))
        out[f"{prefix}_global_burden"] = float(safe_divide(np.sum(pa), np.sum(va)))
        out[f"{prefix}_stenosis_integral"] = float(np.sum(pos))
        out[f"{prefix}_stenosis_integral_norm"] = float(np.mean(pos)) if len(pos) else 0.0
        min_lumen_values.append(float(np.min(la)) if len(la) else 0.0)
        burden_mean_values.append(float(np.mean(bu)) if len(bu) else 0.0)
        stenosis_max_values.append(float(np.max(st)) if len(st) else 0.0)

    out["v2_seg5_min_lumen_segment_id"] = int(np.argmin(min_lumen_values))
    out["v2_seg5_max_burden_segment_id"] = int(np.argmax(burden_mean_values))
    out["v2_seg5_max_stenosis_segment_id"] = int(np.argmax(stenosis_max_values))


def extract_v2_features_one(path):
    path = Path(path)
    out = {"serial_no": path.stem, "v2_read_ok": True, "v2_error": ""}
    try:
        arr = read_mha(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got {arr.shape}")

        lumen = arr == 1
        plaque = arr == 2
        vessel = lumen | plaque
        lumen_all = lumen.sum(axis=(1, 2)).astype(np.float64)
        plaque_all = plaque.sum(axis=(1, 2)).astype(np.float64)
        vessel_all = vessel.sum(axis=(1, 2)).astype(np.float64)

        nonempty = vessel_all > 0
        nonempty_idx = np.where(nonempty)[0]
        if len(nonempty_idx) == 0:
            nonempty_idx = np.arange(arr.shape[0])
            nonempty = np.ones(arr.shape[0], dtype=bool)

        lumen_area = lumen_all[nonempty]
        plaque_area = plaque_all[nonempty]
        vessel_area = vessel_all[nonempty]

        out["v2_num_nonempty_frames"] = int(len(lumen_area))
        out["v2_num_total_frames"] = int(arr.shape[0])
        out["v2_nonempty_ratio"] = float(safe_divide(len(lumen_area), arr.shape[0]))

        add_reference_lumen_features(out, lumen_area)
        add_stenosis_length_features(out, lumen_area)
        add_diameter_features(out, lumen_area, vessel_area)
        add_shape_features(out, arr, nonempty_idx)
        add_segment5_features(out, lumen_area, plaque_area, vessel_area)

        for k, v in list(out.items()):
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                out[k] = 0.0
    except Exception as e:
        out["v2_read_ok"] = False
        out["v2_error"] = repr(e)
        out["v2_traceback"] = traceback.format_exc()
    return out


def extract_many(paths, num_workers):
    paths = [str(p) for p in paths]
    rows = []
    if num_workers <= 1:
        for i, p in enumerate(paths, 1):
            rows.append(extract_v2_features_one(p))
            if i % 50 == 0:
                print(f"[INFO] extracted {i}/{len(paths)}")
        return rows
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(extract_v2_features_one, p): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 50 == 0:
                print(f"[INFO] extracted {i}/{len(paths)}")
    return rows


def merge_base_features(base_path, v2_df, out_path):
    base = pd.read_csv(base_path)
    base["serial_no"] = base["serial_no"].astype(str)
    v2_df = v2_df.copy()
    v2_df["serial_no"] = v2_df["serial_no"].astype(str)
    drop_cols = [c for c in v2_df.columns if c in base.columns and c != "serial_no"]
    if drop_cols:
        v2_df = v2_df.drop(columns=drop_cols)
    merged = base.merge(v2_df, on="serial_no", how="left")
    merged = merged.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    merged.to_csv(out_path, index=False)
    return merged


def make_eda(train_df, out_dir):
    if "FFR" not in train_df.columns:
        return None
    df = train_df.copy()
    if "ffr_class" not in df.columns:
        df["ffr_class"] = (df["FFR"] >= 0.8).astype(int)
    numeric_cols = [c for c in df.columns if c.startswith("v2_") and pd.api.types.is_numeric_dtype(df[c])]
    y_ffr = df["FFR"].astype(float)
    y_cls = df["ffr_class"].astype(int)
    class0 = df[y_cls == 0]
    class1 = df[y_cls == 1]
    rows = []
    for c in numeric_cols:
        x = df[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pearson = pd.Series(x).corr(pd.Series(y_ffr), method="pearson")
        spearman = pd.Series(x).corr(pd.Series(y_ffr), method="spearman")
        x0 = class0[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x1 = class1[c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rows.append({
            "feature": c,
            "class0_mean": float(x0.mean()),
            "class1_mean": float(x1.mean()),
            "mean_diff_class0_minus_class1": float(x0.mean() - x1.mean()),
            "corr_with_FFR_pearson": float(pearson) if np.isfinite(pearson) else 0.0,
            "corr_with_FFR_spearman": float(spearman) if np.isfinite(spearman) else 0.0,
            "abs_corr_with_FFR_spearman": abs(float(spearman)) if np.isfinite(spearman) else 0.0,
        })
    eda = pd.DataFrame(rows)
    if len(eda):
        eda = eda.sort_values("abs_corr_with_FFR_spearman", ascending=False)
    eda.to_csv(out_dir / "feature_v2_eda.csv", index=False)
    return eda


def write_summary(out_dir, train_v2, test_v2, train_merged=None, test_merged=None, eda=None):
    lines = []
    lines.append("Step 7-0 Feature v2 Extraction Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[V2 only]")
    lines.append(f"train rows: {len(train_v2)}")
    lines.append(f"test rows: {len(test_v2)}")
    lines.append(f"train columns: {len(train_v2.columns)}")
    lines.append(f"test columns: {len(test_v2.columns)}")
    lines.append(f"train read_ok: {int(train_v2['v2_read_ok'].sum())}/{len(train_v2)}")
    lines.append(f"test read_ok: {int(test_v2['v2_read_ok'].sum())}/{len(test_v2)}")
    lines.append("")
    if train_merged is not None:
        lines.append("[Merged]")
        lines.append(f"train merged columns: {len(train_merged.columns)}")
        lines.append(f"test merged columns: {len(test_merged.columns)}")
        numeric_cols = train_merged.select_dtypes(include=[np.number]).columns.tolist()
        nan_count = int(train_merged[numeric_cols].isna().sum().sum()) if numeric_cols else 0
        lines.append(f"train numeric NaN count: {nan_count}")
        lines.append("")
    if eda is not None and len(eda):
        lines.append("[Top v2 features by abs Spearman correlation with FFR]")
        lines.append(eda.head(30).to_string(index=False))
        lines.append("")
    (out_dir / "feature_v2_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./26S_AI536_NE450")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--base-train-features", default=None)
    parser.add_argument("--base-test-features", default=None)
    parser.add_argument("--out-dir", default="step7_0_feature_v2_outputs")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = Path(args.train_dir) if args.train_dir else data_root / "train"
    test_dir = Path(args.test_dir) if args.test_dir else data_root / "test_public"
    labels_path = Path(args.labels) if args.labels else train_dir / "labels.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_path)
    labels_df["serial_no"] = labels_df["serial_no"].astype(str)
    train_paths = [train_dir / f"{s}.mha" for s in labels_df["serial_no"]]
    test_paths = sorted(test_dir.glob("*.mha"))

    print(f"[INFO] train files: {len(train_paths)}")
    print(f"[INFO] test files: {len(test_paths)}")
    print(f"[INFO] num_workers: {args.num_workers}")

    print("[INFO] extracting train v2 features...")
    train_v2 = pd.DataFrame(extract_many(train_paths, args.num_workers))
    train_v2 = labels_df.merge(train_v2, on="serial_no", how="left")
    train_v2 = train_v2.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    train_v2.to_csv(out_dir / "train_features_v2_only.csv", index=False)

    print("[INFO] extracting test v2 features...")
    test_v2 = pd.DataFrame(extract_many(test_paths, args.num_workers))
    test_v2 = test_v2.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    test_v2.to_csv(out_dir / "test_public_features_v2_only.csv", index=False)

    train_merged = None
    test_merged = None
    if args.base_train_features and args.base_test_features:
        print("[INFO] merging with base Step 2 features...")
        train_v2_for_merge = train_v2.drop(columns=[c for c in ["FFR", "ffr_class"] if c in train_v2.columns])
        train_merged = merge_base_features(args.base_train_features, train_v2_for_merge, out_dir / "train_features_v2_merged.csv")
        test_merged = merge_base_features(args.base_test_features, test_v2, out_dir / "test_public_features_v2_merged.csv")

    print("[INFO] making EDA summary...")
    eda_source = train_merged if train_merged is not None else train_v2
    eda = make_eda(eda_source, out_dir)
    write_summary(out_dir, train_v2, test_v2, train_merged, test_merged, eda)

    print("[DONE] Step 7-0 feature v2 extraction complete.")
    print(f"[CHECK] {out_dir / 'feature_v2_summary.txt'}")


if __name__ == "__main__":
    main()
