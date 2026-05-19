from pathlib import Path
import numpy as np
import SimpleITK as sitk

EPS = 1e-8


def read_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def safe_divide(a, b, eps: float = EPS):
    return a / (b + eps)


def longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def add_basic_stats(prefix: str, x: np.ndarray, out: dict):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        stats = {k: np.nan for k in ["mean", "std", "min", "max", "median", "p01", "p05", "p10", "p25", "p75", "p90", "p95", "p99"]}
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
    start = int(round(n * start_ratio))
    end = int(round(n * end_ratio))
    start = max(0, min(n, start))
    end = max(start + 1, min(n, end))
    return slice(start, end)


def add_region_stats(region_name: str, lumen_area: np.ndarray, plaque_area: np.ndarray, vessel_area: np.ndarray, plaque_burden: np.ndarray, out: dict):
    add_basic_stats(f"{region_name}_lumen", lumen_area, out)
    add_basic_stats(f"{region_name}_plaque", plaque_area, out)
    add_basic_stats(f"{region_name}_vessel", vessel_area, out)
    add_basic_stats(f"{region_name}_burden", plaque_burden, out)
    out[f"{region_name}_lumen_sum"] = float(np.sum(lumen_area))
    out[f"{region_name}_plaque_sum"] = float(np.sum(plaque_area))
    out[f"{region_name}_vessel_sum"] = float(np.sum(vessel_area))
    out[f"{region_name}_global_burden"] = float(safe_divide(np.sum(plaque_area), np.sum(vessel_area)))


def extract_features_from_array(arr: np.ndarray, serial_no: str = "") -> dict:
    out = {"serial_no": serial_no, "read_ok": True, "error": ""}
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
    if int(nonempty.sum()) == 0:
        nonempty = np.ones(num_frames, dtype=bool)

    lumen_area_ne = lumen_area[nonempty]
    plaque_area_ne = plaque_area[nonempty]
    vessel_area_ne = vessel_area[nonempty]
    plaque_burden = safe_divide(plaque_area_ne, vessel_area_ne)
    lumen_ratio = safe_divide(lumen_area_ne, vessel_area_ne)
    plaque_to_lumen = safe_divide(plaque_area_ne, lumen_area_ne)

    out["lumen_pixel_count"] = float(lumen_area_ne.sum())
    out["plaque_pixel_count"] = float(plaque_area_ne.sum())
    out["vessel_pixel_count"] = float(vessel_area_ne.sum())
    out["global_lumen_ratio"] = float(safe_divide(out["lumen_pixel_count"], out["vessel_pixel_count"]))
    out["global_plaque_burden"] = float(safe_divide(out["plaque_pixel_count"], out["vessel_pixel_count"]))
    out["global_plaque_to_lumen_ratio"] = float(safe_divide(out["plaque_pixel_count"], out["lumen_pixel_count"]))

    add_basic_stats("lumen_area", lumen_area_ne, out)
    add_basic_stats("plaque_area", plaque_area_ne, out)
    add_basic_stats("vessel_area", vessel_area_ne, out)
    add_basic_stats("plaque_burden", plaque_burden, out)
    add_basic_stats("lumen_ratio", lumen_ratio, out)
    add_basic_stats("plaque_to_lumen", plaque_to_lumen, out)

    out["max_plaque_to_min_lumen_ratio"] = float(safe_divide(out["plaque_area_max"], out["lumen_area_min"]))
    out["mean_plaque_to_mean_lumen_ratio"] = float(safe_divide(out["plaque_area_mean"], out["lumen_area_mean"]))
    out["min_to_mean_lumen_ratio"] = float(safe_divide(out["lumen_area_min"], out["lumen_area_mean"]))
    out["min_to_max_lumen_ratio"] = float(safe_divide(out["lumen_area_min"], out["lumen_area_max"]))
    out["mean_to_max_lumen_ratio"] = float(safe_divide(out["lumen_area_mean"], out["lumen_area_max"]))
    out["lumen_area_range"] = float(out["lumen_area_max"] - out["lumen_area_min"])
    out["plaque_area_range"] = float(out["plaque_area_max"] - out["plaque_area_min"])
    out["lumen_range_to_mean_ratio"] = float(safe_divide(out["lumen_area_range"], out["lumen_area_mean"]))
    out["plaque_range_to_mean_ratio"] = float(safe_divide(out["plaque_area_range"], out["plaque_area_mean"]))

    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        high = plaque_burden >= thr
        key_thr = int(thr * 100)
        run = longest_true_run(high)
        out[f"burden_ge_{key_thr}_frame_count"] = int(high.sum())
        out[f"burden_ge_{key_thr}_frame_ratio"] = float(safe_divide(high.sum(), plaque_burden.size))
        out[f"burden_ge_{key_thr}_longest_run"] = run
        out[f"burden_ge_{key_thr}_longest_run_ratio"] = float(safe_divide(run, plaque_burden.size))

    lumen_median = out["lumen_area_median"]
    lumen_mean = out["lumen_area_mean"]
    lumen_max = out["lumen_area_max"]
    for base_name, base_value in [("median", lumen_median), ("mean", lumen_mean), ("max", lumen_max)]:
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            narrow = lumen_area_ne <= ratio * base_value
            key = f"lumen_le_{int(ratio * 100)}pct_{base_name}"
            run = longest_true_run(narrow)
            out[f"{key}_frame_count"] = int(narrow.sum())
            out[f"{key}_frame_ratio"] = float(safe_divide(narrow.sum(), lumen_area_ne.size))
            out[f"{key}_longest_run"] = run
            out[f"{key}_longest_run_ratio"] = float(safe_divide(run, lumen_area_ne.size))

    for pct in [5, 10, 20]:
        threshold = np.percentile(lumen_area_ne, pct)
        narrow = lumen_area_ne <= threshold
        run = longest_true_run(narrow)
        out[f"lumen_bottom_{pct}pct_threshold"] = float(threshold)
        out[f"lumen_bottom_{pct}pct_frame_ratio"] = float(safe_divide(narrow.sum(), lumen_area_ne.size))
        out[f"lumen_bottom_{pct}pct_longest_run"] = run
        out[f"lumen_bottom_{pct}pct_longest_run_ratio"] = float(safe_divide(run, lumen_area_ne.size))

    min_lumen_idx = int(np.argmin(lumen_area_ne))
    out["min_lumen_idx_nonempty"] = min_lumen_idx
    out["min_lumen_relative_position"] = float(safe_divide(min_lumen_idx, max(1, lumen_area_ne.size - 1)))
    out["plaque_burden_at_min_lumen"] = float(plaque_burden[min_lumen_idx])
    out["plaque_area_at_min_lumen"] = float(plaque_area_ne[min_lumen_idx])
    out["vessel_area_at_min_lumen"] = float(vessel_area_ne[min_lumen_idx])

    n = lumen_area_ne.size
    regions = {"proximal": segment_indices(n, 0.0, 1.0 / 3.0), "mid": segment_indices(n, 1.0 / 3.0, 2.0 / 3.0), "distal": segment_indices(n, 2.0 / 3.0, 1.0)}
    for name, sl in regions.items():
        add_region_stats(name, lumen_area_ne[sl], plaque_area_ne[sl], vessel_area_ne[sl], plaque_burden[sl], out)

    out["mid_min_lumen_to_proximal_mean_lumen_ratio"] = float(safe_divide(out["mid_lumen_min"], out["proximal_lumen_mean"]))
    out["distal_mean_lumen_to_proximal_mean_lumen_ratio"] = float(safe_divide(out["distal_lumen_mean"], out["proximal_lumen_mean"]))
    out["mid_burden_mean_to_proximal_burden_mean_ratio"] = float(safe_divide(out["mid_burden_mean"], out["proximal_burden_mean"]))
    out["distal_burden_mean_to_proximal_burden_mean_ratio"] = float(safe_divide(out["distal_burden_mean"], out["proximal_burden_mean"]))
    out["max_region_burden_mean"] = float(np.nanmax([out["proximal_burden_mean"], out["mid_burden_mean"], out["distal_burden_mean"]]))
    out["min_region_lumen_mean"] = float(np.nanmin([out["proximal_lumen_mean"], out["mid_lumen_mean"], out["distal_lumen_mean"]]))

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
        out[f"{name}_mean_lumen_to_global_mean_ratio"] = float(safe_divide(out[f"{name}_lumen_mean"], out["lumen_area_mean"]))
        out[f"{name}_mean_burden_to_global_mean_ratio"] = float(safe_divide(out[f"{name}_burden_mean"], out["plaque_burden_mean"]))

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
        for prefix in ["diff_lumen_area", "abs_diff_lumen_area", "diff_plaque_burden", "abs_diff_plaque_burden"]:
            add_basic_stats(prefix, np.array([], dtype=np.float64), out)
        out["max_lumen_drop"] = np.nan
        out["max_lumen_rise"] = np.nan
        out["max_burden_increase"] = np.nan
        out["max_burden_decrease"] = np.nan
    return out


def extract_feature_dict(path: Path) -> dict:
    arr = read_mha(path)
    return extract_features_from_array(arr, serial_no=Path(path).stem)
