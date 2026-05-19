#!/usr/bin/env python3
# step6_3_prepare_a5_final_submit.py
#
# Build final submission ZIP for current best public model:
#   A5: p_final = 0.5 * p_3seed + 0.5 * p_7seed, threshold = 0.50
#
# Output ZIP contains only .py and .safetensors files.

import argparse
import shutil
import zipfile
from pathlib import Path


A5_MAIN_PY = 'import argparse\nimport csv\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nfrom safetensors.torch import load_file\n\ntry:\n    import helper  # noqa: F401\nexcept ImportError:\n    pass\n\nfrom features import extract_feature_dict\nfrom feature_columns import FEATURE_COLS\nfrom model import AdvancedMLP\n\nSCRIPT_DIR = Path(__file__).resolve().parent\n\nENSEMBLE_WEIGHT_3SEED = 0.50\nENSEMBLE_WEIGHT_7SEED = 0.50\nFINAL_THRESHOLD = 0.50\n\n\ndef parse_args():\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--input-dir", default="/eval/input")\n    parser.add_argument("--output-path", default="/eval/output/predictions.csv")\n    return parser.parse_args()\n\n\ndef _clean_feature_value(v):\n    try:\n        x = float(v)\n    except Exception:\n        x = 0.0\n    if not np.isfinite(x):\n        return 0.0\n    return x\n\n\ndef make_feature_vector(mha_path: Path) -> np.ndarray:\n    feat = extract_feature_dict(mha_path)\n    values = [_clean_feature_value(feat.get(col, 0.0)) for col in FEATURE_COLS]\n    return np.asarray(values, dtype=np.float32)\n\n\ndef _infer_hidden_dims_from_state(state_dict):\n    hidden = []\n    layer_ids = []\n    for key, value in state_dict.items():\n        if key.startswith("backbone.") and key.endswith(".weight") and value.ndim == 2:\n            parts = key.split(".")\n            try:\n                layer_ids.append((int(parts[1]), int(value.shape[0])))\n            except Exception:\n                pass\n    for _, out_dim in sorted(layer_ids):\n        hidden.append(out_dim)\n    if len(hidden) == 0:\n        hidden = [512, 256, 128]\n    return hidden\n\n\ndef load_one_model(ckpt_path: Path):\n    ckpt = load_file(str(ckpt_path), device="cpu")\n\n    if "feature.selected_idx" in ckpt:\n        selected_idx = ckpt["feature.selected_idx"].cpu().numpy().astype(np.int64)\n    else:\n        selected_idx = np.arange(len(FEATURE_COLS), dtype=np.int64)\n\n    scaler_mean = ckpt["scaler.mean"].cpu().numpy().astype(np.float32)\n    scaler_scale = ckpt["scaler.scale"].cpu().numpy().astype(np.float32)\n    scaler_scale = np.where(np.abs(scaler_scale) < 1e-8, 1.0, scaler_scale).astype(np.float32)\n\n    state_dict = {}\n    for key, value in ckpt.items():\n        if key.startswith("model."):\n            state_dict[key[len("model."):]] = value\n\n    input_dim = int(scaler_mean.shape[0])\n    hidden_dims = _infer_hidden_dims_from_state(state_dict)\n\n    model = AdvancedMLP(\n        input_dim=input_dim,\n        hidden_dims=hidden_dims,\n        dropout=0.0,\n        aux_regression=False,\n    )\n    model.load_state_dict(state_dict, strict=True)\n    model.eval()\n\n    return {\n        "path": ckpt_path,\n        "model": model,\n        "selected_idx": selected_idx,\n        "scaler_mean": scaler_mean,\n        "scaler_scale": scaler_scale,\n    }\n\n\ndef load_model_group(prefix: str):\n    paths = sorted(SCRIPT_DIR.glob(f"{prefix}_seed*_fold*.safetensors"))\n    if len(paths) == 0:\n        raise RuntimeError(f"No checkpoints found for prefix={prefix}: expected {prefix}_seed*_fold*.safetensors")\n    return [load_one_model(path) for path in paths]\n\n\ndef predict_group(x_full: np.ndarray, model_items) -> float:\n    probs = []\n    with torch.no_grad():\n        for item in model_items:\n            idx = item["selected_idx"]\n            x = x_full[idx]\n            x = (x - item["scaler_mean"]) / item["scaler_scale"]\n            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)\n            logits = item["model"](x_tensor)\n            p = torch.sigmoid(logits).reshape(-1)[0].item()\n            probs.append(float(p))\n    return float(np.mean(probs))\n\n\ndef predict_one(x_full: np.ndarray, models_3seed, models_7seed) -> float:\n    p3 = predict_group(x_full, models_3seed)\n    p7 = predict_group(x_full, models_7seed)\n    return ENSEMBLE_WEIGHT_3SEED * p3 + ENSEMBLE_WEIGHT_7SEED * p7\n\n\ndef main():\n    args = parse_args()\n\n    input_dir = Path(args.input_dir)\n    output_path = Path(args.output_path)\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n\n    models_3seed = load_model_group("mlp3")\n    models_7seed = load_model_group("mlp7")\n\n    rows = []\n    for path in sorted(input_dir.glob("*.mha")):\n        x = make_feature_vector(path)\n        p_risk = predict_one(x, models_3seed, models_7seed)\n        ffr_class = 0 if p_risk >= FINAL_THRESHOLD else 1\n        rows.append({"serial_no": path.stem, "ffr_class": int(ffr_class)})\n\n    with output_path.open("w", encoding="utf-8", newline="") as handle:\n        writer = csv.DictWriter(handle, fieldnames=["serial_no", "ffr_class"])\n        writer.writeheader()\n        writer.writerows(rows)\n\n\nif __name__ == "__main__":\n    main()\n'


def copy_required_py(template_dir: Path, out_dir: Path):
    required = ["features.py", "model.py", "feature_columns.py"]
    for name in required:
        src = template_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"Missing {src}. Provide --template-dir pointing to a previous MLP final submit folder."
            )
        shutil.copy2(src, out_dir / name)


def copy_checkpoints(src_dir: Path, out_dir: Path, prefix: str):
    paths = sorted(src_dir.glob("mlp_seed*_fold*.safetensors"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No mlp_seed*_fold*.safetensors found in {src_dir}")

    copied = []
    for p in paths:
        new_name = p.name.replace("mlp_", f"{prefix}_", 1)
        dst = out_dir / new_name
        shutil.copy2(p, dst)
        copied.append(dst)
    return copied


def zip_submission(out_dir: Path, zip_path: Path):
    allowed = {".py", ".safetensors"}
    files = sorted([p for p in out_dir.iterdir() if p.is_file()])

    for p in files:
        if p.suffix not in allowed:
            raise RuntimeError(f"Invalid file would be included: {p}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert "main.py" in names
        assert "features.py" in names
        assert "model.py" in names
        assert "feature_columns.py" in names
        assert any(name.startswith("mlp3_") and name.endswith(".safetensors") for name in names)
        assert any(name.startswith("mlp7_") and name.endswith(".safetensors") for name in names)
        for name in names:
            if name.endswith("/"):
                raise RuntimeError(f"Directory found in zip: {name}")
            suffix = "." + name.split(".")[-1]
            if suffix not in allowed:
                raise RuntimeError(f"Invalid file in zip: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-dir", default="group06_ffr_submit",
                        help="Previous MLP final submit folder containing features.py, model.py, feature_columns.py")
    parser.add_argument("--mlp3-dir", default="step3_6_seed_ensemble_mlp_outputs")
    parser.add_argument("--mlp7-dir", default="step5_0_seed_ensemble_7seeds_outputs")
    parser.add_argument("--out-dir", default="group06_ffr_a5_submit")
    parser.add_argument("--zip-path", default="group06_ffr_a5.zip")
    args = parser.parse_args()

    template_dir = Path(args.template_dir)
    mlp3_dir = Path(args.mlp3_dir)
    mlp7_dir = Path(args.mlp7_dir)
    out_dir = Path(args.out_dir)
    zip_path = Path(args.zip_path)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "main.py").write_text(A5_MAIN_PY, encoding="utf-8")
    copy_required_py(template_dir, out_dir)

    copied_3 = copy_checkpoints(mlp3_dir, out_dir, "mlp3")
    copied_7 = copy_checkpoints(mlp7_dir, out_dir, "mlp7")

    print(f"[INFO] copied 3-seed checkpoints: {len(copied_3)}")
    print(f"[INFO] copied 7-seed checkpoints: {len(copied_7)}")

    if len(copied_3) != 15:
        print(f"[WARN] expected 15 3-seed checkpoints, got {len(copied_3)}")
    if len(copied_7) != 35:
        print(f"[WARN] expected 35 7-seed checkpoints, got {len(copied_7)}")

    zip_submission(out_dir, zip_path)

    print(f"[DONE] submit folder: {out_dir}")
    print(f"[DONE] zip created: {zip_path}")
    print("")
    print("Dry-run:")
    print(f"cd {out_dir}")
    print("python main.py --input-dir ../26S_AI536_NE450/test_public --output-path ./predictions.csv")


if __name__ == "__main__":
    main()
