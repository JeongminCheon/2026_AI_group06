#!/usr/bin/env python3
# step3_advanced_mlp.py

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from safetensors.torch import save_file

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


EPS = 1e-8


# -----------------------------
# Utility
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_risk_target(ffr_class: np.ndarray) -> np.ndarray:
    """
    ffr_class:
        0 = FFR < 0.8
        1 = FFR >= 0.8

    risk target:
        1 = FFR < 0.8
        0 = FFR >= 0.8
    """
    return 1 - ffr_class.astype(int)


def risk_prob_to_ffr_class(p_risk: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p_risk >= threshold, 0, 1).astype(int)


def compute_metrics(y_true_ffr_class: np.ndarray, p_risk: np.ndarray, threshold: float) -> dict:
    y_pred_ffr_class = risk_prob_to_ffr_class(p_risk, threshold)
    y_true_risk = get_risk_target(y_true_ffr_class)

    cm = confusion_matrix(y_true_ffr_class, y_pred_ffr_class, labels=[0, 1])

    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_ffr_class, y_pred_ffr_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_ffr_class, y_pred_ffr_class)),
        "f1_macro": float(f1_score(y_true_ffr_class, y_pred_ffr_class, average="macro", zero_division=0)),
        "f1_class0": float(f1_score(y_true_ffr_class, y_pred_ffr_class, pos_label=0, zero_division=0)),
        "f1_class1": float(f1_score(y_true_ffr_class, y_pred_ffr_class, pos_label=1, zero_division=0)),
        "precision_class0": float(precision_score(y_true_ffr_class, y_pred_ffr_class, pos_label=0, zero_division=0)),
        "recall_class0": float(recall_score(y_true_ffr_class, y_pred_ffr_class, pos_label=0, zero_division=0)),
        "precision_class1": float(precision_score(y_true_ffr_class, y_pred_ffr_class, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y_true_ffr_class, y_pred_ffr_class, pos_label=1, zero_division=0)),
        "cm_true0_pred0": int(cm[0, 0]),
        "cm_true0_pred1": int(cm[0, 1]),
        "cm_true1_pred0": int(cm[1, 0]),
        "cm_true1_pred1": int(cm[1, 1]),
    }

    try:
        out["roc_auc_risk"] = float(roc_auc_score(y_true_risk, p_risk))
    except Exception:
        out["roc_auc_risk"] = np.nan

    return out


def tune_threshold(y_true_ffr_class: np.ndarray, p_risk: np.ndarray, metric_name: str):
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 4)

    best_thr = 0.5
    best_metrics = None
    best_value = -1e18

    for thr in thresholds:
        metrics = compute_metrics(y_true_ffr_class, p_risk, float(thr))
        value = metrics[metric_name]
        if value > best_value:
            best_value = value
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


# -----------------------------
# Feature preparation
# -----------------------------
def prepare_feature_columns(train_df: pd.DataFrame, min_std: float = 1e-12):
    drop_cols = {
        "serial_no",
        "file_path",
        "error",
        "unique_values",
        "FFR",
        "ffr_class",
    }

    candidate_cols = []
    for col in train_df.columns:
        if col in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            candidate_cols.append(col)

    X = train_df[candidate_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    std = X.std(axis=0)
    keep_cols = std[std > min_std].index.tolist()

    return keep_cols


def make_arrays(train_df: pd.DataFrame, feature_cols: list[str]):
    X_df = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.values.astype(np.float32)
    ffr = train_df["FFR"].astype(float).values.astype(np.float32)
    y_ffr_class = train_df["ffr_class"].astype(int).values
    y_risk = get_risk_target(y_ffr_class).astype(np.float32)

    return X, ffr, y_ffr_class, y_risk


def select_features(
    method: str,
    X_train_raw: np.ndarray,
    y_train_risk: np.ndarray,
    ffr_train: np.ndarray,
    top_k: int,
    seed: int,
):
    """
    Feature selection is performed inside each fold using only fold-train data.

    method:
        none
        corr_risk
        corr_ffr
        tree
    """
    n_features = X_train_raw.shape[1]

    if method == "none" or top_k <= 0 or top_k >= n_features:
        return np.arange(n_features)

    if method == "corr_risk":
        target = y_train_risk.astype(np.float32)
        scores = []
        for j in range(n_features):
            x = X_train_raw[:, j]
            if np.std(x) < EPS:
                scores.append(0.0)
            else:
                scores.append(abs(np.corrcoef(x, target)[0, 1]))
        scores = np.nan_to_num(np.asarray(scores), nan=0.0)
        return np.argsort(scores)[::-1][:top_k]

    if method == "corr_ffr":
        target = ffr_train.astype(np.float32)
        scores = []
        for j in range(n_features):
            x = X_train_raw[:, j]
            if np.std(x) < EPS:
                scores.append(0.0)
            else:
                scores.append(abs(np.corrcoef(x, target)[0, 1]))
        scores = np.nan_to_num(np.asarray(scores), nan=0.0)
        return np.argsort(scores)[::-1][:top_k]

    if method == "tree":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train_raw, y_train_risk.astype(int))
        scores = clf.feature_importances_
        return np.argsort(scores)[::-1][:top_k]

    raise ValueError(f"Unknown feature selection method: {method}")


# -----------------------------
# Model
# -----------------------------
class AdvancedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        aux_regression: bool,
    ):
        super().__init__()
        self.aux_regression = aux_regression

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev_dim, 1)

        if aux_regression:
            self.reg_head = nn.Linear(prev_dim, 1)
        else:
            self.reg_head = None

    def forward(self, x):
        h = self.backbone(x)
        logit = self.cls_head(h).squeeze(-1)

        if self.aux_regression:
            ffr_z = self.reg_head(h).squeeze(-1)
            return logit, ffr_z

        return logit


def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    probs = []

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs, axis=0)


# -----------------------------
# Training
# -----------------------------
def train_one_fold(
    seed: int,
    fold: int,
    X: np.ndarray,
    ffr: np.ndarray,
    y_ffr_class: np.ndarray,
    y_risk: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    args,
    device: torch.device,
    out_dir: Path,
):
    X_train_raw_full = X[train_idx]
    X_val_raw_full = X[val_idx]

    y_train_risk = y_risk[train_idx]
    y_val_ffr_class = y_ffr_class[val_idx]
    ffr_train = ffr[train_idx]
    ffr_val = ffr[val_idx]

    # Fold-internal feature selection
    selected_idx = select_features(
        method=args.feature_selection,
        X_train_raw=X_train_raw_full,
        y_train_risk=y_train_risk,
        ffr_train=ffr_train,
        top_k=args.top_k,
        seed=seed + fold,
    )

    X_train_raw = X_train_raw_full[:, selected_idx]
    X_val_raw = X_val_raw_full[:, selected_idx]

    selected_feature_cols = [feature_cols[i] for i in selected_idx.tolist()]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    # FFR regression target normalization within fold
    ffr_mean = float(np.mean(ffr_train))
    ffr_std = float(np.std(ffr_train) + EPS)
    ffr_train_z = ((ffr_train - ffr_mean) / ffr_std).astype(np.float32)

    train_tensors = [
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_risk, dtype=torch.float32),
    ]

    if args.aux_regression:
        train_tensors.append(torch.tensor(ffr_train_z, dtype=torch.float32))

    train_ds = TensorDataset(*train_tensors)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = AdvancedMLP(
        input_dim=X_train.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        aux_regression=args.aux_regression,
    ).to(device)

    pos_count = float((y_train_risk == 1).sum())
    neg_count = float((y_train_risk == 0).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -1e18
    best_epoch = -1
    best_state = None
    patience_count = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_bce_losses = []
        train_reg_losses = []

        for batch in train_loader:
            xb = batch[0].to(device)
            yb = batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)

            if args.aux_regression:
                ffr_zb = batch[2].to(device)
                logits, pred_ffr_z = model(xb)
                loss_bce = bce_loss(logits, yb)
                loss_reg = mse_loss(pred_ffr_z, ffr_zb)
                loss = loss_bce + args.reg_alpha * loss_reg
            else:
                logits = model(xb)
                loss_bce = bce_loss(logits, yb)
                loss_reg = torch.tensor(0.0, device=device)
                loss = loss_bce

            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            train_bce_losses.append(float(loss_bce.detach().cpu().item()))
            train_reg_losses.append(float(loss_reg.detach().cpu().item()))

        p_val = predict_proba(model, X_val, device=device, batch_size=args.eval_batch_size)
        val_metrics_05 = compute_metrics(y_val_ffr_class, p_val, threshold=0.5)

        # Early stopping score
        if args.early_metric == "roc_auc":
            score = val_metrics_05["roc_auc_risk"]
            if not np.isfinite(score):
                score = val_metrics_05["balanced_accuracy"]
        elif args.early_metric == "balanced_accuracy":
            score = val_metrics_05["balanced_accuracy"]
        else:
            score = val_metrics_05["f1_macro"]

        history.append({
            "seed": seed,
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_bce_loss": float(np.mean(train_bce_losses)),
            "train_reg_loss": float(np.mean(train_reg_losses)),
            "val_roc_auc_risk": val_metrics_05["roc_auc_risk"],
            "val_accuracy_05": val_metrics_05["accuracy"],
            "val_balanced_accuracy_05": val_metrics_05["balanced_accuracy"],
            "val_recall_class0_05": val_metrics_05["recall_class0"],
            "num_selected_features": int(len(selected_idx)),
        })

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[seed {seed} fold {fold}] epoch {epoch:03d} "
                f"loss={np.mean(train_losses):.5f} "
                f"bce={np.mean(train_bce_losses):.5f} "
                f"reg={np.mean(train_reg_losses):.5f} "
                f"auc={val_metrics_05['roc_auc_risk']:.4f} "
                f"bal@0.5={val_metrics_05['balanced_accuracy']:.4f} "
                f"recall0@0.5={val_metrics_05['recall_class0']:.4f}"
            )

        if patience_count >= args.patience:
            print(
                f"[seed {seed} fold {fold}] early stopping at epoch {epoch}, "
                f"best_epoch={best_epoch}, best_score={best_score:.5f}"
            )
            break

    model.load_state_dict(best_state)
    p_val = predict_proba(model, X_val, device=device, batch_size=args.eval_batch_size)

    # Save model checkpoint
    ckpt = {}
    for key, value in model.state_dict().items():
        ckpt[f"model.{key}"] = value.detach().cpu()

    ckpt["scaler.mean"] = torch.tensor(scaler.mean_, dtype=torch.float32)
    ckpt["scaler.scale"] = torch.tensor(scaler.scale_, dtype=torch.float32)
    ckpt["feature.selected_idx"] = torch.tensor(selected_idx.copy(), dtype=torch.int64)
    ckpt["meta.input_dim"] = torch.tensor([X_train.shape[1]], dtype=torch.int64)
    ckpt["meta.pos_weight"] = torch.tensor([pos_weight_value], dtype=torch.float32)
    ckpt["meta.ffr_mean"] = torch.tensor([ffr_mean], dtype=torch.float32)
    ckpt["meta.ffr_std"] = torch.tensor([ffr_std], dtype=torch.float32)

    ckpt_path = out_dir / f"mlp_seed{seed}_fold{fold}.safetensors"
    save_file(ckpt, str(ckpt_path))
    print(f"[SAVE] {ckpt_path}")

    # Save feature names for this fold
    fold_feature_path = out_dir / f"features_seed{seed}_fold{fold}.json"
    fold_feature_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "fold": fold,
                "feature_selection": args.feature_selection,
                "top_k": args.top_k,
                "num_selected_features": len(selected_feature_cols),
                "selected_feature_cols": selected_feature_cols,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(history).to_csv(out_dir / f"history_seed{seed}_fold{fold}.csv", index=False)

    return {
        "seed": seed,
        "fold": fold,
        "model": model,
        "scaler": scaler,
        "selected_idx": selected_idx,
        "val_idx": val_idx,
        "p_val": p_val,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "pos_weight": pos_weight_value,
        "ffr_mean": ffr_mean,
        "ffr_std": ffr_std,
        "checkpoint_path": str(ckpt_path),
    }


def train_one_seed(seed: int, X, ffr, y_ffr_class, y_risk, feature_cols, args, device, out_dir):
    seed_everything(seed)

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = np.full(len(y_ffr_class), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(X, y_ffr_class)):
        folds[val_idx] = fold

    oof_p = np.zeros(len(y_ffr_class), dtype=np.float64)
    fold_results = []
    per_fold_metrics = []

    for fold in range(args.n_splits):
        print("")
        print("=" * 80)
        print(f"[TRAIN] seed={seed}, fold={fold}")

        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        result = train_one_fold(
            seed=seed,
            fold=fold,
            X=X,
            ffr=ffr,
            y_ffr_class=y_ffr_class,
            y_risk=y_risk,
            train_idx=train_idx,
            val_idx=val_idx,
            feature_cols=feature_cols,
            args=args,
            device=device,
            out_dir=out_dir,
        )

        fold_results.append(result)
        oof_p[val_idx] = result["p_val"]

        metrics_05 = compute_metrics(y_ffr_class[val_idx], result["p_val"], threshold=0.5)
        metrics_05.update({
            "seed": seed,
            "fold": fold,
            "threshold_type": "fixed_0.5",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "num_selected_features": int(len(result["selected_idx"])),
        })
        per_fold_metrics.append(metrics_05)

        thr_bal, metrics_bal = tune_threshold(
            y_ffr_class[val_idx],
            result["p_val"],
            metric_name="balanced_accuracy",
        )
        metrics_bal.update({
            "seed": seed,
            "fold": fold,
            "threshold_type": "fold_tuned_balanced_accuracy",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "num_selected_features": int(len(result["selected_idx"])),
        })
        per_fold_metrics.append(metrics_bal)

    return {
        "seed": seed,
        "folds": folds,
        "oof_p": oof_p,
        "fold_results": fold_results,
        "per_fold_metrics": per_fold_metrics,
    }


def predict_test_for_fold_results(fold_results, X_test_raw, device, batch_size):
    probs = []

    for item in fold_results:
        model = item["model"]
        scaler = item["scaler"]
        selected_idx = item["selected_idx"]

        X_selected = X_test_raw[:, selected_idx]
        X_scaled = scaler.transform(X_selected).astype(np.float32)

        p = predict_proba(model, X_scaled, device=device, batch_size=batch_size)
        probs.append(p)

    return np.mean(np.stack(probs, axis=0), axis=0)


def save_quickcheck_csv(serials, p_risk, threshold, out_path: Path):
    pred_class = risk_prob_to_ffr_class(p_risk, threshold)
    df = pd.DataFrame({
        "serial_no": serials,
        "ffr_class": pred_class.astype(int),
    })
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


def write_summary(summary_df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("Step 3 Advanced MLP Summary")
    lines.append("=" * 80)
    lines.append("")

    lines.append("[OOF metrics]")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("[Reference]")
    lines.append("Step 3-1 ExtraTrees balanced_accuracy = 0.7296, ROC-AUC = 0.7904")
    lines.append("Step 3-2 basic MLP balanced_accuracy = 0.7401, ROC-AUC = 0.7939")
    lines.append("")

    lines.append("[Notes]")
    lines.append("- oof_tuned_balanced_accuracy is best for balanced clinical interpretation.")
    lines.append("- oof_tuned_accuracy is useful if public leaderboard uses plain accuracy.")
    lines.append("- Compare recall_class0 to check whether FFR<0.8 cases are being missed.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-features", type=str, default="step2_outputs/train_features.csv")
    parser.add_argument("--test-features", type=str, default="step2_outputs/test_public_features.csv")
    parser.add_argument("--out-dir", type=str, default="step3_advanced_mlp_outputs")

    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-splits", type=int, default=5)

    # Model size: Step 3-3
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.25)

    # Multi-task auxiliary regression: Step 3-4
    parser.add_argument("--aux-regression", action="store_true")
    parser.add_argument("--reg-alpha", type=float, default=0.1)

    # Feature selection: Step 3-5
    parser.add_argument(
        "--feature-selection",
        type=str,
        default="none",
        choices=["none", "corr_risk", "corr_ffr", "tree"],
    )
    parser.add_argument("--top-k", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--early-metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "balanced_accuracy", "f1_macro"],
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] device = {device}")
    print(f"[INFO] seeds = {args.seeds}")
    print(f"[INFO] hidden_dims = {args.hidden_dims}")
    print(f"[INFO] aux_regression = {args.aux_regression}")
    print(f"[INFO] feature_selection = {args.feature_selection}, top_k = {args.top_k}")

    train_df = pd.read_csv(args.train_features)
    test_df = pd.read_csv(args.test_features) if Path(args.test_features).exists() else None

    feature_cols = prepare_feature_columns(train_df)
    X, ffr, y_ffr_class, y_risk = make_arrays(train_df, feature_cols)

    print(f"[INFO] train rows = {len(train_df)}")
    print(f"[INFO] base feature dim = {len(feature_cols)}")
    print(f"[INFO] ffr_class counts = {dict(pd.Series(y_ffr_class).value_counts().sort_index())}")
    print(f"[INFO] risk counts = {dict(pd.Series(y_risk.astype(int)).value_counts().sort_index())}")

    metadata = {
        "feature_cols": feature_cols,
        "num_base_features": len(feature_cols),
        "seeds": args.seeds,
        "n_splits": args.n_splits,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "aux_regression": args.aux_regression,
        "reg_alpha": args.reg_alpha,
        "feature_selection": args.feature_selection,
        "top_k": args.top_k,
        "target_definition": "risk=1 if FFR<0.8 else 0",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[SAVE] {out_dir / 'metadata.json'}")

    all_seed_results = []
    all_oof_seed_probs = []
    all_per_fold_metrics = []

    for seed in args.seeds:
        result = train_one_seed(
            seed=seed,
            X=X,
            ffr=ffr,
            y_ffr_class=y_ffr_class,
            y_risk=y_risk,
            feature_cols=feature_cols,
            args=args,
            device=device,
            out_dir=out_dir,
        )

        all_seed_results.append(result)
        all_oof_seed_probs.append(result["oof_p"])
        all_per_fold_metrics.extend(result["per_fold_metrics"])

        pd.DataFrame({
            "serial_no": train_df["serial_no"].astype(str).values,
            "fold": result["folds"],
            "FFR": ffr,
            "ffr_class": y_ffr_class,
            "risk": y_risk.astype(int),
            "p_risk": result["oof_p"],
            "seed": seed,
        }).to_csv(out_dir / f"oof_predictions_seed{seed}.csv", index=False)

    # Ensemble across seeds
    oof_p_ensemble = np.mean(np.stack(all_oof_seed_probs, axis=0), axis=0)

    summary_rows = []

    for threshold_type, threshold, metric_name in [
        ("fixed_0.5", 0.5, None),
        ("oof_tuned_accuracy", None, "accuracy"),
        ("oof_tuned_balanced_accuracy", None, "balanced_accuracy"),
        ("oof_tuned_f1_macro", None, "f1_macro"),
    ]:
        if threshold is None:
            thr, metrics = tune_threshold(y_ffr_class, oof_p_ensemble, metric_name)
        else:
            thr = threshold
            metrics = compute_metrics(y_ffr_class, oof_p_ensemble, threshold)

        metrics.update({
            "model": "advanced_mlp_seed_ensemble",
            "num_seeds": len(args.seeds),
            "seeds": ",".join(map(str, args.seeds)),
            "threshold_type": threshold_type,
            "selected_threshold": float(thr),
            "hidden_dims": "-".join(map(str, args.hidden_dims)),
            "dropout": args.dropout,
            "aux_regression": args.aux_regression,
            "reg_alpha": args.reg_alpha,
            "feature_selection": args.feature_selection,
            "top_k": args.top_k,
        })
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    print(f"[SAVE] {out_dir / 'cv_results_summary.csv'}")

    pd.DataFrame(all_per_fold_metrics).to_csv(out_dir / "per_fold_metrics.csv", index=False)
    print(f"[SAVE] {out_dir / 'per_fold_metrics.csv'}")

    oof_df = pd.DataFrame({
        "serial_no": train_df["serial_no"].astype(str).values,
        "FFR": ffr,
        "true_ffr_class": y_ffr_class,
        "true_risk": y_risk.astype(int),
        "p_risk": oof_p_ensemble,
    })

    # Add tuned predictions
    thr_acc = summary_df.loc[summary_df["threshold_type"] == "oof_tuned_accuracy", "selected_threshold"].iloc[0]
    thr_bal = summary_df.loc[summary_df["threshold_type"] == "oof_tuned_balanced_accuracy", "selected_threshold"].iloc[0]
    thr_f1 = summary_df.loc[summary_df["threshold_type"] == "oof_tuned_f1_macro", "selected_threshold"].iloc[0]

    oof_df["pred_ffr_class_thr05"] = risk_prob_to_ffr_class(oof_p_ensemble, 0.5)
    oof_df["pred_ffr_class_thr_acc"] = risk_prob_to_ffr_class(oof_p_ensemble, thr_acc)
    oof_df["pred_ffr_class_thr_bal"] = risk_prob_to_ffr_class(oof_p_ensemble, thr_bal)
    oof_df["pred_ffr_class_thr_f1"] = risk_prob_to_ffr_class(oof_p_ensemble, thr_f1)
    oof_df.to_csv(out_dir / "oof_predictions_ensemble.csv", index=False)
    print(f"[SAVE] {out_dir / 'oof_predictions_ensemble.csv'}")

    # Save thresholds as safetensors
    save_file(
        {
            "threshold.fixed_0p5": torch.tensor([0.5], dtype=torch.float32),
            "threshold.oof_accuracy": torch.tensor([thr_acc], dtype=torch.float32),
            "threshold.oof_balanced_accuracy": torch.tensor([thr_bal], dtype=torch.float32),
            "threshold.oof_f1_macro": torch.tensor([thr_f1], dtype=torch.float32),
        },
        str(out_dir / "thresholds.safetensors"),
    )
    print(f"[SAVE] {out_dir / 'thresholds.safetensors'}")

    write_summary(summary_df, out_dir / "model_comparison.txt")

    # Test prediction
    if test_df is not None:
        X_test_df = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_test_df = X_test_df.fillna(train_df[feature_cols].median(numeric_only=True))
        X_test_raw = X_test_df.values.astype(np.float32)

        test_probs_seed = []

        for result in all_seed_results:
            p_seed = predict_test_for_fold_results(
                fold_results=result["fold_results"],
                X_test_raw=X_test_raw,
                device=device,
                batch_size=args.eval_batch_size,
            )
            test_probs_seed.append(p_seed)

        p_test_ensemble = np.mean(np.stack(test_probs_seed, axis=0), axis=0)

        test_prob_df = pd.DataFrame({
            "serial_no": test_df["serial_no"].astype(str).values,
            "p_risk": p_test_ensemble,
            "ffr_class_thr05": risk_prob_to_ffr_class(p_test_ensemble, 0.5),
            "ffr_class_thr_acc": risk_prob_to_ffr_class(p_test_ensemble, thr_acc),
            "ffr_class_thr_bal": risk_prob_to_ffr_class(p_test_ensemble, thr_bal),
            "ffr_class_thr_f1": risk_prob_to_ffr_class(p_test_ensemble, thr_f1),
        })
        test_prob_df.to_csv(out_dir / "test_public_probabilities.csv", index=False)
        print(f"[SAVE] {out_dir / 'test_public_probabilities.csv'}")

        save_quickcheck_csv(
            test_df["serial_no"].astype(str).values,
            p_test_ensemble,
            0.5,
            out_dir / "quickcheck_advanced_mlp_thr05.csv",
        )
        save_quickcheck_csv(
            test_df["serial_no"].astype(str).values,
            p_test_ensemble,
            thr_acc,
            out_dir / "quickcheck_advanced_mlp_acc.csv",
        )
        save_quickcheck_csv(
            test_df["serial_no"].astype(str).values,
            p_test_ensemble,
            thr_bal,
            out_dir / "quickcheck_advanced_mlp_bal.csv",
        )
        save_quickcheck_csv(
            test_df["serial_no"].astype(str).values,
            p_test_ensemble,
            thr_f1,
            out_dir / "quickcheck_advanced_mlp_f1.csv",
        )

    print("")
    print("[DONE] Step 3 advanced MLP completed.")
    print(f"[CHECK] {out_dir / 'model_comparison.txt'}")


if __name__ == "__main__":
    main()