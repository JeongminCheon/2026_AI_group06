#!/usr/bin/env python3
# step3_2_train_mlp.py

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

    risk:
        1 = FFR < 0.8
        0 = FFR >= 0.8
    """
    return 1 - ffr_class.astype(int)


def risk_prob_to_ffr_class(p_risk: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p_risk >= threshold, 0, 1).astype(int)


def compute_metrics(y_true_ffr_class: np.ndarray, p_risk: np.ndarray, threshold: float) -> dict:
    y_pred_ffr_class = risk_prob_to_ffr_class(p_risk, threshold)

    y_true_risk = 1 - y_true_ffr_class.astype(int)

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


def tune_threshold(
    y_true_ffr_class: np.ndarray,
    p_risk: np.ndarray,
    metric_name: str,
) -> tuple[float, dict]:
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
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    std = X.std(axis=0)
    keep_cols = std[std > min_std].index.tolist()

    return keep_cols


def make_xy(
    train_df: pd.DataFrame,
    feature_cols: list[str],
):
    X_df = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(np.float32)

    y_ffr_class = train_df["ffr_class"].astype(int).values
    y_risk = get_risk_target(y_ffr_class).astype(np.float32)

    return X, y_ffr_class, y_risk


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512):
    model.eval()
    probs = []

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs, axis=0)


def train_one_fold(
    fold: int,
    X: np.ndarray,
    y_ffr_class: np.ndarray,
    y_risk: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    args,
    device: torch.device,
    out_dir: Path,
):
    X_train_raw = X[train_idx]
    X_val_raw = X[val_idx]

    y_train = y_risk[train_idx]
    y_val_ffr_class = y_ffr_class[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = TabularMLP(
        input_dim=X.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    pos_weight_value = neg_count / max(pos_count, 1.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -1e18
    best_state = None
    best_epoch = -1
    patience_count = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        p_val = predict_proba(model, X_val, device=device, batch_size=args.eval_batch_size)

        # For early stopping, use threshold-free ROC-AUC first.
        # If ROC-AUC fails, fallback to balanced accuracy at 0.5.
        val_metrics_05 = compute_metrics(y_val_ffr_class, p_val, threshold=0.5)
        score = val_metrics_05.get("roc_auc_risk", np.nan)
        if not np.isfinite(score):
            score = val_metrics_05["balanced_accuracy"]

        history_row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_roc_auc_risk": val_metrics_05["roc_auc_risk"],
            "val_accuracy_05": val_metrics_05["accuracy"],
            "val_balanced_accuracy_05": val_metrics_05["balanced_accuracy"],
            "val_recall_class0_05": val_metrics_05["recall_class0"],
        }
        history.append(history_row)

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"[fold {fold}] epoch {epoch:03d} "
                f"loss={np.mean(losses):.5f} "
                f"auc={val_metrics_05['roc_auc_risk']:.4f} "
                f"bal_acc@0.5={val_metrics_05['balanced_accuracy']:.4f} "
                f"recall0@0.5={val_metrics_05['recall_class0']:.4f}"
            )

        if patience_count >= args.patience:
            print(f"[fold {fold}] early stopping at epoch {epoch}, best_epoch={best_epoch}")
            break

    model.load_state_dict(best_state)
    p_val = predict_proba(model, X_val, device=device, batch_size=args.eval_batch_size)

    # Save fold checkpoint.
    checkpoint = {}

    for key, value in model.state_dict().items():
        checkpoint[f"model.{key}"] = value.detach().cpu()

    checkpoint["scaler.mean"] = torch.tensor(scaler.mean_, dtype=torch.float32)
    checkpoint["scaler.scale"] = torch.tensor(scaler.scale_, dtype=torch.float32)
    checkpoint["meta.pos_weight"] = torch.tensor([pos_weight_value], dtype=torch.float32)
    checkpoint["meta.input_dim"] = torch.tensor([X.shape[1]], dtype=torch.int64)

    ckpt_path = out_dir / f"mlp_fold{fold}.safetensors"
    save_file(checkpoint, str(ckpt_path))
    print(f"[SAVE] {ckpt_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"history_fold{fold}.csv", index=False)

    return {
        "fold": fold,
        "model": model,
        "scaler": scaler,
        "val_idx": val_idx,
        "p_val": p_val,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "pos_weight": pos_weight_value,
        "checkpoint_path": str(ckpt_path),
    }


def save_quickcheck_csv(serials, p_risk, threshold, out_path: Path):
    pred_class = risk_prob_to_ffr_class(p_risk, threshold)
    df = pd.DataFrame({
        "serial_no": serials,
        "ffr_class": pred_class.astype(int),
    })
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


def write_comparison(summary_df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("Step 3-2 PyTorch MLP Result Summary")
    lines.append("=" * 70)
    lines.append("")

    lines.append("[OOF metrics]")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("[Interpretation guide]")
    lines.append("- Compare with Step 3-1 ExtraTrees balanced_accuracy = 0.7296, ROC-AUC = 0.7904.")
    lines.append("- If MLP reaches balanced_accuracy >= 0.70 and ROC-AUC >= 0.76, it is usable as a final-friendly baseline.")
    lines.append("- If MLP is much worse than ExtraTrees, keep it as final-submission baseline but continue with CNN/Transformer experiments.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-features", type=str, default="step2_outputs/train_features.csv")
    parser.add_argument("--test-features", type=str, default="step2_outputs/test_public_features.csv")
    parser.add_argument("--out-dir", type=str, default="step3_2_mlp_outputs")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] device = {device}")

    train_df = pd.read_csv(args.train_features)
    test_df = pd.read_csv(args.test_features) if Path(args.test_features).exists() else None

    feature_cols = prepare_feature_columns(train_df)
    X, y_ffr_class, y_risk = make_xy(train_df, feature_cols)

    print(f"[INFO] train rows = {len(train_df)}")
    print(f"[INFO] feature dim = {len(feature_cols)}")
    print(f"[INFO] class counts ffr_class = {dict(pd.Series(y_ffr_class).value_counts().sort_index())}")
    print(f"[INFO] risk counts = {dict(pd.Series(y_risk.astype(int)).value_counts().sort_index())}")

    # Save feature metadata.
    feature_meta = {
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "target_definition": "risk=1 if FFR<0.8 else 0; ffr_class=0 if risk probability >= threshold",
    }

    (out_dir / "feature_metadata.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[SAVE] {out_dir / 'feature_metadata.json'}")

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    folds = np.full(len(train_df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(X, y_ffr_class)):
        folds[val_idx] = fold

    pd.DataFrame({
        "serial_no": train_df["serial_no"].astype(str).values,
        "fold": folds,
        "FFR": train_df["FFR"].values,
        "ffr_class": y_ffr_class,
        "risk": y_risk.astype(int),
    }).to_csv(out_dir / "fold_assignments.csv", index=False)

    oof_p_risk = np.zeros(len(train_df), dtype=np.float64)
    fold_results = []

    trained_folds = []

    for fold in range(args.n_splits):
        print("")
        print("=" * 80)
        print(f"[TRAIN] fold {fold}")

        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        result = train_one_fold(
            fold=fold,
            X=X,
            y_ffr_class=y_ffr_class,
            y_risk=y_risk,
            train_idx=train_idx,
            val_idx=val_idx,
            feature_cols=feature_cols,
            args=args,
            device=device,
            out_dir=out_dir,
        )

        oof_p_risk[val_idx] = result["p_val"]
        trained_folds.append(result)

        fold_metrics_05 = compute_metrics(y_ffr_class[val_idx], result["p_val"], threshold=0.5)
        fold_metrics_05.update({
            "fold": fold,
            "threshold_type": "fixed_0.5",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "pos_weight": result["pos_weight"],
        })
        fold_results.append(fold_metrics_05)

        thr_bal, metrics_bal = tune_threshold(
            y_ffr_class[val_idx],
            result["p_val"],
            metric_name="balanced_accuracy",
        )
        metrics_bal.update({
            "fold": fold,
            "threshold_type": "fold_tuned_balanced_accuracy",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
            "pos_weight": result["pos_weight"],
        })
        fold_results.append(metrics_bal)

    # OOF-level metrics and threshold tuning.
    summary_rows = []

    metrics_05 = compute_metrics(y_ffr_class, oof_p_risk, threshold=0.5)
    metrics_05.update({
        "model": "pytorch_mlp",
        "threshold_type": "fixed_0.5",
        "selected_threshold": 0.5,
    })
    summary_rows.append(metrics_05)

    thr_acc, metrics_acc = tune_threshold(y_ffr_class, oof_p_risk, metric_name="accuracy")
    metrics_acc.update({
        "model": "pytorch_mlp",
        "threshold_type": "oof_tuned_accuracy",
        "selected_threshold": thr_acc,
    })
    summary_rows.append(metrics_acc)

    thr_bal, metrics_bal = tune_threshold(y_ffr_class, oof_p_risk, metric_name="balanced_accuracy")
    metrics_bal.update({
        "model": "pytorch_mlp",
        "threshold_type": "oof_tuned_balanced_accuracy",
        "selected_threshold": thr_bal,
    })
    summary_rows.append(metrics_bal)

    thr_f1, metrics_f1 = tune_threshold(y_ffr_class, oof_p_risk, metric_name="f1_macro")
    metrics_f1.update({
        "model": "pytorch_mlp",
        "threshold_type": "oof_tuned_f1_macro",
        "selected_threshold": thr_f1,
    })
    summary_rows.append(metrics_f1)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    print(f"[SAVE] {out_dir / 'cv_results_summary.csv'}")

    per_fold_df = pd.DataFrame(fold_results)
    per_fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    print(f"[SAVE] {out_dir / 'per_fold_metrics.csv'}")

    oof_df = pd.DataFrame({
        "serial_no": train_df["serial_no"].astype(str).values,
        "fold": folds,
        "FFR": train_df["FFR"].values,
        "true_ffr_class": y_ffr_class,
        "true_risk": y_risk.astype(int),
        "p_risk": oof_p_risk,
        "pred_ffr_class_thr05": risk_prob_to_ffr_class(oof_p_risk, 0.5),
        "pred_ffr_class_thr_bal": risk_prob_to_ffr_class(oof_p_risk, thr_bal),
        "pred_ffr_class_thr_acc": risk_prob_to_ffr_class(oof_p_risk, thr_acc),
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    print(f"[SAVE] {out_dir / 'oof_predictions.csv'}")

    # Save global metadata including thresholds.
    global_meta_tensors = {
        "threshold.fixed_0p5": torch.tensor([0.5], dtype=torch.float32),
        "threshold.oof_accuracy": torch.tensor([thr_acc], dtype=torch.float32),
        "threshold.oof_balanced_accuracy": torch.tensor([thr_bal], dtype=torch.float32),
        "threshold.oof_f1_macro": torch.tensor([thr_f1], dtype=torch.float32),
    }
    save_file(global_meta_tensors, str(out_dir / "thresholds.safetensors"))
    print(f"[SAVE] {out_dir / 'thresholds.safetensors'}")

    write_comparison(summary_df, out_dir / "model_comparison.txt")

    # Test prediction with fold ensemble.
    if test_df is not None:
        X_test_df = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_test_df = X_test_df.fillna(train_df[feature_cols].median(numeric_only=True))
        X_test_raw = X_test_df.values.astype(np.float32)

        test_probs = []

        for result in trained_folds:
            model = result["model"]
            scaler = result["scaler"]

            X_test = scaler.transform(X_test_raw).astype(np.float32)
            p_test = predict_proba(
                model,
                X_test,
                device=device,
                batch_size=args.eval_batch_size,
            )
            test_probs.append(p_test)

        p_test_mean = np.mean(np.stack(test_probs, axis=0), axis=0)

        test_pred_df = pd.DataFrame({
            "serial_no": test_df["serial_no"].astype(str).values,
            "p_risk": p_test_mean,
            "ffr_class_thr05": risk_prob_to_ffr_class(p_test_mean, 0.5),
            "ffr_class_thr_bal": risk_prob_to_ffr_class(p_test_mean, thr_bal),
            "ffr_class_thr_acc": risk_prob_to_ffr_class(p_test_mean, thr_acc),
            "ffr_class_thr_f1": risk_prob_to_ffr_class(p_test_mean, thr_f1),
        })
        test_pred_df.to_csv(out_dir / "test_public_probabilities.csv", index=False)
        print(f"[SAVE] {out_dir / 'test_public_probabilities.csv'}")

        save_quickcheck_csv(
            serials=test_df["serial_no"].astype(str).values,
            p_risk=p_test_mean,
            threshold=thr_bal,
            out_path=out_dir / "quickcheck_mlp_bal.csv",
        )

        save_quickcheck_csv(
            serials=test_df["serial_no"].astype(str).values,
            p_risk=p_test_mean,
            threshold=thr_acc,
            out_path=out_dir / "quickcheck_mlp_acc.csv",
        )

        save_quickcheck_csv(
            serials=test_df["serial_no"].astype(str).values,
            p_risk=p_test_mean,
            threshold=0.5,
            out_path=out_dir / "quickcheck_mlp_thr05.csv",
        )

    print("")
    print("[DONE] Step 3-2 PyTorch MLP completed.")
    print(f"[CHECK] {out_dir / 'model_comparison.txt'}")


if __name__ == "__main__":
    main()