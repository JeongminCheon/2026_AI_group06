#!/usr/bin/env python3
# step4_3_train_attention_mil.py
#
# Phase 4: Simple CNN + Attention MIL pooling.
#
# This script is intentionally close to step4_1_train_cnn_models.py, but replaces
# mean/max pooling with learnable attention pooling over slice embeddings.
#
# Recommended runs:
#
# A. Compare directly with previous 4-6b small setting:
#   python step4_3_train_attention_mil.py \
#     --data-root ./26S_AI536_NE450 \
#     --out-dir step4_6b_attention_s32_outputs \
#     --num-slices 32 --image-size 128 --base-ch 16 --embed-dim 128 \
#     --batch-size 2 --eval-batch-size 2 \
#     --epochs 100 --patience 20 --augment --device cuda
#
# B. More slices with attention pooling:
#   python step4_3_train_attention_mil.py \
#     --data-root ./26S_AI536_NE450 \
#     --out-dir step4_6b_attention_s64_outputs \
#     --num-slices 64 --image-size 128 --base-ch 16 --embed-dim 128 \
#     --batch-size 2 --eval-batch-size 2 \
#     --epochs 100 --patience 20 --augment --device cuda
#
# C. Larger attention model:
#   python step4_3_train_attention_mil.py \
#     --data-root ./26S_AI536_NE450 \
#     --out-dir step4_6b_attention_s64_large_outputs \
#     --num-slices 64 --image-size 128 --base-ch 24 --embed-dim 256 \
#     --batch-size 2 --eval-batch-size 2 \
#     --epochs 100 --patience 20 --augment --device cuda

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


EPS = 1e-8


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_risk_target_from_ffr(ffr: float) -> int:
    return int(ffr < 0.8)


def get_ffr_class_from_ffr(ffr: float) -> int:
    return int(ffr >= 0.8)


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


def read_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return arr.astype(np.uint8, copy=False)


def uniform_indices(n: int, k: int) -> np.ndarray:
    if k <= 1:
        return np.array([n // 2], dtype=np.int64)
    return np.linspace(0, n - 1, k).round().astype(np.int64)


def random_jitter_indices(base_idx: np.ndarray, n: int, jitter: int) -> np.ndarray:
    if jitter <= 0:
        return base_idx
    offset = np.random.randint(-jitter, jitter + 1, size=len(base_idx))
    return np.clip(base_idx + offset, 0, n - 1).astype(np.int64)


def sample_2d_mask_slices(arr, num_slices, image_size, is_train, slice_jitter, augment):
    n = arr.shape[0]
    idx = uniform_indices(n, num_slices)
    if is_train:
        idx = random_jitter_indices(idx, n, slice_jitter)

    sampled = arr[idx]
    x = np.stack([(sampled == 1), (sampled == 2)], axis=1).astype(np.float32)
    x = torch.from_numpy(x)

    if image_size != sampled.shape[-1]:
        x = F.interpolate(x, size=(image_size, image_size), mode="nearest")

    if is_train and augment:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        if random.random() < 0.5:
            x = torch.rot90(x, k=random.randint(0, 3), dims=[-2, -1])

    return x.contiguous()


class IVUS2DSliceDataset(Dataset):
    def __init__(self, df, data_dir, num_slices, image_size, is_train, slice_jitter, augment, has_label):
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.num_slices = num_slices
        self.image_size = image_size
        self.is_train = is_train
        self.slice_jitter = slice_jitter
        self.augment = augment
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        serial_no = str(row["serial_no"])
        arr = read_mha(self.data_dir / f"{serial_no}.mha")
        x = sample_2d_mask_slices(
            arr=arr,
            num_slices=self.num_slices,
            image_size=self.image_size,
            is_train=self.is_train,
            slice_jitter=self.slice_jitter,
            augment=self.augment,
        )
        if self.has_label:
            ffr = float(row["FFR"])
            return {
                "x": x,
                "risk": torch.tensor(get_risk_target_from_ffr(ffr), dtype=torch.float32),
                "ffr_class": torch.tensor(get_ffr_class_from_ffr(ffr), dtype=torch.long),
                "serial_no": serial_no,
            }
        return {"x": x, "serial_no": serial_no}


class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleSliceCNNEncoder(nn.Module):
    def __init__(self, in_ch=2, base_ch=16, embed_dim=128, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct2d(in_ch, base_ch),
            ConvBNAct2d(base_ch, base_ch),
            nn.MaxPool2d(2),

            ConvBNAct2d(base_ch, base_ch * 2),
            ConvBNAct2d(base_ch * 2, base_ch * 2),
            nn.MaxPool2d(2),

            ConvBNAct2d(base_ch * 2, base_ch * 4),
            ConvBNAct2d(base_ch * 4, base_ch * 4),
            nn.MaxPool2d(2),

            ConvBNAct2d(base_ch * 4, base_ch * 8),
            nn.Dropout2d(dropout),
        )
        self.proj = nn.Linear(base_ch * 8, embed_dim)

    def forward(self, x):
        h = self.features(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.proj(h)


class AttentionMILPool(nn.Module):
    """
    Gated attention MIL pooling.
    Ilse et al.-style attention:
      a_i = softmax(w^T (tanh(Vh_i) * sigmoid(Uh_i)))
      z = sum_i a_i h_i
    """
    def __init__(self, embed_dim, attn_dim=None, dropout=0.0):
        super().__init__()
        if attn_dim is None:
            attn_dim = max(64, embed_dim // 2)
        self.v = nn.Linear(embed_dim, attn_dim)
        self.u = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb, return_attention=False):
        # emb: (B,K,E)
        h = torch.tanh(self.v(emb)) * torch.sigmoid(self.u(emb))
        h = self.dropout(h)
        scores = self.w(h).squeeze(-1)  # (B,K)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(emb * weights.unsqueeze(-1), dim=1)
        if return_attention:
            return pooled, weights
        return pooled


class SimpleCNNAttentionMIL(nn.Module):
    def __init__(self, base_ch=16, embed_dim=128, dropout=0.25, attn_dim=None):
        super().__init__()
        self.encoder = SimpleSliceCNNEncoder(
            in_ch=2,
            base_ch=base_ch,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.attn_pool = AttentionMILPool(embed_dim=embed_dim, attn_dim=attn_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        # x: (B,K,2,H,W)
        b, k, c, h, w = x.shape
        x = x.view(b * k, c, h, w)
        emb = self.encoder(x).view(b, k, -1)
        pooled = self.attn_pool(emb)
        return self.classifier(pooled).squeeze(-1)


def make_loader(df, data_dir, args, is_train, has_label):
    ds = IVUS2DSliceDataset(
        df=df,
        data_dir=data_dir,
        num_slices=args.num_slices,
        image_size=args.image_size,
        is_train=is_train,
        slice_jitter=args.slice_jitter,
        augment=args.augment and is_train,
        has_label=has_label,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )


def predict_proba(model, loader, device):
    model.eval()
    probs = []
    serials = []
    true_classes = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            serials.extend(batch["serial_no"])
            if "ffr_class" in batch:
                true_classes.append(batch["ffr_class"].detach().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    true_classes = np.concatenate(true_classes, axis=0) if true_classes else None
    return serials, probs, true_classes


def build_model(args):
    return SimpleCNNAttentionMIL(
        base_ch=args.base_ch,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        attn_dim=args.attn_dim,
    )


def train_one_fold(fold, train_df, val_df, train_dir, args, device, out_dir):
    train_loader = make_loader(train_df, train_dir, args, is_train=True, has_label=True)
    val_loader = make_loader(val_df, train_dir, args, is_train=False, has_label=True)
    model = build_model(args).to(device)

    y_train_risk = train_df["risk"].astype(int).values
    pos = float((y_train_risk == 1).sum())
    neg = float((y_train_risk == 0).sum())
    pos_weight_value = neg / max(pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        scheduler = None

    best_score = -1e18
    best_state = None
    best_epoch = -1
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        start_time = time.time()

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["risk"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        if scheduler is not None:
            scheduler.step()

        _, p_val, y_val_ffr_class = predict_proba(model, val_loader, device)
        metrics_05 = compute_metrics(y_val_ffr_class, p_val, threshold=0.5)

        if args.early_metric == "roc_auc":
            score = metrics_05["roc_auc_risk"]
            if not np.isfinite(score):
                score = metrics_05["balanced_accuracy"]
        elif args.early_metric == "balanced_accuracy":
            score = metrics_05["balanced_accuracy"]
        else:
            score = metrics_05["f1_macro"]

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy_05": metrics_05["accuracy"],
            "val_balanced_accuracy_05": metrics_05["balanced_accuracy"],
            "val_recall_class0_05": metrics_05["recall_class0"],
            "val_roc_auc_risk": metrics_05["roc_auc_risk"],
            "epoch_seconds": time.time() - start_time,
        }
        history.append(row)

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[fold {fold}] epoch {epoch:03d} "
                f"loss={np.mean(losses):.5f} "
                f"auc={metrics_05['roc_auc_risk']:.4f} "
                f"bal@0.5={metrics_05['balanced_accuracy']:.4f} "
                f"recall0@0.5={metrics_05['recall_class0']:.4f} "
                f"time={time.time() - start_time:.1f}s"
            )

        if patience_count >= args.patience:
            print(f"[fold {fold}] early stopping at epoch {epoch}, best_epoch={best_epoch}, best_score={best_score:.5f}")
            break

    model.load_state_dict(best_state)
    serials_val, p_val, y_val_ffr_class = predict_proba(model, val_loader, device)

    ckpt = {}
    for k, v in model.state_dict().items():
        ckpt[f"model.{k}"] = v.detach().cpu()
    ckpt["meta.pos_weight"] = torch.tensor([pos_weight_value], dtype=torch.float32)
    ckpt["meta.best_epoch"] = torch.tensor([best_epoch], dtype=torch.int64)

    ckpt_path = out_dir / f"attention_mil_fold{fold}.safetensors"
    save_file(ckpt, str(ckpt_path))
    print(f"[SAVE] {ckpt_path}")

    pd.DataFrame(history).to_csv(out_dir / f"history_fold{fold}.csv", index=False)

    return {
        "fold": fold,
        "model": model,
        "p_val": p_val,
        "serials_val": serials_val,
        "y_val_ffr_class": y_val_ffr_class,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "checkpoint_path": str(ckpt_path),
    }


def save_quickcheck_csv(serials, p_risk, threshold, out_path):
    pred_class = risk_prob_to_ffr_class(p_risk, threshold)
    pd.DataFrame({"serial_no": serials, "ffr_class": pred_class.astype(int)}).to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


def write_summary(summary_df, out_path):
    lines = []
    lines.append("Step 4 Simple CNN Attention MIL Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[OOF metrics]")
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("[Reference]")
    lines.append("Simple CNN mean_max baseline small: balanced_accuracy=0.6756, ROC-AUC=0.7513")
    lines.append("Step 3-6 MLP ensemble: balanced_accuracy=0.7548, ROC-AUC=0.8048")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--train-dir", type=str, default="train")
    parser.add_argument("--test-dir", type=str, default="test_public")
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="step4_6b_attention_mil_outputs")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-slices", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--slice-jitter", type=int, default=8)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--attn-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--early-metric", type=str, default="roc_auc", choices=["roc_auc", "balanced_accuracy", "f1_macro"])
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()
    seed_everything(args.seed)

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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] model=attention_mil device={device}")
    print(f"[INFO] train_dir={train_dir} test_dir={test_dir} labels_path={labels_path}")

    labels_df = pd.read_csv(labels_path)
    labels_df["serial_no"] = labels_df["serial_no"].astype(str)
    labels_df["ffr_class"] = (labels_df["FFR"] >= 0.8).astype(int)
    labels_df["risk"] = (labels_df["FFR"] < 0.8).astype(int)

    exists = labels_df["serial_no"].apply(lambda s: (train_dir / f"{s}.mha").exists())
    if exists.sum() != len(labels_df):
        print(f"[WARN] missing train files: {len(labels_df) - int(exists.sum())}")
    labels_df = labels_df[exists].reset_index(drop=True)

    y_ffr_class_all = labels_df["ffr_class"].astype(int).values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = np.full(len(labels_df), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(labels_df, y_ffr_class_all)):
        folds[val_idx] = fold
    labels_df["fold"] = folds

    labels_df.to_csv(out_dir / "fold_assignments.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps({**vars(args), "target": "risk=1 if FFR<0.8 else 0"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    oof_p = np.zeros(len(labels_df), dtype=np.float64)
    oof_serials = np.array(labels_df["serial_no"].astype(str).values)
    fold_rows = []
    fold_results = []

    for fold in range(args.n_splits):
        print("")
        print("=" * 80)
        print(f"[TRAIN] fold {fold}")

        train_df = labels_df[labels_df["fold"] != fold].reset_index(drop=True)
        val_df = labels_df[labels_df["fold"] == fold].reset_index(drop=True)

        result = train_one_fold(fold, train_df, val_df, train_dir, args, device, out_dir)
        fold_results.append(result)

        val_index_map = {s: i for i, s in enumerate(oof_serials)}
        for s, p in zip(result["serials_val"], result["p_val"]):
            oof_p[val_index_map[s]] = p

        metrics_05 = compute_metrics(result["y_val_ffr_class"], result["p_val"], threshold=0.5)
        metrics_05.update({
            "model": "simple_cnn_attention_mil",
            "fold": fold,
            "threshold_type": "fixed_0.5",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
        })
        fold_rows.append(metrics_05)

        thr_bal, metrics_bal = tune_threshold(result["y_val_ffr_class"], result["p_val"], "balanced_accuracy")
        metrics_bal.update({
            "model": "simple_cnn_attention_mil",
            "fold": fold,
            "threshold_type": "fold_tuned_balanced_accuracy",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
        })
        fold_rows.append(metrics_bal)

    y_true = y_ffr_class_all
    summary_rows = []
    for name, thr, metric in [
        ("fixed_0.5", 0.5, None),
        ("oof_tuned_accuracy", None, "accuracy"),
        ("oof_tuned_balanced_accuracy", None, "balanced_accuracy"),
        ("oof_tuned_f1_macro", None, "f1_macro"),
    ]:
        if thr is None:
            selected_thr, metrics = tune_threshold(y_true, oof_p, metric)
        else:
            selected_thr = thr
            metrics = compute_metrics(y_true, oof_p, thr)

        metrics.update({
            "model": "simple_cnn_attention_mil",
            "threshold_type": name,
            "selected_threshold": float(selected_thr),
            "num_slices": args.num_slices,
            "image_size": args.image_size,
            "base_ch": args.base_ch,
            "embed_dim": args.embed_dim,
            "dropout": args.dropout,
        })
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "per_fold_metrics.csv", index=False)

    thr_acc = float(summary_df.loc[summary_df["threshold_type"] == "oof_tuned_accuracy", "selected_threshold"].iloc[0])
    thr_bal = float(summary_df.loc[summary_df["threshold_type"] == "oof_tuned_balanced_accuracy", "selected_threshold"].iloc[0])
    thr_f1 = float(summary_df.loc[summary_df["threshold_type"] == "oof_tuned_f1_macro", "selected_threshold"].iloc[0])

    oof_df = pd.DataFrame({
        "serial_no": oof_serials,
        "FFR": labels_df["FFR"].values,
        "true_ffr_class": y_true,
        "true_risk": labels_df["risk"].values,
        "p_risk": oof_p,
        "pred_ffr_class_thr05": risk_prob_to_ffr_class(oof_p, 0.5),
        "pred_ffr_class_acc": risk_prob_to_ffr_class(oof_p, thr_acc),
        "pred_ffr_class_bal": risk_prob_to_ffr_class(oof_p, thr_bal),
        "pred_ffr_class_f1": risk_prob_to_ffr_class(oof_p, thr_f1),
    })
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    save_file(
        {
            "threshold.fixed_0p5": torch.tensor([0.5], dtype=torch.float32),
            "threshold.oof_accuracy": torch.tensor([thr_acc], dtype=torch.float32),
            "threshold.oof_balanced_accuracy": torch.tensor([thr_bal], dtype=torch.float32),
            "threshold.oof_f1_macro": torch.tensor([thr_f1], dtype=torch.float32),
        },
        str(out_dir / "thresholds.safetensors"),
    )

    write_summary(summary_df, out_dir / "model_comparison.txt")

    # public test prediction
    if test_dir.exists():
        test_files = sorted(test_dir.glob("*.mha"))
        if len(test_files) > 0:
            test_df = pd.DataFrame({"serial_no": [p.stem for p in test_files]})
            all_test_probs = []
            test_serials = None
            for result in fold_results:
                test_loader = make_loader(test_df, test_dir, args, is_train=False, has_label=False)
                serials, p_test, _ = predict_proba(result["model"], test_loader, device)
                if test_serials is None:
                    test_serials = serials
                all_test_probs.append(p_test)

            p_test_mean = np.mean(np.stack(all_test_probs, axis=0), axis=0)

            test_prob_df = pd.DataFrame({
                "serial_no": test_serials,
                "p_risk": p_test_mean,
                "ffr_class_thr05": risk_prob_to_ffr_class(p_test_mean, 0.5),
                "ffr_class_acc": risk_prob_to_ffr_class(p_test_mean, thr_acc),
                "ffr_class_bal": risk_prob_to_ffr_class(p_test_mean, thr_bal),
                "ffr_class_f1": risk_prob_to_ffr_class(p_test_mean, thr_f1),
            })
            test_prob_df.to_csv(out_dir / "test_public_probabilities.csv", index=False)

            save_quickcheck_csv(test_serials, p_test_mean, 0.5, out_dir / "quickcheck_attention_mil_thr05.csv")
            save_quickcheck_csv(test_serials, p_test_mean, thr_acc, out_dir / "quickcheck_attention_mil_acc.csv")
            save_quickcheck_csv(test_serials, p_test_mean, thr_bal, out_dir / "quickcheck_attention_mil_bal.csv")
            save_quickcheck_csv(test_serials, p_test_mean, thr_f1, out_dir / "quickcheck_attention_mil_f1.csv")

    print("")
    print("[DONE] Simple CNN Attention MIL completed.")
    print(f"[CHECK] {out_dir / 'model_comparison.txt'}")


if __name__ == "__main__":
    main()
