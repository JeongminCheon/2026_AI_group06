#!/usr/bin/env python3
# step4_train_transformer_models.py
#
# Exp 7-A: Slice-level ViT/DeiT + volume pooling
# Exp 7-B: Slice-feature sequence Transformer
#
# Target:
#   risk = 1 if FFR < 0.8 else 0
#   ffr_class = 0 if risk probability >= threshold else 1
#
# Recommended quick sanity checks:
#
# Exp 7-B sequence transformer:
#   python step4_train_transformer_models.py \
#     --model seq_transformer \
#     --data-root /path/to/26S_AI536_NE450 \
#     --out-dir step4_7b_seq_transformer_test \
#     --seq-len 256 \
#     --epochs 2 \
#     --n-splits 2 \
#     --batch-size 16 \
#     --device cuda
#
# Exp 7-A slice-level ViT/DeiT:
#   python step4_train_transformer_models.py \
#     --model slice_vit \
#     --vit-name deit_tiny_patch16_224 \
#     --data-root /path/to/26S_AI536_NE450 \
#     --out-dir step4_7a_slice_vit_test \
#     --num-slices 16 \
#     --image-size 224 \
#     --epochs 2 \
#     --n-splits 2 \
#     --batch-size 1 \
#     --device cuda
#
# Full-ish runs:
#
#   python step4_train_transformer_models.py \
#     --model seq_transformer \
#     --data-root /path/to/26S_AI536_NE450 \
#     --out-dir step4_7b_seq_transformer_outputs \
#     --seq-len 256 \
#     --embed-dim 128 \
#     --num-heads 4 \
#     --num-layers 4 \
#     --batch-size 16 \
#     --eval-batch-size 32 \
#     --epochs 100 \
#     --patience 20 \
#     --augment \
#     --device cuda
#
#   python step4_train_transformer_models.py \
#     --model slice_vit \
#     --vit-name deit_tiny_patch16_224 \
#     --data-root /path/to/26S_AI536_NE450 \
#     --out-dir step4_7a_deit_tiny_outputs \
#     --num-slices 32 \
#     --image-size 224 \
#     --batch-size 1 \
#     --eval-batch-size 2 \
#     --epochs 60 \
#     --patience 12 \
#     --augment \
#     --device cuda

import argparse
import csv
import json
import math
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

try:
    import timm
except Exception:
    timm = None


EPS = 1e-8


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# .mha reading and input conversion
# -----------------------------------------------------------------------------
def read_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return arr.astype(np.uint8, copy=False)


def uniform_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("num_frames must be positive.")
    if k <= 1:
        return np.array([n // 2], dtype=np.int64)
    return np.linspace(0, n - 1, k).round().astype(np.int64)


def random_jitter_indices(base_idx: np.ndarray, n: int, jitter: int) -> np.ndarray:
    if jitter <= 0:
        return base_idx
    offsets = np.random.randint(-jitter, jitter + 1, size=len(base_idx))
    return np.clip(base_idx + offsets, 0, n - 1).astype(np.int64)


def sample_vit_slices(
    arr: np.ndarray,
    num_slices: int,
    image_size: int,
    is_train: bool,
    slice_jitter: int,
    augment: bool,
) -> torch.Tensor:
    """
    Return tensor with shape (K, 3, image_size, image_size).

    channel 0 = background, mask == 0
    channel 1 = lumen,      mask == 1
    channel 2 = plaque,     mask == 2

    For ViT/DeiT, 3 channels are convenient because timm models expect RGB-like input.
    We use one-hot mask channels instead of natural-image RGB.
    """
    n = arr.shape[0]
    idx = uniform_indices(n, num_slices)
    if is_train:
        idx = random_jitter_indices(idx, n, slice_jitter)

    sampled = arr[idx]  # (K,H,W)
    x = np.stack([(sampled == 0), (sampled == 1), (sampled == 2)], axis=1).astype(np.float32)
    x = torch.from_numpy(x)  # (K,3,H,W)

    if image_size != sampled.shape[-1]:
        x = F.interpolate(x, size=(image_size, image_size), mode="nearest")

    if is_train and augment:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        if random.random() < 0.5:
            k_rot = random.randint(0, 3)
            x = torch.rot90(x, k=k_rot, dims=[-2, -1])

    return x.contiguous()


def compute_slice_features(arr: np.ndarray) -> np.ndarray:
    """
    Compute per-slice sequence features.
    Return shape: (num_frames, feature_dim).

    Features:
      0 lumen_area
      1 plaque_area
      2 vessel_area
      3 plaque_burden
      4 lumen_ratio
      5 plaque_to_lumen
      6 delta_lumen_area
      7 delta_plaque_burden
    """
    lumen = arr == 1
    plaque = arr == 2
    vessel = lumen | plaque

    lumen_area = lumen.sum(axis=(1, 2)).astype(np.float32)
    plaque_area = plaque.sum(axis=(1, 2)).astype(np.float32)
    vessel_area = vessel.sum(axis=(1, 2)).astype(np.float32)

    plaque_burden = plaque_area / (vessel_area + EPS)
    lumen_ratio = lumen_area / (vessel_area + EPS)
    plaque_to_lumen = plaque_area / (lumen_area + EPS)

    # Normalized deltas can help longitudinal pattern modeling.
    d_lumen = np.zeros_like(lumen_area)
    d_burden = np.zeros_like(plaque_burden)
    if len(lumen_area) > 1:
        d_lumen[1:] = np.diff(lumen_area)
        d_burden[1:] = np.diff(plaque_burden)

    seq = np.stack(
        [
            lumen_area,
            plaque_area,
            vessel_area,
            plaque_burden,
            lumen_ratio,
            plaque_to_lumen,
            d_lumen,
            d_burden,
        ],
        axis=1,
    ).astype(np.float32)

    return seq


def resample_sequence(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Linear resample sequence along length dimension.
    Input:  (N,C)
    Output: (seq_len,C)
    """
    n, c = seq.shape
    if n == seq_len:
        return seq.astype(np.float32, copy=False)

    old_x = np.linspace(0.0, 1.0, n)
    new_x = np.linspace(0.0, 1.0, seq_len)

    out = np.zeros((seq_len, c), dtype=np.float32)
    for j in range(c):
        out[:, j] = np.interp(new_x, old_x, seq[:, j]).astype(np.float32)

    return out


def normalize_sequence_features(seq: np.ndarray) -> np.ndarray:
    """
    Per-sample robust normalization for sequence features.
    This avoids needing global mean/std and makes the model more stable.
    """
    seq = seq.astype(np.float32, copy=True)

    # Area-like features: log1p then z-score per sample.
    for j in [0, 1, 2, 5, 6]:
        x = seq[:, j]
        if j in [0, 1, 2, 5]:
            x = np.log1p(np.maximum(x, 0.0))
        # signed delta can be negative
        mean = float(np.mean(x))
        std = float(np.std(x) + EPS)
        seq[:, j] = (x - mean) / std

    # Ratio/burden features already scale roughly 0-1, but standardize lightly.
    for j in [3, 4, 7]:
        x = seq[:, j]
        mean = float(np.mean(x))
        std = float(np.std(x) + EPS)
        seq[:, j] = (x - mean) / std

    return seq


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
class IVUSSliceViTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        num_slices: int,
        image_size: int,
        is_train: bool,
        slice_jitter: int,
        augment: bool,
        has_label: bool,
    ):
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
        path = self.data_dir / f"{serial_no}.mha"

        arr = read_mha(path)
        x = sample_vit_slices(
            arr=arr,
            num_slices=self.num_slices,
            image_size=self.image_size,
            is_train=self.is_train,
            slice_jitter=self.slice_jitter,
            augment=self.augment,
        )

        if self.has_label:
            ffr = float(row["FFR"])
            risk = get_risk_target_from_ffr(ffr)
            ffr_class = get_ffr_class_from_ffr(ffr)
            return {
                "x": x,
                "risk": torch.tensor(risk, dtype=torch.float32),
                "ffr_class": torch.tensor(ffr_class, dtype=torch.long),
                "serial_no": serial_no,
            }

        return {
            "x": x,
            "serial_no": serial_no,
        }


class IVUSSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        seq_len: int,
        is_train: bool,
        augment: bool,
        has_label: bool,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.is_train = is_train
        self.augment = augment
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        serial_no = str(row["serial_no"])
        path = self.data_dir / f"{serial_no}.mha"

        arr = read_mha(path)
        seq = compute_slice_features(arr)
        seq = resample_sequence(seq, self.seq_len)
        seq = normalize_sequence_features(seq)

        if self.is_train and self.augment:
            # Mild sequence augmentation.
            # 1) Random reversal: vessel direction may be arbitrary.
            if random.random() < 0.5:
                seq = seq[::-1].copy()

            # 2) Mild Gaussian noise on normalized features.
            if random.random() < 0.5:
                seq = seq + np.random.normal(0.0, 0.02, size=seq.shape).astype(np.float32)

        x = torch.tensor(seq, dtype=torch.float32)  # (L,C)

        if self.has_label:
            ffr = float(row["FFR"])
            risk = get_risk_target_from_ffr(ffr)
            ffr_class = get_ffr_class_from_ffr(ffr)
            return {
                "x": x,
                "risk": torch.tensor(risk, dtype=torch.float32),
                "ffr_class": torch.tensor(ffr_class, dtype=torch.long),
                "serial_no": serial_no,
            }

        return {
            "x": x,
            "serial_no": serial_no,
        }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SliceViTClassifier(nn.Module):
    """
    Exp 7-A:
      K sampled mask slices
      -> timm ViT/DeiT encoder per slice
      -> mean/max pooling over slices
      -> classifier

    Input: (B,K,3,H,W)
    """
    def __init__(
        self,
        vit_name: str,
        pretrained: bool,
        embed_dim: int,
        dropout: float,
        pooling: str,
    ):
        super().__init__()

        if timm is None:
            raise ImportError("timm is not installed. Install timm or use --model seq_transformer.")

        self.pooling = pooling

        # num_classes=0 makes many timm models return feature vector.
        self.backbone = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="token",
        )

        backbone_dim = self.backbone.num_features

        if pooling == "mean":
            pooled_dim = backbone_dim
        elif pooling == "max":
            pooled_dim = backbone_dim
        elif pooling == "mean_max":
            pooled_dim = backbone_dim * 2
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        b, k, c, h, w = x.shape
        x = x.view(b * k, c, h, w)
        emb = self.backbone(x)  # (B*K,E)
        emb = emb.view(b, k, -1)

        if self.pooling == "mean":
            pooled = emb.mean(dim=1)
        elif self.pooling == "max":
            pooled = emb.max(dim=1).values
        else:
            pooled = torch.cat([emb.mean(dim=1), emb.max(dim=1).values], dim=1)

        return self.classifier(pooled).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (B,L,E)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SequenceTransformerClassifier(nn.Module):
    """
    Exp 7-B:
      slice-wise sequence features
      -> linear projection
      -> positional encoding
      -> Transformer Encoder
      -> mean/max/CLS pooling
      -> classifier

    Input: (B,L,C)
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        pooling: str,
        use_cls_token: bool,
        max_len: int,
    ):
        super().__init__()
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Linear(input_dim, embed_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos = PositionalEncoding(embed_dim=embed_dim, max_len=max_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if pooling == "cls":
            pooled_dim = embed_dim
        elif pooling == "mean":
            pooled_dim = embed_dim
        elif pooling == "max":
            pooled_dim = embed_dim
        elif pooling == "mean_max":
            pooled_dim = embed_dim * 2
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B,L,C)
        h = self.input_proj(x)

        if self.use_cls_token:
            cls = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)

        h = self.pos(h)
        h = self.encoder(h)

        if self.pooling == "cls":
            if not self.use_cls_token:
                pooled = h[:, 0]
            else:
                pooled = h[:, 0]
        elif self.pooling == "mean":
            if self.use_cls_token:
                pooled = h[:, 1:].mean(dim=1)
            else:
                pooled = h.mean(dim=1)
        elif self.pooling == "max":
            if self.use_cls_token:
                pooled = h[:, 1:].max(dim=1).values
            else:
                pooled = h.max(dim=1).values
        else:
            if self.use_cls_token:
                body = h[:, 1:]
            else:
                body = h
            pooled = torch.cat([body.mean(dim=1), body.max(dim=1).values], dim=1)

        return self.classifier(pooled).squeeze(-1)


def build_model(args):
    if args.model == "slice_vit":
        return SliceViTClassifier(
            vit_name=args.vit_name,
            pretrained=args.pretrained,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            pooling=args.pooling,
        )

    if args.model == "seq_transformer":
        return SequenceTransformerClassifier(
            input_dim=8,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            pooling=args.seq_pooling,
            use_cls_token=args.use_cls_token,
            max_len=args.seq_len,
        )

    raise ValueError(f"Unknown model: {args.model}")


# -----------------------------------------------------------------------------
# Loader / train / predict
# -----------------------------------------------------------------------------
def make_loader(df, data_dir, args, is_train, has_label):
    if args.model == "slice_vit":
        ds = IVUSSliceViTDataset(
            df=df,
            data_dir=data_dir,
            num_slices=args.num_slices,
            image_size=args.image_size,
            is_train=is_train,
            slice_jitter=args.slice_jitter,
            augment=args.augment and is_train,
            has_label=has_label,
        )
    elif args.model == "seq_transformer":
        ds = IVUSSequenceDataset(
            df=df,
            data_dir=data_dir,
            seq_len=args.seq_len,
            is_train=is_train,
            augment=args.augment and is_train,
            has_label=has_label,
        )
    else:
        raise ValueError(args.model)

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
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)

            serials.extend(batch["serial_no"])

            if "ffr_class" in batch:
                true_classes.append(batch["ffr_class"].detach().cpu().numpy())

    probs = np.concatenate(probs, axis=0)

    if len(true_classes) > 0:
        true_classes = np.concatenate(true_classes, axis=0)
    else:
        true_classes = None

    return serials, probs, true_classes


def train_one_fold(fold, train_df, val_df, train_dir, args, device, out_dir):
    train_loader = make_loader(train_df, train_dir, args, is_train=True, has_label=True)
    val_loader = make_loader(val_df, train_dir, args, is_train=False, has_label=True)

    model = build_model(args).to(device)

    # Optional backbone freeze for ViT warmup.
    if args.model == "slice_vit" and args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False

    y_train_risk = train_df["risk"].astype(int).values
    pos = float((y_train_risk == 1).sum())
    neg = float((y_train_risk == 0).sum())
    pos_weight_value = neg / max(pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
        # Unfreeze ViT backbone after warmup.
        if args.model == "slice_vit" and epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - epoch + 1)
                )

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

    ckpt_path = out_dir / f"{args.model}_fold{fold}.safetensors"
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
    df = pd.DataFrame({
        "serial_no": serials,
        "ffr_class": pred_class.astype(int),
    })
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


def write_summary(summary_df, out_path):
    lines = []
    lines.append("Step 4 Exp 7 Transformer Model Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[OOF metrics]")
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("[Reference feature-based model]")
    lines.append("Step 3-6 multi-seed MLP ensemble:")
    lines.append("  accuracy = 0.7505")
    lines.append("  balanced_accuracy = 0.7548")
    lines.append("  roc_auc = 0.8048")
    lines.append("")
    lines.append("[Interpretation]")
    lines.append("- seq_transformer is expected to be much faster and often more stable than slice_vit.")
    lines.append("- slice_vit is a direct image-transformer baseline but may overfit because data has only 1058 volumes.")
    lines.append("- If transformer ROC-AUC is lower than 0.80, feature-based MLP remains stronger.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["slice_vit", "seq_transformer"])

    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--train-dir", type=str, default="train")
    parser.add_argument("--test-dir", type=str, default="test_public")
    parser.add_argument("--labels", type=str, default=None)

    parser.add_argument("--out-dir", type=str, default="step4_transformer_outputs")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Exp 7-A: ViT/DeiT
    parser.add_argument("--vit-name", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-slices", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--slice-jitter", type=int, default=8)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)

    # Exp 7-B: sequence transformer
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--seq-pooling", type=str, default="mean_max", choices=["cls", "mean", "max", "mean_max"])
    parser.add_argument("--use-cls-token", action="store_true")

    # Shared model settings
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "max", "mean_max"])

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--early-metric", type=str, default="roc_auc", choices=["roc_auc", "balanced_accuracy", "f1_macro"])
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.model == "slice_vit" and timm is None:
        raise ImportError("timm is required for --model slice_vit.")

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

    print(f"[INFO] model = {args.model}")
    print(f"[INFO] device = {device}")
    print(f"[INFO] train_dir = {train_dir}")
    print(f"[INFO] test_dir = {test_dir}")
    print(f"[INFO] labels_path = {labels_path}")

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
    print(f"[SAVE] {out_dir / 'fold_assignments.csv'}")

    metadata = vars(args).copy()
    metadata.update({
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "labels_path": str(labels_path),
        "target": "risk=1 if FFR<0.8 else 0",
        "slice_transformer_note": "slice_vit uses one-hot 3-channel mask slices: background/lumen/plaque.",
        "sequence_transformer_note": "seq_transformer uses per-slice area/burden sequence features computed from mask.",
    })
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

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

        result = train_one_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            train_dir=train_dir,
            args=args,
            device=device,
            out_dir=out_dir,
        )

        fold_results.append(result)

        val_serials = np.array(result["serials_val"])
        val_index_map = {s: i for i, s in enumerate(oof_serials)}
        for s, p in zip(val_serials, result["p_val"]):
            oof_p[val_index_map[s]] = p

        metrics_05 = compute_metrics(result["y_val_ffr_class"], result["p_val"], threshold=0.5)
        metrics_05.update({
            "model": args.model,
            "fold": fold,
            "threshold_type": "fixed_0.5",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
        })
        fold_rows.append(metrics_05)

        thr_bal, metrics_bal = tune_threshold(result["y_val_ffr_class"], result["p_val"], "balanced_accuracy")
        metrics_bal.update({
            "model": args.model,
            "fold": fold,
            "threshold_type": "fold_tuned_balanced_accuracy",
            "best_epoch": result["best_epoch"],
            "best_score": result["best_score"],
        })
        fold_rows.append(metrics_bal)

    # OOF summary
    summary_rows = []
    y_true = y_ffr_class_all

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
            "model": args.model,
            "threshold_type": name,
            "selected_threshold": float(selected_thr),
            "vit_name": args.vit_name if args.model == "slice_vit" else "",
            "num_slices": args.num_slices if args.model == "slice_vit" else np.nan,
            "seq_len": args.seq_len if args.model == "seq_transformer" else np.nan,
            "image_size": args.image_size if args.model == "slice_vit" else np.nan,
            "embed_dim": args.embed_dim,
            "dropout": args.dropout,
            "pooling": args.pooling if args.model == "slice_vit" else args.seq_pooling,
        })
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    print(f"[SAVE] {out_dir / 'cv_results_summary.csv'}")

    pd.DataFrame(fold_rows).to_csv(out_dir / "per_fold_metrics.csv", index=False)
    print(f"[SAVE] {out_dir / 'per_fold_metrics.csv'}")

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
    print(f"[SAVE] {out_dir / 'oof_predictions.csv'}")

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

    # Public test prediction
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
            print(f"[SAVE] {out_dir / 'test_public_probabilities.csv'}")

            save_quickcheck_csv(test_serials, p_test_mean, 0.5, out_dir / f"quickcheck_{args.model}_thr05.csv")
            save_quickcheck_csv(test_serials, p_test_mean, thr_acc, out_dir / f"quickcheck_{args.model}_acc.csv")
            save_quickcheck_csv(test_serials, p_test_mean, thr_bal, out_dir / f"quickcheck_{args.model}_bal.csv")
            save_quickcheck_csv(test_serials, p_test_mean, thr_f1, out_dir / f"quickcheck_{args.model}_f1.csv")

    print("")
    print("[DONE] Step 4 Exp 7 Transformer training completed.")
    print(f"[CHECK] {out_dir / 'model_comparison.txt'}")


if __name__ == "__main__":
    main()