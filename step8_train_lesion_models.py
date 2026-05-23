#!/usr/bin/env python3
# step8_train_lesion_models.py
#
# Step 8: lesion-aware experiments.
# Supports three modes:
#   1) lesion_cnn
#   2) fusion_cnn_tabular
#   3) ffr_regression
# Saves epoch-by-epoch histories and optional learning-curve PNGs.

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
from safetensors.torch import save_file

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

EPS = 1e-8


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_mha(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.uint8, copy=False)


def safe_div(a, b):
    return a / (b + EPS)


def ffr_to_class(ffr):
    return int(ffr >= 0.8)


def ffr_to_risk(ffr):
    return int(ffr < 0.8)


def risk_prob_to_ffr_class(p_risk, threshold):
    return np.where(p_risk >= threshold, 0, 1).astype(int)


def compute_classification_metrics(y_true_ffr_class, p_risk, threshold):
    y_pred = risk_prob_to_ffr_class(p_risk, threshold)
    y_true_ffr_class = np.asarray(y_true_ffr_class).astype(int)
    y_true_risk = 1 - y_true_ffr_class
    cm = confusion_matrix(y_true_ffr_class, y_pred, labels=[0, 1])
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_ffr_class, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_ffr_class, y_pred)),
        "f1_macro": float(f1_score(y_true_ffr_class, y_pred, average="macro", zero_division=0)),
        "f1_class0": float(f1_score(y_true_ffr_class, y_pred, pos_label=0, zero_division=0)),
        "f1_class1": float(f1_score(y_true_ffr_class, y_pred, pos_label=1, zero_division=0)),
        "precision_class0": float(precision_score(y_true_ffr_class, y_pred, pos_label=0, zero_division=0)),
        "recall_class0": float(recall_score(y_true_ffr_class, y_pred, pos_label=0, zero_division=0)),
        "precision_class1": float(precision_score(y_true_ffr_class, y_pred, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y_true_ffr_class, y_pred, pos_label=1, zero_division=0)),
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


def tune_risk_threshold(y_true_ffr_class, p_risk, metric_name):
    thresholds = np.round(np.arange(0.02, 0.981, 0.01), 4)
    best, best_value, best_thr = None, -1e18, 0.5
    for thr in thresholds:
        m = compute_classification_metrics(y_true_ffr_class, p_risk, thr)
        value = m.get(metric_name, -1e18)
        if np.isfinite(value) and value > best_value:
            best_value, best, best_thr = value, m, float(thr)
    return best_thr, best


def compute_regression_metrics(y_true_ffr, pred_ffr, threshold_ffr):
    y_true_ffr = np.asarray(y_true_ffr).astype(float)
    pred_ffr = np.asarray(pred_ffr).astype(float)
    y_true_class = (y_true_ffr >= 0.8).astype(int)
    y_pred_class = (pred_ffr >= threshold_ffr).astype(int)
    y_true_risk = 1 - y_true_class
    risk_score = -pred_ffr
    cm = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1])
    out = {
        "ffr_threshold": float(threshold_ffr),
        "mae": float(np.mean(np.abs(y_true_ffr - pred_ffr))),
        "rmse": float(np.sqrt(np.mean((y_true_ffr - pred_ffr) ** 2))),
        "accuracy": float(accuracy_score(y_true_class, y_pred_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_class, y_pred_class)),
        "f1_macro": float(f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)),
        "recall_class0": float(recall_score(y_true_class, y_pred_class, pos_label=0, zero_division=0)),
        "recall_class1": float(recall_score(y_true_class, y_pred_class, pos_label=1, zero_division=0)),
        "cm_true0_pred0": int(cm[0, 0]),
        "cm_true0_pred1": int(cm[0, 1]),
        "cm_true1_pred0": int(cm[1, 0]),
        "cm_true1_pred1": int(cm[1, 1]),
    }
    try:
        out["roc_auc_risk"] = float(roc_auc_score(y_true_risk, risk_score))
    except Exception:
        out["roc_auc_risk"] = np.nan
    return out


def tune_ffr_threshold(y_true_ffr, pred_ffr, metric_name):
    thresholds = np.round(np.arange(0.70, 0.901, 0.005), 4)
    best, best_value, best_thr = None, -1e18, 0.8
    for thr in thresholds:
        m = compute_regression_metrics(y_true_ffr, pred_ffr, thr)
        value = m.get(metric_name, -1e18)
        if np.isfinite(value) and value > best_value:
            best_value, best, best_thr = value, m, float(thr)
    return best_thr, best


def choose_lesion_centers(arr, center_mode="hybrid", multi_centers=3):
    lumen = arr == 1
    plaque = arr == 2
    vessel = lumen | plaque
    lumen_area = lumen.sum(axis=(1, 2)).astype(np.float64)
    plaque_area = plaque.sum(axis=(1, 2)).astype(np.float64)
    vessel_area = vessel.sum(axis=(1, 2)).astype(np.float64)
    nonempty = vessel_area > 0
    if nonempty.sum() == 0:
        return [arr.shape[0] // 2]
    idxs = np.where(nonempty)[0]
    la, pa, va = lumen_area[nonempty], plaque_area[nonempty], vessel_area[nonempty]
    burden = safe_div(pa, va)
    ratio = safe_div(pa, la)
    ref = max(np.mean(np.sort(la)[-max(1, int(math.ceil(len(la) * 0.2))):]), EPS)
    stenosis = np.clip(1.0 - safe_div(la, ref), 0.0, 1.0)

    def z(x):
        s = float(np.std(x))
        return np.zeros_like(x) if s < 1e-8 else (x - np.mean(x)) / s

    candidates = []
    if center_mode in {"min_lumen", "hybrid", "multi"}:
        candidates.append(int(idxs[int(np.argmin(la))]))
    if center_mode in {"max_burden", "hybrid", "multi"}:
        candidates.append(int(idxs[int(np.argmax(burden))]))
    if center_mode in {"max_ratio", "hybrid", "multi"}:
        candidates.append(int(idxs[int(np.argmax(ratio))]))
    if center_mode in {"hybrid", "multi"}:
        score = z(stenosis) + z(burden) + z(np.log1p(ratio))
        order = np.argsort(-score)
        for j in order[:max(1, multi_centers)]:
            candidates.append(int(idxs[int(j)]))
    if not candidates:
        candidates.append(int(idxs[int(np.argmin(la))]))
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    if center_mode == "multi":
        return unique[:max(1, multi_centers)]
    return [unique[0]]


def sample_window_indices(n, center, k, jitter=0, is_train=False):
    if is_train and jitter > 0:
        center = int(np.clip(center + np.random.randint(-jitter, jitter + 1), 0, n - 1))
    start = center - k // 2
    idx = np.arange(start, start + k)
    return np.clip(idx, 0, n - 1).astype(np.int64)


def sample_lesion_slices(arr, center_mode, window_slices, image_size, is_train, slice_jitter, augment, multi_centers):
    centers = choose_lesion_centers(arr, center_mode=center_mode, multi_centers=multi_centers)
    xs = []
    for c in centers:
        idx = sample_window_indices(arr.shape[0], c, window_slices, jitter=slice_jitter, is_train=is_train)
        sampled = arr[idx]
        x = np.stack([(sampled == 1), (sampled == 2)], axis=1).astype(np.float32)
        x = torch.from_numpy(x)
        if image_size != sampled.shape[-1]:
            x = F.interpolate(x, size=(image_size, image_size), mode="nearest")
        xs.append(x)
    x = torch.cat(xs, dim=0)
    if is_train and augment:
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        if random.random() < 0.5:
            x = torch.rot90(x, k=random.randint(0, 3), dims=[-2, -1])
    return x.contiguous()


def get_feature_columns(df):
    exclude = {"serial_no", "FFR", "ffr_class", "risk"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_feature_maps(train_features_path, test_features_path):
    train_feat = pd.read_csv(train_features_path).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    train_feat["serial_no"] = train_feat["serial_no"].astype(str)
    feature_cols = get_feature_columns(train_feat)
    train_map = {row["serial_no"]: row[feature_cols].astype(np.float32).values for _, row in train_feat.iterrows()}
    test_feat = pd.read_csv(test_features_path).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    test_feat["serial_no"] = test_feat["serial_no"].astype(str)
    test_map = {row["serial_no"]: row[feature_cols].astype(np.float32).values for _, row in test_feat.iterrows()}
    return feature_cols, train_map, test_map


class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, args, is_train, has_label, feature_map=None, scaler=None):
        self.df = df.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.args = args
        self.is_train = is_train
        self.has_label = has_label
        self.feature_map = feature_map
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        serial_no = str(row["serial_no"])
        out = {"serial_no": serial_no}
        if self.args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
            arr = read_mha(self.data_dir / f"{serial_no}.mha")
            out["x"] = sample_lesion_slices(arr, self.args.center_mode, self.args.window_slices, self.args.image_size, self.is_train, self.args.slice_jitter, self.args.augment, self.args.multi_centers)
        if self.args.mode in {"fusion_cnn_tabular", "ffr_regression"}:
            feat = np.asarray(self.feature_map[serial_no], dtype=np.float32)
            if self.scaler is not None:
                mean, scale = self.scaler
                feat = (feat - mean) / scale
            out["tab"] = torch.tensor(feat, dtype=torch.float32)
        if self.has_label:
            ffr = float(row["FFR"])
            out["ffr"] = torch.tensor(ffr, dtype=torch.float32)
            out["risk"] = torch.tensor(ffr_to_risk(ffr), dtype=torch.float32)
            out["ffr_class"] = torch.tensor(ffr_to_class(ffr), dtype=torch.long)
        return out


class ConvBNAct2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))
    def forward(self, x):
        return self.block(x)


class SliceCNNEncoder(nn.Module):
    def __init__(self, in_ch=2, base_ch=16, embed_dim=128, dropout=0.15):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct2d(in_ch, base_ch), ConvBNAct2d(base_ch, base_ch), nn.MaxPool2d(2),
            ConvBNAct2d(base_ch, base_ch * 2), ConvBNAct2d(base_ch * 2, base_ch * 2), nn.MaxPool2d(2),
            ConvBNAct2d(base_ch * 2, base_ch * 4), ConvBNAct2d(base_ch * 4, base_ch * 4), nn.MaxPool2d(2),
            ConvBNAct2d(base_ch * 4, base_ch * 8), nn.Dropout2d(dropout),
        )
        self.proj = nn.Linear(base_ch * 8, embed_dim)
    def forward(self, x):
        h = self.features(x)
        return self.proj(F.adaptive_avg_pool2d(h, 1).flatten(1))


class AttentionPool(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        attn_dim = max(64, embed_dim // 2)
        self.v = nn.Linear(embed_dim, attn_dim)
        self.u = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, emb):
        h = torch.tanh(self.v(emb)) * torch.sigmoid(self.u(emb))
        scores = self.w(self.drop(h)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(emb * weights.unsqueeze(-1), dim=1)


class LesionCNNClassifier(nn.Module):
    def __init__(self, base_ch=16, embed_dim=128, dropout=0.25, pooling="attention"):
        super().__init__()
        self.encoder = SliceCNNEncoder(2, base_ch, embed_dim, dropout)
        self.pooling = pooling
        if pooling == "attention":
            self.attn_pool = AttentionPool(embed_dim, dropout)
            pooled_dim = embed_dim
        elif pooling == "meanmax":
            pooled_dim = embed_dim * 2
        else:
            raise ValueError(pooling)
        self.classifier = nn.Sequential(nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, embed_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1))
    def pooled_embedding(self, x):
        b, k, c, h, w = x.shape
        emb = self.encoder(x.view(b * k, c, h, w)).view(b, k, -1)
        if self.pooling == "attention":
            return self.attn_pool(emb)
        return torch.cat([emb.mean(dim=1), emb.max(dim=1).values], dim=1)
    def forward(self, x):
        return self.classifier(self.pooled_embedding(x)).squeeze(-1)


class FusionCNNTabularClassifier(nn.Module):
    def __init__(self, tab_dim, base_ch=16, embed_dim=128, tab_embed_dim=128, dropout=0.25, pooling="attention"):
        super().__init__()
        self.img = LesionCNNClassifier(base_ch, embed_dim, dropout, pooling)
        img_dim = embed_dim if pooling == "attention" else embed_dim * 2
        self.tab_mlp = nn.Sequential(nn.LayerNorm(tab_dim), nn.Linear(tab_dim, tab_embed_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(tab_embed_dim, tab_embed_dim), nn.SiLU())
        self.classifier = nn.Sequential(nn.LayerNorm(img_dim + tab_embed_dim), nn.Linear(img_dim + tab_embed_dim, embed_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1))
    def forward(self, x, tab):
        return self.classifier(torch.cat([self.img.pooled_embedding(x), self.tab_mlp(tab)], dim=1)).squeeze(-1)


class TabularFFRRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.25):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, tab):
        return self.net(tab).squeeze(-1)


def make_loader(df, data_dir, args, is_train, has_label, feature_map=None, scaler=None):
    ds = LesionDataset(df, data_dir, args, is_train, has_label, feature_map, scaler)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size if is_train else args.eval_batch_size, shuffle=is_train, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)


def build_model(args, tab_dim=None):
    if args.mode == "lesion_cnn":
        return LesionCNNClassifier(args.base_ch, args.embed_dim, args.dropout, args.pooling)
    if args.mode == "fusion_cnn_tabular":
        return FusionCNNTabularClassifier(tab_dim, args.base_ch, args.embed_dim, args.tab_embed_dim, args.dropout, args.pooling)
    if args.mode == "ffr_regression":
        return TabularFFRRegressor(tab_dim, args.hidden_dims, args.dropout)
    raise ValueError(args.mode)


def predict_classifier(model, loader, device, mode):
    model.eval(); serials=[]; probs=[]; y_classes=[]; y_ffr=[]
    with torch.no_grad():
        for batch in loader:
            if mode == "lesion_cnn":
                logits = model(batch["x"].to(device, non_blocking=True))
            else:
                logits = model(batch["x"].to(device, non_blocking=True), batch["tab"].to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            serials.extend(batch["serial_no"])
            if "ffr_class" in batch:
                y_classes.append(batch["ffr_class"].detach().cpu().numpy())
                y_ffr.append(batch["ffr"].detach().cpu().numpy())
    return serials, np.concatenate(probs), (np.concatenate(y_classes) if y_classes else None), (np.concatenate(y_ffr) if y_ffr else None)


def predict_regression(model, loader, device):
    model.eval(); serials=[]; preds=[]; y_classes=[]; y_ffr=[]
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["tab"].to(device, non_blocking=True)).detach().cpu().numpy()
            preds.append(pred); serials.extend(batch["serial_no"])
            if "ffr_class" in batch:
                y_classes.append(batch["ffr_class"].detach().cpu().numpy())
                y_ffr.append(batch["ffr"].detach().cpu().numpy())
    return serials, np.concatenate(preds), (np.concatenate(y_classes) if y_classes else None), (np.concatenate(y_ffr) if y_ffr else None)


def save_quickcheck(serials, classes, out_path):
    pd.DataFrame({"serial_no": [str(s) for s in serials], "ffr_class": np.asarray(classes).astype(int)}).to_csv(out_path, index=False)


def train_one_fold(seed, fold, train_df, val_df, train_dir, args, device, out_dir, feature_map_train=None, scaler=None, tab_dim=None):
    train_loader = make_loader(train_df, train_dir, args, True, True, feature_map_train, scaler)
    val_loader = make_loader(val_df, train_dir, args, False, True, feature_map_train, scaler)
    model = build_model(args, tab_dim=tab_dim).to(device)

    if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
        y_risk = train_df["risk"].astype(int).values
        pos, neg = float((y_risk == 1).sum()), float((y_risk == 0).sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    else:
        criterion = nn.SmoothL1Loss(beta=args.huber_beta)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs)) if args.scheduler == "cosine" else None

    best_score, best_state, best_epoch, patience_count = -1e18, None, -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]; start=time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "lesion_cnn":
                loss = criterion(model(batch["x"].to(device, non_blocking=True)), batch["risk"].to(device, non_blocking=True))
            elif args.mode == "fusion_cnn_tabular":
                loss = criterion(model(batch["x"].to(device, non_blocking=True), batch["tab"].to(device, non_blocking=True)), batch["risk"].to(device, non_blocking=True))
            else:
                loss = criterion(model(batch["tab"].to(device, non_blocking=True)), batch["ffr"].to(device, non_blocking=True))
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); losses.append(float(loss.detach().cpu().item()))
        if scheduler is not None:
            scheduler.step()

        if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
            _, p_val, y_val_class, _ = predict_classifier(model, val_loader, device, args.mode)
            m05 = compute_classification_metrics(y_val_class, p_val, 0.5)
            _, mb = tune_risk_threshold(y_val_class, p_val, "balanced_accuracy")
            _, ma = tune_risk_threshold(y_val_class, p_val, "accuracy")
            score = m05["roc_auc_risk"] if args.early_metric == "roc_auc" else mb["balanced_accuracy"]
            row = {"seed": seed, "fold": fold, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_roc_auc_risk": m05["roc_auc_risk"], "val_acc_05": m05["accuracy"], "val_bal_05": m05["balanced_accuracy"], "val_recall0_05": m05["recall_class0"], "val_best_bal": mb["balanced_accuracy"], "val_best_bal_thr": mb["threshold"], "val_best_acc": ma["accuracy"], "val_best_acc_thr": ma["threshold"], "epoch_seconds": time.time() - start}
        else:
            _, pred_val, _, y_val_ffr = predict_regression(model, val_loader, device)
            m08 = compute_regression_metrics(y_val_ffr, pred_val, 0.8)
            _, mb = tune_ffr_threshold(y_val_ffr, pred_val, "balanced_accuracy")
            _, ma = tune_ffr_threshold(y_val_ffr, pred_val, "accuracy")
            score = m08["roc_auc_risk"] if args.early_metric == "roc_auc" else (-m08["mae"] if args.early_metric == "mae" else mb["balanced_accuracy"])
            row = {"seed": seed, "fold": fold, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": m08["mae"], "val_rmse": m08["rmse"], "val_roc_auc_risk": m08["roc_auc_risk"], "val_acc_ffr080": m08["accuracy"], "val_bal_ffr080": m08["balanced_accuracy"], "val_best_bal": mb["balanced_accuracy"], "val_best_bal_ffr_thr": mb["ffr_threshold"], "val_best_acc": ma["accuracy"], "val_best_acc_ffr_thr": ma["ffr_threshold"], "epoch_seconds": time.time() - start}
        history.append(row)

        if np.isfinite(score) and score > best_score + args.min_delta:
            best_score, best_epoch = float(score), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % args.print_every == 0:
            if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
                print(f"[seed {seed} fold {fold}] epoch {epoch:03d} loss={np.mean(losses):.5f} auc={row['val_roc_auc_risk']:.4f} bal@0.5={row['val_bal_05']:.4f} best_bal={row['val_best_bal']:.4f} thr={row['val_best_bal_thr']:.2f}")
            else:
                print(f"[seed {seed} fold {fold}] epoch {epoch:03d} loss={np.mean(losses):.5f} mae={row['val_mae']:.4f} auc={row['val_roc_auc_risk']:.4f} bal@0.8={row['val_bal_ffr080']:.4f} best_bal={row['val_best_bal']:.4f}")
        if patience_count >= args.patience:
            print(f"[seed {seed} fold {fold}] early stopping epoch={epoch}, best_epoch={best_epoch}, best_score={best_score:.5f}")
            break

    model.load_state_dict(best_state)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / f"history_seed{seed}_fold{fold}.csv", index=False)
    tensors = {f"model.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    tensors["meta.seed"] = torch.tensor([seed], dtype=torch.int64)
    tensors["meta.fold"] = torch.tensor([fold], dtype=torch.int64)
    tensors["meta.best_epoch"] = torch.tensor([best_epoch], dtype=torch.int64)
    if scaler is not None:
        tensors["scaler.mean"] = torch.tensor(scaler[0], dtype=torch.float32)
        tensors["scaler.scale"] = torch.tensor(scaler[1], dtype=torch.float32)
    save_file(tensors, str(out_dir / f"{args.mode}_seed{seed}_fold{fold}.safetensors"))

    if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
        serials_val, score_val, y_val_class, y_val_ffr = predict_classifier(model, val_loader, device, args.mode)
    else:
        serials_val, score_val, y_val_class, y_val_ffr = predict_regression(model, val_loader, device)
    return {"model": model, "history": hist_df, "serials_val": serials_val, "score_val": score_val, "y_val_class": y_val_class, "y_val_ffr": y_val_ffr, "best_epoch": best_epoch, "best_score": best_score}


def make_learning_curve_summary(out_dir):
    rows = [pd.read_csv(p) for p in sorted(out_dir.glob("history_seed*_fold*.csv"))]
    if not rows:
        return
    all_hist = pd.concat(rows, ignore_index=True)
    all_hist.to_csv(out_dir / "learning_curve_all_epochs.csv", index=False)
    numeric_cols = [c for c in all_hist.columns if c not in {"seed", "fold"} and pd.api.types.is_numeric_dtype(all_hist[c])]
    summary = all_hist.groupby("epoch")[numeric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.reset_index().to_csv(out_dir / "learning_curve_summary.csv", index=False)


def save_plots(out_dir, mode):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable; skipping plots."); return
    path = out_dir / "learning_curve_all_epochs.csv"
    if not path.exists(): return
    df = pd.read_csv(path)
    metrics = ["train_loss", "val_roc_auc_risk", "val_bal_05", "val_best_bal", "val_best_acc"] if mode != "ffr_regression" else ["train_loss", "val_mae", "val_roc_auc_risk", "val_bal_ffr080", "val_best_bal", "val_best_acc"]
    for metric in metrics:
        if metric not in df.columns: continue
        plt.figure()
        for (_, _), g in df.groupby(["seed", "fold"]):
            plt.plot(g["epoch"], g[metric], alpha=0.35)
        mean = df.groupby("epoch")[metric].mean()
        plt.plot(mean.index, mean.values, linewidth=3)
        plt.xlabel("epoch"); plt.ylabel(metric); plt.title(metric); plt.tight_layout()
        plt.savefig(out_dir / f"curve_{metric}.png", dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["lesion_cnn", "fusion_cnn_tabular", "ffr_regression"])
    parser.add_argument("--data-root", default="./26S_AI536_NE450")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--train-features", default=None)
    parser.add_argument("--test-features", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--center-mode", default="hybrid", choices=["min_lumen", "max_burden", "max_ratio", "hybrid", "multi"])
    parser.add_argument("--multi-centers", type=int, default=3)
    parser.add_argument("--window-slices", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--slice-jitter", type=int, default=8)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--tab-embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pooling", default="attention", choices=["attention", "meanmax"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    parser.add_argument("--early-metric", default="roc_auc", choices=["roc_auc", "balanced_accuracy", "mae"])
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    train_dir = Path(args.train_dir) if args.train_dir else data_root / "train"
    test_dir = Path(args.test_dir) if args.test_dir else data_root / "test_public"
    labels_path = Path(args.labels) if args.labels else train_dir / "labels.csv"
    if args.mode in {"fusion_cnn_tabular", "ffr_regression"} and (not args.train_features or not args.test_features):
        raise ValueError("--train-features and --test-features are required for fusion_cnn_tabular and ffr_regression.")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto": device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_df = pd.read_csv(labels_path)
    labels_df["serial_no"] = labels_df["serial_no"].astype(str)
    labels_df["ffr_class"] = (labels_df["FFR"] >= 0.8).astype(int)
    labels_df["risk"] = (labels_df["FFR"] < 0.8).astype(int)
    if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
        exists = labels_df["serial_no"].apply(lambda s: (train_dir / f"{s}.mha").exists())
        labels_df = labels_df[exists].reset_index(drop=True)

    feature_cols = None; train_feature_map = None; test_feature_map = None; tab_dim = None
    if args.mode in {"fusion_cnn_tabular", "ffr_regression"}:
        feature_cols, train_feature_map, test_feature_map = build_feature_maps(args.train_features, args.test_features)
        tab_dim = len(feature_cols); print(f"[INFO] tabular features: {tab_dim}")
    metadata = vars(args).copy(); metadata["feature_cols"] = feature_cols
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    y_all = labels_df["ffr_class"].astype(int).values
    y_ffr_all = labels_df["FFR"].astype(float).values
    oof_score_sum = np.zeros(len(labels_df), dtype=np.float64); oof_count = np.zeros(len(labels_df), dtype=np.float64)
    oof_serials = np.asarray(labels_df["serial_no"].astype(str).values)
    all_results=[]; fold_metrics=[]

    for seed in args.seeds:
        seed_everything(seed)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(labels_df, y_all)):
            print("\n" + "="*80 + f"\n[TRAIN] mode={args.mode} seed={seed} fold={fold}")
            train_df, val_df = labels_df.iloc[tr_idx].reset_index(drop=True), labels_df.iloc[va_idx].reset_index(drop=True)
            scaler = None
            if args.mode in {"fusion_cnn_tabular", "ffr_regression"}:
                x_train = np.stack([train_feature_map[s] for s in train_df["serial_no"].astype(str)], axis=0).astype(np.float32)
                mean, scale = x_train.mean(axis=0).astype(np.float32), x_train.std(axis=0).astype(np.float32)
                scale = np.where(np.abs(scale) < 1e-8, 1.0, scale).astype(np.float32)
                scaler = (mean, scale)
            result = train_one_fold(seed, fold, train_df, val_df, train_dir, args, device, out_dir, train_feature_map, scaler, tab_dim)
            all_results.append((seed, fold, result, scaler))
            val_map = {s: i for i, s in enumerate(oof_serials)}
            for s, score in zip(result["serials_val"], result["score_val"]):
                idx = val_map[str(s)]; oof_score_sum[idx] += float(score); oof_count[idx] += 1
            if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
                m05 = compute_classification_metrics(result["y_val_class"], result["score_val"], 0.5); _, mb = tune_risk_threshold(result["y_val_class"], result["score_val"], "balanced_accuracy")
                m05.update({"seed": seed, "fold": fold, "threshold_type": "fixed_0.5", "best_epoch": result["best_epoch"], "best_score": result["best_score"]}); mb.update({"seed": seed, "fold": fold, "threshold_type": "fold_tuned_balanced", "best_epoch": result["best_epoch"], "best_score": result["best_score"]}); fold_metrics += [m05, mb]
            else:
                m08 = compute_regression_metrics(result["y_val_ffr"], result["score_val"], 0.8); _, mb = tune_ffr_threshold(result["y_val_ffr"], result["score_val"], "balanced_accuracy")
                m08.update({"seed": seed, "fold": fold, "threshold_type": "fixed_ffr_0.8", "best_epoch": result["best_epoch"], "best_score": result["best_score"]}); mb.update({"seed": seed, "fold": fold, "threshold_type": "fold_tuned_balanced", "best_epoch": result["best_epoch"], "best_score": result["best_score"]}); fold_metrics += [m08, mb]

    oof_score = oof_score_sum / np.maximum(oof_count, 1)
    summary_rows=[]
    if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
        for name, thr, metric in [("fixed_0.5",0.5,None),("oof_tuned_accuracy",None,"accuracy"),("oof_tuned_balanced_accuracy",None,"balanced_accuracy"),("oof_tuned_f1_macro",None,"f1_macro")]:
            selected_thr, metrics = (thr, compute_classification_metrics(y_all, oof_score, thr)) if thr is not None else tune_risk_threshold(y_all, oof_score, metric)
            metrics.update({"mode": args.mode, "threshold_type": name, "selected_threshold": float(selected_thr), "num_seeds": len(args.seeds), "seeds": ",".join(map(str,args.seeds))}); summary_rows.append(metrics)
        pd.DataFrame({"serial_no": oof_serials, "FFR": y_ffr_all, "true_ffr_class": y_all, "p_risk": oof_score}).to_csv(out_dir / "oof_predictions_ensemble.csv", index=False)
    else:
        for name, thr, metric in [("fixed_ffr_0.8",0.8,None),("oof_tuned_accuracy",None,"accuracy"),("oof_tuned_balanced_accuracy",None,"balanced_accuracy"),("oof_tuned_f1_macro",None,"f1_macro")]:
            selected_thr, metrics = (thr, compute_regression_metrics(y_ffr_all, oof_score, thr)) if thr is not None else tune_ffr_threshold(y_ffr_all, oof_score, metric)
            metrics.update({"mode": args.mode, "threshold_type": name, "selected_ffr_threshold": float(selected_thr), "num_seeds": len(args.seeds), "seeds": ",".join(map(str,args.seeds))}); summary_rows.append(metrics)
        pd.DataFrame({"serial_no": oof_serials, "FFR": y_ffr_all, "true_ffr_class": y_all, "pred_ffr": oof_score}).to_csv(out_dir / "oof_predictions_ensemble.csv", index=False)

    summary_df = pd.DataFrame(summary_rows); summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False); pd.DataFrame(fold_metrics).to_csv(out_dir / "per_fold_metrics.csv", index=False)
    make_learning_curve_summary(out_dir)
    if args.save_plots: save_plots(out_dir, args.mode)

    test_df = pd.DataFrame({"serial_no": sorted(list(test_feature_map.keys()))}) if args.mode == "ffr_regression" else pd.DataFrame({"serial_no": [p.stem for p in sorted(test_dir.glob('*.mha'))]})
    if len(test_df) > 0:
        test_scores=[]; test_serials=None
        for _, _, result, scaler in all_results:
            model = result["model"]
            if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
                loader = make_loader(test_df, test_dir, args, False, False, test_feature_map, scaler); serials, score, _, _ = predict_classifier(model, loader, device, args.mode)
            else:
                loader = make_loader(test_df, None, args, False, False, test_feature_map, scaler); serials, score, _, _ = predict_regression(model, loader, device)
            if test_serials is None: test_serials = serials
            test_scores.append(score)
        test_score = np.mean(np.stack(test_scores, axis=0), axis=0)
        if args.mode in {"lesion_cnn", "fusion_cnn_tabular"}:
            pd.DataFrame({"serial_no": test_serials, "p_risk": test_score}).to_csv(out_dir / "test_public_probabilities.csv", index=False)
            thr_acc = float(summary_df.loc[summary_df["threshold_type"]=="oof_tuned_accuracy","selected_threshold"].iloc[0]); thr_bal = float(summary_df.loc[summary_df["threshold_type"]=="oof_tuned_balanced_accuracy","selected_threshold"].iloc[0]); thr_f1 = float(summary_df.loc[summary_df["threshold_type"]=="oof_tuned_f1_macro","selected_threshold"].iloc[0])
            for name, thr in [("thr05",0.5),("acc",thr_acc),("bal",thr_bal),("f1",thr_f1)]: save_quickcheck(test_serials, risk_prob_to_ffr_class(test_score, thr), out_dir / f"quickcheck_{name}.csv")
        else:
            pd.DataFrame({"serial_no": test_serials, "pred_ffr": test_score}).to_csv(out_dir / "test_public_regression.csv", index=False)
            thr_acc = float(summary_df.loc[summary_df["threshold_type"]=="oof_tuned_accuracy","selected_ffr_threshold"].iloc[0]); thr_bal = float(summary_df.loc[summary_df["threshold_type"]=="oof_tuned_balanced_accuracy","selected_ffr_threshold"].iloc[0]); thr_f1 = float(summary_df.loc[summary_df["threshold_type"]=="oof_tuned_f1_macro","selected_ffr_threshold"].iloc[0])
            for name, thr in [("ffr080",0.8),("acc",thr_acc),("bal",thr_bal),("f1",thr_f1)]: save_quickcheck(test_serials, (test_score >= thr).astype(int), out_dir / f"quickcheck_{name}.csv")

    lines=[f"Step 8 Summary: {args.mode}", "="*80, "", "[CV Results]", summary_df.to_string(index=False), "", "[Training history files]", "history_seed{seed}_fold{fold}.csv", "learning_curve_all_epochs.csv", "learning_curve_summary.csv"]
    if args.save_plots: lines.append("curve_*.png")
    (out_dir / "model_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n[DONE] Step 8 completed."); print(f"[CHECK] {out_dir / 'model_comparison.txt'}"); print(f"[CHECK] {out_dir / 'learning_curve_summary.csv'}")


if __name__ == "__main__":
    main()
