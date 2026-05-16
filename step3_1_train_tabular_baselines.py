#!/usr/bin/env python3
# step3_train_tabular_baselines.py

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from sklearn.utils.class_weight import compute_sample_weight


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


RANDOM_STATE = 42


def get_risk_target(ffr_class: np.ndarray) -> np.ndarray:
    """
    Submission class:
        ffr_class = 0 if FFR < 0.8
        ffr_class = 1 if FFR >= 0.8

    For training, define risk target:
        risk = 1 if FFR < 0.8
        risk = 0 if FFR >= 0.8

    Therefore:
        risk = 1 - ffr_class
    """
    return 1 - ffr_class.astype(int)


def risk_prob_to_ffr_class(p_risk: np.ndarray, threshold: float) -> np.ndarray:
    """
    p_risk = P(FFR < 0.8)

    If p_risk >= threshold:
        predict risk lesion -> ffr_class = 0
    Else:
        predict normal/non-risk -> ffr_class = 1
    """
    return np.where(p_risk >= threshold, 0, 1).astype(int)


def ffr_class_to_risk_label(ffr_class_pred: np.ndarray) -> np.ndarray:
    """
    Convert predicted ffr_class to risk label.

    ffr_class = 0 -> risk = 1
    ffr_class = 1 -> risk = 0
    """
    return 1 - ffr_class_pred.astype(int)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    min_std: float = 1e-12,
):
    """
    Prepare numeric feature matrix.

    Removes:
      - label columns
      - string metadata
      - constant columns based on train set

    Keeps same feature columns for train and test.
    """
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

    X_train_df = train_df[candidate_cols].copy()

    # Replace inf/-inf just in case.
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)

    # NaN handling: Step 2 summary says no NaN, but keep robust.
    train_median = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(train_median)

    # Remove constant or near-constant features.
    std = X_train_df.std(axis=0)
    keep_cols = std[std > min_std].index.tolist()

    X_train_df = X_train_df[keep_cols]

    if test_df is not None:
        X_test_df = test_df.reindex(columns=keep_cols).copy()
        X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)
        X_test_df = X_test_df.fillna(train_median.reindex(keep_cols))
    else:
        X_test_df = None

    return X_train_df, X_test_df, keep_cols


def compute_metrics(
    y_true_ffr_class: np.ndarray,
    p_risk: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute metrics using ffr_class and risk probability.

    y_true_ffr_class:
        0 = FFR < 0.8
        1 = FFR >= 0.8

    p_risk:
        P(FFR < 0.8)
    """
    y_pred_ffr_class = risk_prob_to_ffr_class(p_risk, threshold)

    y_true_risk = ffr_class_to_risk_label(y_true_ffr_class)
    y_pred_risk = ffr_class_to_risk_label(y_pred_ffr_class)

    tn, fp, fn, tp = confusion_matrix(
        y_true_ffr_class,
        y_pred_ffr_class,
        labels=[0, 1],
    ).ravel()

    # Confusion matrix labels=[0,1]:
    # row true class 0/1, col pred class 0/1
    # cm[0,0] = true class0 predicted class0
    # cm[0,1] = true class0 predicted class1
    # cm[1,0] = true class1 predicted class0
    # cm[1,1] = true class1 predicted class1
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
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    Select threshold by maximizing a metric.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 4)

    best_thr = 0.5
    best_metrics = None
    best_value = -1e18

    for thr in thresholds:
        metrics = compute_metrics(y_true_ffr_class, p_risk, threshold=float(thr))
        value = metrics[metric_name]
        if value > best_value:
            best_value = value
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


def make_models():
    """
    Return model configs.

    All models are trained to predict risk target:
        risk = 1 if FFR < 0.8 else 0

    Therefore model.predict_proba(X)[:, 1] is p_risk.
    """
    models = {}

    models["logreg_C0p1"] = {
        "model": LogisticRegression(
            penalty="l2",
            C=0.1,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "scale": True,
        "sample_weight": False,
    }

    models["logreg_C1"] = {
        "model": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "scale": True,
        "sample_weight": False,
    }

    models["logreg_C3"] = {
        "model": LogisticRegression(
            penalty="l2",
            C=3.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "scale": True,
        "sample_weight": False,
    }

    models["random_forest"] = {
        "model": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale": False,
        "sample_weight": False,
    }

    models["random_forest_depth10"] = {
        "model": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale": False,
        "sample_weight": False,
    }

    models["extra_trees"] = {
        "model": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale": False,
        "sample_weight": False,
    }

    models["extra_trees_leaf2"] = {
        "model": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale": False,
        "sample_weight": False,
    }

    models["hist_gbdt_lr003"] = {
        "model": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            min_samples_leaf=20,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "scale": False,
        "sample_weight": True,
    }

    models["hist_gbdt_lr005"] = {
        "model": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            min_samples_leaf=20,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "scale": False,
        "sample_weight": True,
    }

    models["hist_gbdt_small"] = {
        "model": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=400,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            min_samples_leaf=30,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "scale": False,
        "sample_weight": True,
    }

    return models


def get_p_risk_from_model(model, X):
    """
    Return P(risk=1).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))

    pred = model.predict(X)
    return pred.astype(float)


def train_single_model_cv(
    model_name: str,
    config: dict,
    X: pd.DataFrame,
    y_ffr_class: np.ndarray,
    folds: np.ndarray,
    out_dir: Path,
):
    """
    Train one model with predefined folds.
    """
    y_risk = get_risk_target(y_ffr_class)

    n = len(X)
    oof_p_risk = np.zeros(n, dtype=np.float64)
    per_fold_rows = []
    fitted_models = []

    for fold in sorted(np.unique(folds)):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()

        y_tr_risk = y_risk[tr_idx]
        y_va_ffr_class = y_ffr_class[va_idx]

        scaler = None
        if config["scale"]:
            scaler = StandardScaler()
            X_tr_np = scaler.fit_transform(X_tr)
            X_va_np = scaler.transform(X_va)
        else:
            X_tr_np = X_tr.values
            X_va_np = X_va.values

        model = clone(config["model"])

        fit_kwargs = {}
        if config.get("sample_weight", False):
            fit_kwargs["sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=y_tr_risk,
            )

        model.fit(X_tr_np, y_tr_risk, **fit_kwargs)

        p_va = get_p_risk_from_model(model, X_va_np)
        oof_p_risk[va_idx] = p_va

        # Fold metrics using default threshold 0.5
        fold_metrics_05 = compute_metrics(y_va_ffr_class, p_va, threshold=0.5)
        fold_metrics_05.update({
            "model": model_name,
            "fold": int(fold),
            "threshold_type": "fixed_0.5",
        })
        per_fold_rows.append(fold_metrics_05)

        # Fold metrics using balanced-accuracy-tuned threshold inside the fold validation.
        # This is optimistic if used per-fold, but useful for seeing threshold sensitivity.
        thr_bal, fold_metrics_bal = tune_threshold(
            y_va_ffr_class,
            p_va,
            metric_name="balanced_accuracy",
        )
        fold_metrics_bal.update({
            "model": model_name,
            "fold": int(fold),
            "threshold_type": "fold_tuned_balanced_accuracy",
        })
        per_fold_rows.append(fold_metrics_bal)

        fitted_models.append({
            "fold": int(fold),
            "model": model,
            "scaler": scaler,
        })

    # OOF-level threshold tuning.
    threshold_acc, oof_metrics_acc = tune_threshold(
        y_ffr_class,
        oof_p_risk,
        metric_name="accuracy",
    )
    threshold_bal, oof_metrics_bal = tune_threshold(
        y_ffr_class,
        oof_p_risk,
        metric_name="balanced_accuracy",
    )
    threshold_f1, oof_metrics_f1 = tune_threshold(
        y_ffr_class,
        oof_p_risk,
        metric_name="f1_macro",
    )

    oof_metrics_05 = compute_metrics(y_ffr_class, oof_p_risk, threshold=0.5)

    summary_rows = []
    for name, thr, metrics in [
        ("fixed_0.5", 0.5, oof_metrics_05),
        ("oof_tuned_accuracy", threshold_acc, oof_metrics_acc),
        ("oof_tuned_balanced_accuracy", threshold_bal, oof_metrics_bal),
        ("oof_tuned_f1_macro", threshold_f1, oof_metrics_f1),
    ]:
        row = dict(metrics)
        row.update({
            "model": model_name,
            "threshold_type": name,
            "selected_threshold": float(thr),
        })
        summary_rows.append(row)

    oof_df = pd.DataFrame({
        "serial_no": None,
        "fold": folds,
        "true_ffr_class": y_ffr_class,
        "true_risk": y_risk,
        "p_risk": oof_p_risk,
    })

    return {
        "model_name": model_name,
        "summary_rows": summary_rows,
        "per_fold_rows": per_fold_rows,
        "oof_p_risk": oof_p_risk,
        "oof_df": oof_df,
        "fitted_models": fitted_models,
        "best_threshold_bal": threshold_bal,
        "best_threshold_acc": threshold_acc,
        "best_threshold_f1": threshold_f1,
    }


def predict_test_with_cv_models(
    fitted_models: list[dict],
    config: dict,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Average fold models' risk probabilities on test set.
    """
    preds = []

    for item in fitted_models:
        model = item["model"]
        scaler = item["scaler"]

        if scaler is not None:
            X_np = scaler.transform(X_test)
        else:
            X_np = X_test.values

        p = get_p_risk_from_model(model, X_np)
        preds.append(p)

    return np.mean(np.stack(preds, axis=0), axis=0)


def majority_baseline(
    y_ffr_class: np.ndarray,
):
    """
    Always predict ffr_class=1.

    This corresponds to p_risk=0 for all samples.
    """
    p_risk = np.zeros(len(y_ffr_class), dtype=np.float64)

    rows = []
    for threshold_type, threshold in [
        ("fixed_0.5", 0.5),
        ("majority_all_class1", 1.0),
    ]:
        metrics = compute_metrics(y_ffr_class, p_risk, threshold=threshold)
        metrics.update({
            "model": "majority_class1",
            "threshold_type": threshold_type,
            "selected_threshold": float(threshold),
        })
        rows.append(metrics)

    return p_risk, rows


def save_quickcheck_csv(
    serials: np.ndarray,
    p_risk: np.ndarray,
    threshold: float,
    out_path: Path,
):
    """
    Save quick check CSV.

    Required format:
        serial_no,ffr_class

    ffr_class:
        0 when FFR < 0.8
        1 when FFR >= 0.8
    """
    pred_class = risk_prob_to_ffr_class(p_risk, threshold=threshold)

    df = pd.DataFrame({
        "serial_no": serials,
        "ffr_class": pred_class.astype(int),
    })
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


def write_model_comparison(summary_df: pd.DataFrame, out_path: Path):
    lines = []

    lines.append("Step 3 Tabular Baseline Model Comparison")
    lines.append("=" * 70)
    lines.append("")

    lines.append("[Best rows by threshold_type = oof_tuned_balanced_accuracy]")
    bal = summary_df[summary_df["threshold_type"] == "oof_tuned_balanced_accuracy"].copy()
    bal = bal.sort_values(["balanced_accuracy", "accuracy"], ascending=False)
    lines.append(bal.to_string(index=False))
    lines.append("")

    lines.append("[Best rows by threshold_type = oof_tuned_accuracy]")
    acc = summary_df[summary_df["threshold_type"] == "oof_tuned_accuracy"].copy()
    acc = acc.sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    lines.append(acc.to_string(index=False))
    lines.append("")

    lines.append("[Fixed threshold 0.5]")
    fixed = summary_df[summary_df["threshold_type"] == "fixed_0.5"].copy()
    fixed = fixed.sort_values(["balanced_accuracy", "accuracy"], ascending=False)
    lines.append(fixed.to_string(index=False))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=str, default="step2_outputs/train_features.csv")
    parser.add_argument("--test-features", type=str, default="step2_outputs/test_public_features.csv")
    parser.add_argument("--out-dir", type=str, default="step3_outputs")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quickcheck-threshold-type", type=str, default="balanced",
                        choices=["balanced", "accuracy", "f1", "fixed05"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_features)
    test_path = Path(args.test_features)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    if "serial_no" not in train_df.columns:
        raise ValueError("train_features.csv must contain serial_no column.")
    if "FFR" not in train_df.columns or "ffr_class" not in train_df.columns:
        raise ValueError("train_features.csv must contain FFR and ffr_class columns.")

    y_ffr_class = train_df["ffr_class"].astype(int).values
    y_risk = get_risk_target(y_ffr_class)
    serials_train = train_df["serial_no"].astype(str).values

    X_train, X_test, feature_cols = prepare_features(train_df, test_df)

    print(f"[INFO] train rows: {len(train_df)}")
    print(f"[INFO] test rows: {len(test_df) if test_df is not None else 0}")
    print(f"[INFO] usable features after filtering: {len(feature_cols)}")
    print(f"[INFO] class counts ffr_class: {dict(pd.Series(y_ffr_class).value_counts().sort_index())}")
    print(f"[INFO] risk counts: {dict(pd.Series(y_risk).value_counts().sort_index())}")

    # Save selected feature columns.
    feature_info = {
        "num_features": len(feature_cols),
        "feature_cols": feature_cols,
        "dropped_or_excluded_note": [
            "serial_no",
            "file_path",
            "error",
            "unique_values",
            "FFR",
            "ffr_class",
            "constant_or_near_constant_features",
        ],
    }
    (out_dir / "feature_columns.json").write_text(
        json.dumps(feature_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[SAVE] {out_dir / 'feature_columns.json'}")

    # Build folds.
    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    folds = np.full(len(train_df), -1, dtype=int)
    for fold, (_, va_idx) in enumerate(skf.split(X_train, y_ffr_class)):
        folds[va_idx] = fold

    pd.DataFrame({
        "serial_no": serials_train,
        "fold": folds,
        "ffr_class": y_ffr_class,
        "risk": y_risk,
        "FFR": train_df["FFR"].values,
    }).to_csv(out_dir / "fold_assignments.csv", index=False)
    print(f"[SAVE] {out_dir / 'fold_assignments.csv'}")

    all_summary_rows = []
    all_per_fold_rows = []
    all_oof_frames = []

    # Majority baseline.
    majority_p_risk, majority_rows = majority_baseline(y_ffr_class)
    all_summary_rows.extend(majority_rows)

    majority_oof = pd.DataFrame({
        "serial_no": serials_train,
        "fold": folds,
        "model": "majority_class1",
        "true_ffr_class": y_ffr_class,
        "true_risk": y_risk,
        "p_risk": majority_p_risk,
    })
    all_oof_frames.append(majority_oof)

    # Model baselines.
    models = make_models()

    test_pred_store = {}

    for model_name, config in models.items():
        print("")
        print("=" * 80)
        print(f"[TRAIN] {model_name}")

        result = train_single_model_cv(
            model_name=model_name,
            config=config,
            X=X_train,
            y_ffr_class=y_ffr_class,
            folds=folds,
            out_dir=out_dir,
        )

        all_summary_rows.extend(result["summary_rows"])
        all_per_fold_rows.extend(result["per_fold_rows"])

        oof_df = result["oof_df"].copy()
        oof_df["serial_no"] = serials_train
        oof_df["model"] = model_name
        all_oof_frames.append(oof_df)

        # Test prediction by CV model averaging.
        if X_test is not None:
            p_test = predict_test_with_cv_models(
                fitted_models=result["fitted_models"],
                config=config,
                X_test=X_test,
            )
            test_pred_store[model_name] = {
                "p_test_risk": p_test,
                "threshold_bal": result["best_threshold_bal"],
                "threshold_acc": result["best_threshold_acc"],
                "threshold_f1": result["best_threshold_f1"],
            }

            if args.quickcheck_threshold_type == "balanced":
                thr = result["best_threshold_bal"]
                suffix = "bal"
            elif args.quickcheck_threshold_type == "accuracy":
                thr = result["best_threshold_acc"]
                suffix = "acc"
            elif args.quickcheck_threshold_type == "f1":
                thr = result["best_threshold_f1"]
                suffix = "f1"
            else:
                thr = 0.5
                suffix = "thr05"

            save_quickcheck_csv(
                serials=test_df["serial_no"].astype(str).values,
                p_risk=p_test,
                threshold=thr,
                out_path=out_dir / f"quickcheck_{model_name}_{suffix}.csv",
            )

    # Save summary files.
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df = summary_df.sort_values(
        ["threshold_type", "balanced_accuracy", "accuracy"],
        ascending=[True, False, False],
    )
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    print(f"[SAVE] {out_dir / 'cv_results_summary.csv'}")

    per_fold_df = pd.DataFrame(all_per_fold_rows)
    per_fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    print(f"[SAVE] {out_dir / 'per_fold_metrics.csv'}")

    oof_all = pd.concat(all_oof_frames, axis=0, ignore_index=True)
    oof_all.to_csv(out_dir / "oof_predictions.csv", index=False)
    print(f"[SAVE] {out_dir / 'oof_predictions.csv'}")

    write_model_comparison(summary_df, out_dir / "model_comparison.txt")

    # Save an ensemble quickcheck using top models by OOF balanced accuracy.
    if X_test is not None and len(test_pred_store) > 0:
        bal_rows = summary_df[
            summary_df["threshold_type"] == "oof_tuned_balanced_accuracy"
        ].copy()
        bal_rows = bal_rows[bal_rows["model"].isin(test_pred_store.keys())]
        bal_rows = bal_rows.sort_values(["balanced_accuracy", "accuracy"], ascending=False)

        top_models = bal_rows["model"].head(3).tolist()
        if len(top_models) > 0:
            p_ensemble = np.mean(
                np.stack([test_pred_store[m]["p_test_risk"] for m in top_models], axis=0),
                axis=0,
            )

            # Tune ensemble threshold using OOF probabilities from selected models.
            oof_p_list = []
            for m in top_models:
                m_oof = oof_all[oof_all["model"] == m].sort_values("serial_no")
                # Align by serial_no just to be safe.
                m_oof = m_oof.set_index("serial_no").loc[serials_train]
                oof_p_list.append(m_oof["p_risk"].values)

            p_oof_ensemble = np.mean(np.stack(oof_p_list, axis=0), axis=0)

            thr_ens_bal, ens_metrics_bal = tune_threshold(
                y_ffr_class,
                p_oof_ensemble,
                metric_name="balanced_accuracy",
            )
            thr_ens_acc, ens_metrics_acc = tune_threshold(
                y_ffr_class,
                p_oof_ensemble,
                metric_name="accuracy",
            )

            ensemble_info = {
                "top_models": top_models,
                "threshold_balanced_accuracy": thr_ens_bal,
                "metrics_balanced_accuracy": ens_metrics_bal,
                "threshold_accuracy": thr_ens_acc,
                "metrics_accuracy": ens_metrics_acc,
            }

            (out_dir / "ensemble_info.json").write_text(
                json.dumps(ensemble_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[SAVE] {out_dir / 'ensemble_info.json'}")

            if args.quickcheck_threshold_type == "accuracy":
                thr_ens = thr_ens_acc
                suffix = "acc"
            else:
                thr_ens = thr_ens_bal
                suffix = "bal"

            save_quickcheck_csv(
                serials=test_df["serial_no"].astype(str).values,
                p_risk=p_ensemble,
                threshold=thr_ens,
                out_path=out_dir / f"quickcheck_ensemble_top3_{suffix}.csv",
            )

    print("")
    print("[DONE] Step 3 tabular baselines completed.")
    print(f"[CHECK] Main result: {out_dir / 'model_comparison.txt'}")


if __name__ == "__main__":
    main()