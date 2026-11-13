# -*- coding: utf-8 -*-
"""Low-prevalence credit-card fraud benchmark (nine methods).

Expected final data: 9,943 rows = 9,849 normal + 94 fraud (0.9454%).
If fraudulent_card.csv is present beside this script, it is used unchanged.
Otherwise credit_card_fraud_10k.csv is reduced deterministically by retaining
all normal records and sampling 94 of the 151 fraud records.

Protocol: five repeated stratified 70%/15%/15% train/validation/test splits.
Preprocessing is fitted on training data only; F2 threshold is selected on
validation data and then frozen for test evaluation.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import imblearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    auc, average_precision_score, precision_recall_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, confusion_matrix,
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------- Experiment settings ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "credit_card_fraud_10k_results"
TARGET = "is_fraud"
ID_COL = "transaction_id"
CAT_COLS = ["merchant_category"]
NUM_COLS = [
    "amount", "transaction_hour", "foreign_transaction", "location_mismatch",
    "device_trust_score", "velocity_last_24h", "cardholder_age",
]
FEATURES = NUM_COLS + CAT_COLS

EXPECTED_NORMAL = 9849
EXPECTED_FRAUD_RAW = 151
TARGET_FRAUD = 94
DOWNSAMPLE_SEED = 20261109
REPEAT_SEEDS = [20261109, 20261110, 20261111, 20261112, 20261113]
SAMPLING_RATIO = 0.5
METHODS = [
    "RF-Plain", "RF-Balanced", "SMOTE", "ADASYN", "NearMiss",
    "SMOTE+Tomek", "SMOTE+ENN", "EasyEnsemble", "Isolation Forest",
]
RECALL_GRID = np.linspace(0.0, 1.0, 201)


def make_onehot_encoder():
    """Support both newer and older scikit-learn OneHotEncoder APIs."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", make_onehot_encoder(), CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_rf(seed: int, balanced: bool = False, trees: int = 120) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=1,
    )


def resolve_and_prepare_data(input_path: Path, output_dir: Path):
    """Load low-prevalence data unchanged, or create it from the raw 10k file."""
    df = pd.read_csv(input_path)
    required = [ID_COL] + FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要字段：{missing}\n当前文件：{input_path}")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud 必须为无缺失的 0/1 二分类标签。")
    if df[required].isna().any().any():
        bad = df[required].isna().sum()
        raise ValueError(f"模型所需字段存在缺失值：\n{bad[bad > 0]}")
    if not df[ID_COL].is_unique:
        raise ValueError("transaction_id 不是唯一值，请先检查输入数据。")

    source_n = len(df)
    source_fraud = int(df[TARGET].sum())
    source_normal = int((df[TARGET] == 0).sum())

    # Preserve an already-prepared 9,943-row dataset exactly, including row order.
    if (source_n == EXPECTED_NORMAL + TARGET_FRAUD
            and source_normal == EXPECTED_NORMAL
            and source_fraud == TARGET_FRAUD
            and float(df[TARGET].mean()) < 0.01):
        low = df.copy().reset_index(drop=True)
        reduction_applied = False
        original_rows = None
        original_fraud = None
        original_normal = None

        # If the raw companion file exists, include its counts in the report only.
        raw_companion = input_path.parent / "credit_card_fraud_10k.csv"
        if raw_companion.exists() and raw_companion.resolve() != input_path.resolve():
            raw_df = pd.read_csv(raw_companion)
            if TARGET in raw_df.columns:
                original_rows = len(raw_df)
                original_fraud = int(raw_df[TARGET].sum())
                original_normal = int((raw_df[TARGET] == 0).sum())
    elif source_n == 10000 and source_normal == EXPECTED_NORMAL and source_fraud == EXPECTED_FRAUD_RAW:
        # Deterministically reproduce the previously used 94-positive subset.
        positives = df[df[TARGET] == 1].sample(n=TARGET_FRAUD, random_state=DOWNSAMPLE_SEED)
        negatives = df[df[TARGET] == 0]
        low = pd.concat([positives, negatives], axis=0)
        low = low.sample(frac=1, random_state=DOWNSAMPLE_SEED).reset_index(drop=True)
        reduction_applied = True
        original_rows, original_fraud, original_normal = source_n, source_fraud, source_normal
    else:
        raise ValueError(
            "输入数据不符合本实验的两种预期格式。\n"
            "允许：已经处理好的 9,943 行（9,849 正常 + 94 欺诈）数据；"
            "或原始 10,000 行（9,849 正常 + 151 欺诈）数据。\n"
            f"当前：总数={source_n}，正常={source_normal}，欺诈={source_fraud}，"
            f"正例率={df[TARGET].mean():.4%}。\n"
            "请确认 --input 指向 credit_card_fraud_10k.csv 或 fraudulent_card.csv。"
        )

    low_rate = float(low[TARGET].mean())
    if len(low) != 9943 or int(low[TARGET].sum()) != 94 or not low_rate < 0.01:
        raise AssertionError("低正例率数据集校验失败：要求 9943 行、94 个正例且正例率严格小于 1%。")
    if low.isna().any().any():
        raise ValueError("处理后数据仍含缺失值。")

    output_data = output_dir / "fraudulent_card_under_1pct.csv"
    low.to_csv(output_data, index=False, encoding="utf-8-sig")

    return low, {
        "input_file": str(input_path),
        "input_rows": source_n,
        "input_fraud": source_fraud,
        "input_normal": source_normal,
        "input_prevalence": float(source_fraud / source_n),
        "reduction_applied_in_this_run": reduction_applied,
        "original_rows_if_known": original_rows,
        "original_fraud_if_known": original_fraud,
        "original_normal_if_known": original_normal,
        "retained_rows": len(low),
        "retained_fraud": int(low[TARGET].sum()),
        "retained_normal": int((low[TARGET] == 0).sum()),
        "retained_prevalence": low_rate,
        "removed_fraud_from_raw_if_known": (
            int(original_fraud - TARGET_FRAUD) if original_fraud is not None else None
        ),
        "downsample_seed": DOWNSAMPLE_SEED,
        "prepared_dataset_file": str(output_data),
    }


def select_f2_threshold(y_true, scores) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.max(scores) + 1e-12), 0.0
    p, r = precision[:-1], recall[:-1]
    f2 = (5.0 * p * r) / np.maximum(4.0 * p + r, 1e-15)
    best = np.nanmax(f2)
    tied = np.flatnonzero(np.isclose(f2, best, rtol=1e-10, atol=1e-12))
    # Fixed tie rule: choose the larger threshold (more conservative).
    idx = tied[np.argmax(thresholds[tied])]
    return float(thresholds[idx]), float(f2[idx])


def train_and_score(method, X_train, y_train, X_val, X_test, seed):
    """Fit preprocessing on train only; resample only train; return val/test scores."""
    prep = make_preprocessor()
    Xtr = prep.fit_transform(X_train)
    Xv = prep.transform(X_val)
    Xs = prep.transform(X_test)

    if method in ("RF-Plain", "RF-Balanced"):
        model = make_rf(seed, balanced=(method == "RF-Balanced"))
        model.fit(Xtr, y_train)
        return model.predict_proba(Xv)[:, 1], model.predict_proba(Xs)[:, 1]

    if method == "SMOTE":
        sampler = SMOTE(sampling_strategy=SAMPLING_RATIO, random_state=seed, k_neighbors=5)
    elif method == "ADASYN":
        sampler = ADASYN(sampling_strategy=SAMPLING_RATIO, random_state=seed, n_neighbors=5)
    elif method == "NearMiss":
        sampler = NearMiss(version=1, sampling_strategy=SAMPLING_RATIO, n_neighbors=3)
    elif method == "SMOTE+Tomek":
        sampler = SMOTETomek(
            smote=SMOTE(sampling_strategy=SAMPLING_RATIO, random_state=seed, k_neighbors=5),
            random_state=seed,
        )
    elif method == "SMOTE+ENN":
        sampler = SMOTEENN(
            smote=SMOTE(sampling_strategy=SAMPLING_RATIO, random_state=seed, k_neighbors=5),
            random_state=seed,
        )
    elif method == "EasyEnsemble":
        base_rf = RandomForestClassifier(
            n_estimators=60, max_features="sqrt", min_samples_leaf=1,
            class_weight=None, random_state=seed, n_jobs=1,
        )
        try:
            model = EasyEnsembleClassifier(
                n_estimators=10, estimator=base_rf, random_state=seed, n_jobs=1
            )
        except TypeError:  # imbalanced-learn versions using base_estimator
            model = EasyEnsembleClassifier(
                n_estimators=10, base_estimator=base_rf, random_state=seed, n_jobs=1
            )
        model.fit(Xtr, y_train)
        return model.predict_proba(Xv)[:, 1], model.predict_proba(Xs)[:, 1]
    elif method == "Isolation Forest":
        model = IsolationForest(
            n_estimators=200, contamination="auto", max_samples="auto",
            random_state=seed, n_jobs=1,
        )
        # Unsupervised fitting: no labels are passed to IsolationForest.fit.
        model.fit(Xtr)
        return -model.score_samples(Xv), -model.score_samples(Xs)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(Xtr, y_train)
    model = make_rf(seed, balanced=False)
    model.fit(X_resampled, y_resampled)
    return model.predict_proba(Xv)[:, 1], model.predict_proba(Xs)[:, 1]


def evaluate_scores(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "ap": float(average_precision_score(y_true, scores)),
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def make_mean_curve(recall, precision):
    """Interpolate one PR curve onto a common increasing recall grid."""
    order = np.argsort(recall)
    r_sorted = np.asarray(recall)[order]
    p_sorted = np.asarray(precision)[order]
    # Precision envelope: best precision attainable at this or greater recall.
    p_envelope = np.maximum.accumulate(p_sorted[::-1])[::-1]
    return np.interp(RECALL_GRID, r_sorted, p_envelope,
                     left=p_envelope[0], right=p_envelope[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Credit-card fraud detection: low-prevalence, nine-method comparison"
    )
    # Prefer the already-prepared credit-card dataset. Do NOT auto-select fraudulent.csv,
    # because that filename may belong to the separate URL-fraud project.
    default_input = next((p for p in [
        SCRIPT_DIR / "fraudulent_card.csv",
        SCRIPT_DIR / "credit_card_fraud_10k.csv",
    ] if p.exists()), None)
    parser.add_argument(
        "--input", default=str(default_input) if default_input else None,
        help="输入 CSV；默认优先 fraudulent_card.csv，其次 credit_card_fraud_10k.csv",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="结果输出目录")
    args = parser.parse_args()
    if not args.input:
        parser.error("未找到数据文件。请将 fraudulent_card.csv 或 credit_card_fraud_10k.csv 放在脚本同目录，或使用 --input 指定。")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    low, data_info = resolve_and_prepare_data(input_path, output_dir)
    X = low[FEATURES].copy()
    y = low[TARGET].astype(int).copy()
    metric_rows, split_rows, curve_rows, prediction_rows = [], [], [], []
    mean_curve_store = {method: [] for method in METHODS}

    for repeat, split_seed in enumerate(REPEAT_SEEDS, start=1):
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=split_seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=0.50, stratify=y_rest,
            random_state=split_seed + 1000,
        )

        train_rate = float(y_train.mean())
        val_rate = float(y_val.mean())
        test_rate = float(y_test.mean())
        if max(train_rate, val_rate, test_rate) >= 0.01:
            raise AssertionError("某一数据划分正例率达到或超过 1%，实验终止。")

        split_rows.append({
            "repeat": repeat, "seed": split_seed,
            "train_n": len(y_train), "train_pos": int(y_train.sum()), "train_rate": train_rate,
            "validation_n": len(y_val), "validation_pos": int(y_val.sum()), "validation_rate": val_rate,
            "test_n": len(y_test), "test_pos": int(y_test.sum()), "test_rate": test_rate,
        })
        print(
            f"Round {repeat}/5: train={len(y_train)} ({int(y_train.sum())} fraud, {train_rate:.4%}), "
            f"val={len(y_val)} ({int(y_val.sum())} fraud, {val_rate:.4%}), "
            f"test={len(y_test)} ({int(y_test.sum())} fraud, {test_rate:.4%})",
            flush=True,
        )

        for method in METHODS:
            print(f"  Fitting {method}...", flush=True)
            model_seed = split_seed + repeat
            val_scores, test_scores = train_and_score(
                method, X_train, y_train, X_val, X_test, model_seed
            )
            threshold, validation_f2 = select_f2_threshold(y_val.to_numpy(), val_scores)
            metrics = evaluate_scores(y_test.to_numpy(), test_scores, threshold)
            metric_rows.append({
                "repeat": repeat, "split_seed": split_seed, "model_seed": model_seed,
                "method": method,
                "validation_f2_at_threshold": validation_f2,
                "threshold_from_validation": threshold,
                "validation_positive": int(y_val.sum()),
                "test_positive": int(y_test.sum()),
                "test_size": int(len(y_test)),
                "test_prevalence": test_rate,
                **metrics,
            })

            p, r, _ = precision_recall_curve(y_test.to_numpy(), test_scores)
            mean_curve_store[method].append(make_mean_curve(r, p))
            for rv, pv in zip(r, p):
                curve_rows.append({
                    "repeat": repeat, "method": method,
                    "recall": float(rv), "precision": float(pv),
                })

            # Preserve scored test observations for later FP/FN and threshold analysis.
            test_frame = pd.DataFrame({
                "repeat": repeat,
                "method": method,
                ID_COL: low.loc[X_test.index, ID_COL].to_numpy(),
                "y_true": y_test.to_numpy(),
                "y_score": test_scores,
                "threshold": threshold,
                "y_pred": (test_scores >= threshold).astype(int),
            })
            prediction_rows.extend(test_frame.to_dict("records"))

            # Incremental checkpoint, so completed rounds survive an interruption.
            pd.DataFrame(metric_rows).to_csv(
                output_dir / "metrics_by_round_partial.csv", index=False, encoding="utf-8-sig"
            )
        print(f"Round {repeat}/5 complete.", flush=True)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "metrics_by_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(output_dir / "split_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(output_dir / "pr_curve_points_by_round.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "test_predictions_by_round.csv", index=False, encoding="utf-8-sig")

    metric_cols = ["pr_auc", "ap", "roc_auc", "accuracy", "precision", "recall", "f1", "f2", "mcc", "fpr"]
    summary = metrics_df.groupby("method", sort=False).agg(
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        **{f"{c}_std": (c, "std") for c in metric_cols},
    ).reset_index()
    summary["pr_auc_mean_std"] = summary.apply(
        lambda row: f"{row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}", axis=1
    )
    summary = summary.sort_values("pr_auc_mean", ascending=False)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    # Five-round averaged PR curve family (not a single-round representative curve).
    curve_summary_rows = []
    plt.figure(figsize=(10, 7))
    for method in METHODS:
        curves = np.vstack(mean_curve_store[method])
        mean_precision = curves.mean(axis=0)
        std_precision = curves.std(axis=0, ddof=1)
        plt.plot(RECALL_GRID, mean_precision, linewidth=1.8, label=method)
        plt.fill_between(
            RECALL_GRID,
            np.clip(mean_precision - std_precision, 0, 1),
            np.clip(mean_precision + std_precision, 0, 1),
            alpha=0.10,
        )
        for recall_value, mean_value, std_value in zip(RECALL_GRID, mean_precision, std_precision):
            curve_summary_rows.append({
                "method": method, "recall": recall_value,
                "mean_precision": mean_value, "std_precision": std_value,
            })
    mean_test_prevalence = float(pd.DataFrame(split_rows)["test_rate"].mean())
    plt.axhline(
        mean_test_prevalence, linestyle="--", linewidth=1.2,
        label=f"No-skill test prevalence ({mean_test_prevalence:.3%})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves: Mean over 5 Repeated Stratified Test Splits")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves_family_mean.png", dpi=220)
    plt.close()
    pd.DataFrame(curve_summary_rows).to_csv(
        output_dir / "pr_curve_family_mean_std.csv", index=False, encoding="utf-8-sig"
    )

    # Five-round mean PR-AUC ± sample SD comparison.
    ordered = summary.sort_values("pr_auc_mean", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ordered["method"], ordered["pr_auc_mean"], xerr=ordered["pr_auc_std"], capsize=3)
    plt.xlabel("Mean test PR-AUC (± SD)")
    plt.title("Nine Methods: Test PR-AUC over 5 Repeated Stratified Splits")
    plt.xlim(0, 1.03)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_auc_comparison.png", dpi=220)
    plt.close()

    # Accuracy counterexample for the constructed low-prevalence dataset.
    n_total = len(low)
    n_fraud = int(y.sum())
    n_normal = n_total - n_fraud
    accuracy_all_normal = n_normal / n_total
    counterexample = {
        "scenario": "Predict every transaction as legitimate (0)",
        "total": n_total, "fraud": n_fraud, "normal": n_normal,
        "fraud_prevalence": n_fraud / n_total,
        "accuracy": accuracy_all_normal,
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0, "mcc": 0.0,
    }
    (output_dir / "accuracy_counterexample.json").write_text(
        json.dumps(counterexample, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save exact experimental configuration and environment.
    experiment_plan = {
        "dataset": data_info,
        "target": TARGET,
        "positive_class": 1,
        "features": FEATURES,
        "excluded_features": [ID_COL],
        "numeric_features": NUM_COLS,
        "categorical_features": CAT_COLS,
        "n_repeats": len(REPEAT_SEEDS),
        "split_seeds": REPEAT_SEEDS,
        "model_seed_rule": "model_seed = split_seed + repeat_number",
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "stratified": True},
        "threshold_selection": "Maximum F2 on validation; tie-break chooses larger threshold; frozen for test",
        "rf": {"n_estimators": 120, "max_features": "sqrt", "min_samples_leaf": 1, "n_jobs": 1},
        "rf_plain": {"class_weight": None, "resampling": False},
        "rf_balanced": {"class_weight": "balanced", "resampling": False},
        "sampling_ratio": SAMPLING_RATIO,
        "samplers": {
            "SMOTE": {"sampling_strategy": 0.5, "k_neighbors": 5},
            "ADASYN": {"sampling_strategy": 0.5, "n_neighbors": 5},
            "NearMiss": {"version": 1, "sampling_strategy": 0.5, "n_neighbors": 3},
            "SMOTE+Tomek": {"SMOTE_sampling_strategy": 0.5, "k_neighbors": 5},
            "SMOTE+ENN": {"SMOTE_sampling_strategy": 0.5, "k_neighbors": 5},
            "EasyEnsemble": {"n_estimators": 10, "base_rf_estimators": 60, "sampling_strategy": "library default"},
        },
        "isolation_forest": {"n_estimators": 200, "contamination": "auto", "fit_uses_labels": False},
        "preprocessing": "StandardScaler + OneHotEncoder, fit on training split only",
        "curve_method": "Interpolate each test PR curve to a common recall grid; report mean and SD over five splits",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
            "matplotlib": plt.matplotlib.__version__, "platform": platform.platform(),
        },
        "important_limits": [
            "The 94 retained fraud records are a deterministic subset of the original 151; 57 fraud records were discarded.",
            "Five repeated splits reuse the same 94 positive records; mean and SD describe split variability, not independent-sample confidence intervals.",
            "Each validation and test split contains 14 fraud records; one case changes per-split recall by about 7.14 percentage points.",
            "SMOTE variants interpolate in the one-hot encoded numerical feature space and can create fractional categorical indicator values.",
            "Isolation Forest is unsupervised during fitting; validation labels are used for threshold selection.",
        ],
    }
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(experiment_plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Concise report with actual computed values, avoiding hard-coded outcomes.
    best = summary.iloc[0]
    above_target = int((summary["pr_auc_mean"] >= 0.80).sum())
    split_df = pd.DataFrame(split_rows)
    round_detail = summary[[
        "method", "pr_auc_mean_std", "ap_mean", "precision_mean", "recall_mean",
        "f1_mean", "f2_mean", "mcc_mean", "fpr_mean", "accuracy_mean",
    ]].copy()
    report_lines = [
        "# 信用卡欺诈检测：低于 1% 正例率的九种方法对比实验",
        "",
        "## 1. 数据处理结果",
        "",
        f"- 输入文件：`{input_path.name}`。",
        f"- 本次模型数据：{n_total:,} 条，其中正常 {n_normal:,} 条、欺诈 {n_fraud:,} 条，正例率 **{n_fraud / n_total:.4%}**。",
        f"- 本次运行是否执行下采样：`{reduction_applied_text(data_info['reduction_applied_in_this_run'])}`。",
        "- 若从原始 10,000 条数据构造，则保留全部 9,849 条正常交易，从 151 条欺诈交易中固定随机抽取 94 条，剔除 57 条；若输入已处理好的 `fraudulent_card.csv`，则保持样本和原始行顺序不变。",
        "- 排除唯一标识 `transaction_id`；使用 7 个数值特征和 1 个类别特征 `merchant_category`。",
        "- 类别特征独热编码；数值特征标准化；所有预处理器只在训练集拟合。",
        "",
        "## 2. 实验设置",
        "",
        "- 五轮重复分层划分，每轮 Train/Validation/Test = 70%/15%/15%。",
        "- 每轮模型只在训练集拟合；验证集选择 F2 最大的阈值，若并列则选择较大的阈值；阈值冻结后在测试集评价。",
        "- 比较九种方法：" + "、".join(METHODS) + "。",
        "- RF-Plain：不重采样，`class_weight=None`；RF-Balanced：不重采样，`class_weight='balanced'`。",
        "- SMOTE、ADASYN、NearMiss、SMOTE+Tomek、SMOTE+ENN 的目标采样比例为 0.5；EasyEnsemble 使用其默认欠采样比例。",
        "- Isolation Forest 训练阶段不使用标签；验证集标签仅用于选择操作阈值。",
        "",
        "### 五轮划分数量",
        "",
        markdown_table(split_df),
        "",
        "## 3. 九种方法的测试结果（五轮均值）",
        "",
        "PR-AUC 为均值 ± 样本标准差；其他指标为五轮均值。FPR、Accuracy 以比例保存，制表时可转换为百分比。",
        "",
        markdown_table(round_detail),
        "",
        f"平均测试 PR-AUC 达到 0.80 的方法有 **{above_target}/9** 种。最高者为 **{best['method']}**，PR-AUC = **{best['pr_auc_mean']:.4f} ± {best['pr_auc_std']:.4f}**。是否满足任务书要求，应按任务书对达标对象的定义判断。",
        "",
        "## 4. 图表说明",
        "",
        "- `pr_curves_family_mean.png`：五轮测试集 PR 曲线分别插值到统一 Recall 网格后取平均；阴影表示各轮插值 Precision 的标准差；虚线为五轮测试集平均正例率。",
        "- `pr_auc_comparison.png`：五轮测试 PR-AUC 均值及标准差对比。",
        "- 两张图分别用于展示 PR 曲线整体形态与五轮 PR-AUC 汇总，不应把平均曲线解释成某一轮测试结果。",
        "",
        "## 5. Accuracy 失效反例",
        "",
        f"- 若每笔交易都预测为正常，Accuracy = {n_normal}/{n_total} = **{accuracy_all_normal:.4%}**。",
        "- 但所有欺诈都被漏掉，Recall、Precision、F1、F2 和 MCC 均为 0。",
        "- 因此不应单独使用 Accuracy 评价极不平衡欺诈检测；应重点报告 PR-AUC、AP，并结合具体阈值下的 Precision、Recall、F1、F2、MCC 和 FPR。",
        "",
        "## 6. 结果解释与限制",
        "",
        "- 如果 RF-Plain 取得最高 PR-AUC，说明在本数据和当前参数下，不进行重采样及类别加权的随机森林具有更好的测试排序表现；不能据此概括所有不平衡任务。",
        "- 重采样只利用有限真实欺诈样本构造合成点，可能放大已有模式或影响分类边界；NearMiss 会删除多数类样本，可能损失正常交易分布的多样性。上述机制是可能解释，不能仅凭汇总指标证明具体因果原因。",
        "- 若本次从原始数据构造低正例率数据，剔除了 57/151 个欺诈样本（约 37.7%），模型评估只覆盖保留的 94 个正例。",
        "- 每轮验证集和测试集各只有 14 个欺诈样本；单轮 Recall 每错判 1 个样本就变化约 7.14 个百分点。",
        "- 五轮划分复用同一批 94 个欺诈样本，均值 ± 标准差描述重复划分波动，不是独立样本置信区间。",
        "- SMOTE 系列在独热编码数值空间插值，可能产生非整数类别指示值；若要求严格保留类别变量语义，可另行采用 SMOTENC，但结果将属于不同实验实现。",
        "- PR-AUC 达标不等于真实银行业务部署达标，仍需外部数据或时间外数据验证；原始标签与特征之间是否存在人为生成规则，也应进一步审查。",
        "",
        "## 7. 输出文件",
        "",
        "- `fraudulent_card_under_1pct.csv`：本次实际使用的数据。",
        "- `summary_mean_std.csv`：九种方法五轮指标均值和标准差。",
        "- `metrics_by_round.csv`：逐轮指标、阈值和混淆矩阵计数。",
        "- `split_counts.csv`：五轮训练、验证、测试样本数和正例率。",
        "- `pr_curves_family_mean.png`：五轮平均 PR 曲线族。",
        "- `pr_auc_comparison.png`：五轮 PR-AUC 均值 ± 标准差图。",
        "- `pr_curve_points_by_round.csv`、`pr_curve_family_mean_std.csv`：曲线原始点与平均曲线数据。",
        "- `test_predictions_by_round.csv`：逐笔测试预测分数、阈值、真实标签和预测标签，可用于误报/漏报分析。",
        "- `experiment_plan.json`：实验配置、软件版本和限制说明。",
    ]
    (output_dir / "report_中文.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== DATA CHECK ===")
    print(f"Data: {n_total:,} rows | fraud={n_fraud} | normal={n_normal} | prevalence={n_fraud / n_total:.4%}")
    print("All five train/validation/test splits have prevalence < 1%.")
    print("\n=== SUMMARY: 5-ROUND MEAN TEST METRICS ===")
    display_cols = [
        "method", "pr_auc_mean", "pr_auc_std", "ap_mean", "precision_mean",
        "recall_mean", "f1_mean", "f2_mean", "mcc_mean", "fpr_mean", "accuracy_mean",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nMethods with mean PR-AUC >= 0.80: {above_target}/9")
    print(f"Outputs saved to: {output_dir}")


def markdown_table(frame: pd.DataFrame) -> str:
    """Create a basic Markdown table without requiring the optional tabulate package."""
    data = frame.copy().fillna("")
    headers = [str(c) for c in data.columns]
    rows = [[str(v) for v in row] for row in data.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    head = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([head, sep] + body)


def reduction_applied_text(value):
    return "是（从原始数据中抽取 94 个欺诈样本）" if value else "否（输入已是 9,943 行低正例率数据）"


if __name__ == "__main__":
    main()
