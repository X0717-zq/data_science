"""Reproduce five frozen RF-Plain fits and explain every test occurrence."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve, confusion_matrix

ROOT = Path(__file__).resolve().parent
LABELS = {"amount": "交易金额", "transaction_hour": "交易时段", "foreign_transaction": "境外交易标记", "location_mismatch": "位置不匹配", "device_trust_score": "设备信任评分", "velocity_last_24h": "24小时交易次数", "cardholder_age": "持卡人年龄", "merchant_category": "商户类别"}

def read(path):
    return pd.read_csv(path, float_precision="round_trip")

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")

def csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pr_auc(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "chapter7_explainability_config.json")
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    source, out = ROOT / c["source_results"], ROOT / c["output"]
    for d in ["tables", "figures", "models", "qa", "code_snapshot"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / c["source_manifest"]).read_text(encoding="utf-8"))
    source_hashes = manifest["source_hashes"]
    assert all(sha(ROOT / p) == h for p, h in source_hashes.items()), "Frozen source changed"
    for p in [args.config, Path(__file__), ROOT / c["reference_script"]]:
        shutil.copy2(p, out / "code_snapshot" / p.name)
    spec = importlib.util.spec_from_file_location("chapter7_reference", ROOT / c["reference_script"])
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    df = read(ROOT / c["dataset"])
    assert (len(df), int(df.is_fraud.sum())) == (9943, 94)
    frozen = ROOT / c["frozen_splits"]
    rounds = json.loads((frozen / "frozen_training_subsets.json").read_text(encoding="utf-8"))["rounds"]
    original = read(source / "test_predictions_by_round.csv")
    original = original[original.method.eq("RF-Plain")]
    metrics = read(source / "metrics_by_round.csv")
    metrics = metrics[metrics.method.eq("RF-Plain")]
    all_shap, all_imp, all_perm, checks, all_groups = [], [], [], [], []
    split_hashes = {}
    start = time.time()
    for info in rounds:
        repeat, seed = info["repeat"], info["model_seed"]
        print(f"Round {repeat}/5: reproducing frozen RF-Plain", flush=True)
        parts = {}
        for key, name in info["partitions"].items():
            p = frozen / name
            split_hashes[str(p.relative_to(ROOT))] = sha(p)
            idx = read(p)
            part = df.iloc[idx.row_position.to_numpy()].copy().reset_index(drop=True)
            np.testing.assert_array_equal(part.transaction_id, idx.transaction_id)
            np.testing.assert_array_equal(part.is_fraud, idx.is_fraud)
            parts[key] = part
        tr, va, te = (parts[k] for k in ["train", "validation", "test"])
        ids = {k: set(v.transaction_id) for k, v in parts.items()}
        assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
        prep = ref.make_preprocessor()
        Xtr = prep.fit_transform(tr[ref.FEATURES])
        Xv, Xt = (prep.transform(x[ref.FEATURES]) for x in [va, te])
        model = RandomForestClassifier(random_state=seed, **c["model"])
        model.fit(Xtr, tr.is_fraud)
        vs, ts = model.predict_proba(Xv)[:, 1], model.predict_proba(Xt)[:, 1]
        threshold, vf2 = ref.select_f2_threshold(va.is_fraud, vs)
        expected = original[original.repeat.eq(repeat)].set_index("transaction_id").loc[te.transaction_id]
        met = metrics[metrics.repeat.eq(repeat)].iloc[0]
        np.testing.assert_allclose(ts, expected.y_score, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(te.is_fraud, expected.y_true)
        np.testing.assert_allclose([threshold, vf2], [met.threshold_from_validation, met.validation_f2_at_threshold], rtol=0, atol=1e-12)
        pred = (ts >= threshold).astype(int)
        np.testing.assert_array_equal(pred, expected.y_pred)
        current = ref.evaluate_scores(te.is_fraud.to_numpy(), ts, threshold)
        for key, value in current.items():
            if key in met.index:
                np.testing.assert_allclose(value, met[key], rtol=0, atol=1e-12)
        md = out / "models" / f"round_{repeat:02d}"
        md.mkdir(exist_ok=True)
        joblib.dump({"preprocessor": prep, "model": model, "features": ref.FEATURES, "threshold": threshold, "model_seed": seed}, md / "rf_plain.joblib")
        np.savez_compressed(md / "validation_test_scores.npz", validation_score=vs, test_score=ts, test_transaction_id=te.transaction_id.to_numpy())
        encoded_names = prep.get_feature_names_out().tolist()
        groups = {f: [i] for i, f in enumerate(ref.NUM_COLS)}
        groups["merchant_category"] = list(range(len(ref.NUM_COLS), Xt.shape[1]))
        print(f"Round {repeat}: scores matched; exact TreeSHAP for {len(te)} test rows", flush=True)
        explainer = shap.TreeExplainer(model, feature_perturbation=c["shap"]["feature_perturbation"], model_output=c["shap"]["model_output"])
        sv = explainer.shap_values(Xt, check_additivity=True, approximate=c["shap"]["approximate"])
        cls = int(np.flatnonzero(model.classes_ == c["shap"]["positive_class"])[0])
        sv = sv[cls] if isinstance(sv, list) else sv[:, :, cls]
        base = float(np.asarray(explainer.expected_value)[cls])
        err = float(np.max(np.abs(base + sv.sum(axis=1) - ts)))
        assert err < 1e-10, f"SHAP does not reconstruct probability score: {err}"
        grouped = np.column_stack([sv[:, groups[f]].sum(axis=1) for f in ref.FEATURES])
        np.testing.assert_allclose(grouped.sum(axis=1), sv.sum(axis=1), atol=1e-12, rtol=0)
        np.savez_compressed(md / "shap_encoded.npz", shap_values=sv, expected_value=base, encoded_names=np.array(encoded_names), transaction_id=te.transaction_id.to_numpy())
        rows = te.copy()
        rows.insert(0, "repeat", repeat)
        rows["y_score"], rows["threshold"], rows["y_pred"], rows["base_value"] = ts, threshold, pred, base
        rows["outcome"] = np.select([(te.is_fraud.eq(1) & (pred == 1)), (te.is_fraud.eq(0) & (pred == 1)), (te.is_fraud.eq(1) & (pred == 0))], ["TP", "FP", "FN"], default="TN")
        for i, f in enumerate(ref.FEATURES):
            rows["shap_" + f] = grouped[:, i]
            all_imp.append(dict(repeat=repeat, feature=f, feature_cn=LABELS[f], mean_abs_shap=np.abs(grouped[:, i]).mean(), fraud_mean_abs_shap=np.abs(grouped[te.is_fraud.eq(1), i]).mean(), normal_mean_abs_shap=np.abs(grouped[te.is_fraud.eq(0), i]).mean(), mdi=model.feature_importances_[groups[f]].sum()))
        all_shap.append(rows)
        print(f"Round {repeat}: permuting 8 raw-feature groups, {c['permutation']['repeats_per_feature_per_round']} repetitions each", flush=True)
        baseline = pr_auc(te.is_fraud, ts)
        for j in range(c["permutation"]["repeats_per_feature_per_round"]):
            permutation_seed = seed + c["permutation"]["seed_offset"] + j
            order = np.random.default_rng(permutation_seed).permutation(len(te))
            for f, columns in groups.items():
                changed = Xt.copy()
                changed[:, columns] = Xt[order][:, columns]
                shuffled = pr_auc(te.is_fraud, model.predict_proba(changed)[:, 1])
                all_perm.append(dict(repeat=repeat, feature=f, permutation_index=j + 1, permutation_seed=permutation_seed, baseline_pr_auc=baseline, shuffled_pr_auc=shuffled, pr_auc_drop=baseline - shuffled))
        checks.append(dict(repeat=repeat, split_seed=info["split_seed"], model_seed=seed, train_n=len(tr), validation_n=len(va), test_n=len(te), test_positive=int(te.is_fraud.sum()), max_score_difference=float(np.max(np.abs(ts - expected.y_score.to_numpy()))), threshold=threshold, validation_f2=vf2, test_pr_auc=baseline, shap_base_value=base, shap_max_additivity_error=err, exact_predictions_match=True))
        print(f"Round {repeat} finished; elapsed {time.time() - start:.1f}s", flush=True)
    sh = pd.concat(all_shap, ignore_index=True)
    im = pd.DataFrame(all_imp)
    pm = pd.DataFrame(all_perm)
    pmean = pm.groupby(["repeat", "feature"], as_index=False).agg(permutation_pr_auc_drop=("pr_auc_drop", "mean"), permutation_within_round_sd=("pr_auc_drop", "std"))
    byround = im.merge(pmean, on=["repeat", "feature"], validate="one_to_one")
    summary = byround.groupby(["feature", "feature_cn"], as_index=False).agg(shap_mean=("mean_abs_shap", "mean"), shap_sd=("mean_abs_shap", "std"), permutation_drop_mean=("permutation_pr_auc_drop", "mean"), permutation_drop_sd=("permutation_pr_auc_drop", "std"), mdi_mean=("mdi", "mean"), mdi_sd=("mdi", "std"), fraud_shap_mean=("fraud_mean_abs_shap", "mean"), normal_shap_mean=("normal_mean_abs_shap", "mean"))
    summary["shap_rank"] = summary.shap_mean.rank(ascending=False, method="min").astype(int)
    summary["permutation_rank"] = summary.permutation_drop_mean.rank(ascending=False, method="min").astype(int)
    summary = summary.sort_values("shap_rank")
    csv(sh, out / "tables" / "test_shap_all_occurrences.csv")
    csv(byround, out / "tables" / "feature_importance_by_round.csv")
    csv(summary, out / "tables" / "feature_importance_summary.csv")
    csv(pm, out / "tables" / "permutation_importance_all_repeats.csv")
    csv(pd.DataFrame(checks), out / "tables" / "rf_reproduction_checks.csv")
    # All selected cases were specified before the new explanation was run.
    cases = []
    for case in c["local_cases"]:
        row = sh[sh.repeat.eq(case["repeat"]) & sh.transaction_id.eq(case["transaction_id"])]
        assert len(row) == 1 and row.iloc[0].outcome == case["expected_outcome"]
        cases.append(row)
    csv(pd.concat(cases), out / "tables" / "selected_fp_fn_shap_cases.csv")
    # Descriptive direction summaries; cutpoints come only from round-one TRAIN.
    train_idx = read(frozen / rounds[0]["partitions"]["train"])
    first_train = df.iloc[train_idx.row_position.to_numpy()]
    for f in ref.NUM_COLS:
        if set(df[f].unique()).issubset({0, 1}):
            bins = [-np.inf, 0.5, np.inf]
            labels = ["0", "1"]
        else:
            q = np.unique(first_train[f].quantile([0.25, 0.5, 0.75]).to_numpy())
            bins = np.r_[-np.inf, q, np.inf]
            labels = [f"({a:g}, {b:g}]" for a, b in zip(bins[:-1], bins[1:])]
        g = pd.cut(sh[f], bins=bins, labels=labels, include_lowest=True)
        for label in labels:
            sub = sh[g.eq(label)]
            uniq = sub.drop_duplicates("transaction_id")
            all_groups.append(dict(feature=f, feature_cn=LABELS[f], group=label, definition="round-one training quartiles; binary fields 0/1", prediction_occurrences=len(sub), unique_transactions=len(uniq), unique_fraud=int(uniq.is_fraud.sum()), descriptive_unique_fraud_rate=float(uniq.is_fraud.mean()), mean_shap=float(sub["shap_" + f].mean()), median_shap=float(sub["shap_" + f].median())))
    csv(pd.DataFrame(all_groups), out / "tables" / "feature_direction_profiles.csv")
    # Count and threshold evidence, respecting repeated transaction occurrences.
    pooled = metrics.groupby("method", as_index=False)[["tn", "fp", "fn", "tp"]].sum()
    csv(pooled, out / "tables" / "rf_pooled_confusion_counts.csv")
    errors = sh[sh.outcome.isin(["FP", "FN"])]
    csv(errors, out / "tables" / "all_rf_errors_with_shap.csv")
    report = dict(all_checks_passed=True, source_files_unchanged=all(sha(ROOT / p) == h for p, h in source_hashes.items()), rounds=checks, test_occurrences=len(sh), unique_test_transactions=int(sh.transaction_id.nunique()), test_positive_occurrences=int(sh.is_fraud.sum()), unique_positive_test_transactions=int(sh.loc[sh.is_fraud.eq(1), "transaction_id"].nunique()), features=ref.FEATURES, shap_positive_class=1, shap_output_verified_as="RandomForestClassifier.predict_proba[:,1], uncalibrated score", merchant_category_aggregation="sum signed SHAP across five one-hot columns, then take absolute value for global ranking", permutation_repeats_per_feature_per_round=c["permutation"]["repeats_per_feature_per_round"], permutation_test_analysis_not_used_for_selection=True, no_retraining_after_explanations=True, elapsed_seconds=time.time() - start)
    assert report["source_files_unchanged"]
    save_json(out / "qa" / "numerical_verification.json", report)
    save_json(out / "run_manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=source_hashes, frozen_split_hashes=split_hashes, configuration=c, versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, shap=shap.__version__, numpy=np.__version__, pandas=pd.__version__), code_sha256=sha(__file__)))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "rounds"}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
