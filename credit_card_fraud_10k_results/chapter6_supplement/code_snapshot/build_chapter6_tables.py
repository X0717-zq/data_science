"""Recompute existing test metrics and export auditable ROC/error source tables."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve)

from chapter6_common import C, ROOT, SOURCE, OUT, METHODS, initialize, sha, csv, read_csv, save_json


def main():
    initialize()
    shutil.copy2(__file__, OUT / "code_snapshot" / Path(__file__).name)
    pred = read_csv(SOURCE / "test_predictions_by_round.csv")
    data = pd.read_csv(ROOT / "fraudulent_card.csv")
    expected = read_csv(SOURCE / "metrics_by_round.csv").set_index(["repeat", "method"])
    assert len(pred) == 67140 and not pred.duplicated(["repeat", "method", "transaction_id"]).any()
    assert data.transaction_id.is_unique
    joined = pred.merge(data, on="transaction_id", how="left", validate="many_to_one", sort=False)
    assert len(joined) == len(pred) and not joined.isna().any().any()
    np.testing.assert_array_equal(joined.y_true, joined.is_fraud)
    joined = joined.drop(columns="is_fraud")
    np.testing.assert_array_equal(joined.y_pred, (joined.y_score >= joined.threshold).astype(int))
    joined["outcome"] = np.select([
        (joined.y_true == 1) & (joined.y_pred == 1), (joined.y_true == 0) & (joined.y_pred == 1),
        (joined.y_true == 0) & (joined.y_pred == 0), (joined.y_true == 1) & (joined.y_pred == 0)],
        ["TP", "FP", "TN", "FN"], default="invalid")
    assert not joined.outcome.eq("invalid").any()
    joined["score_minus_threshold"] = joined.y_score - joined.threshold
    joined["score_type"] = np.where(joined.method.eq("Isolation Forest"),
        "negative_score_samples; anomaly score, not probability", "predict_proba class 1; not calibrated")
    csv(joined, OUT / "tables" / "test_predictions_with_features.csv")
    shutil.copy2(SOURCE / "test_predictions_by_round.csv", OUT / "tables" / "test_predictions_by_round.csv")
    errors = joined[joined.outcome.isin(["FP", "FN"])].copy()
    csv(errors, OUT / "case_analysis" / "fp_fn_all_occurrences.csv")
    metrics, curve_rows, grid_rows = [], [], []
    grid = np.unique(np.r_[np.linspace(0, 1, 1001), np.linspace(0, .05, 501)])
    for (repeat, method), g in joined.groupby(["repeat", "method"], sort=False):
        assert len(g) == 1492 and g.y_true.sum() == 14 and g.threshold.nunique() == 1
        y, score, label = g.y_true.to_numpy(), g.y_score.to_numpy(), g.y_pred.to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, label, labels=[0,1]).ravel()
        p, r, _ = precision_recall_curve(y, score)
        m = dict(roc_auc=roc_auc_score(y,score), pr_auc=auc(r,p), ap=average_precision_score(y,score),
            accuracy=accuracy_score(y,label), precision=precision_score(y,label,zero_division=0),
            recall=recall_score(y,label,zero_division=0), f1=f1_score(y,label,zero_division=0),
            f2=fbeta_score(y,label,beta=2,zero_division=0), mcc=matthews_corrcoef(y,label),
            fpr=fp/(fp+tn), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
        for k,v in m.items():
            np.testing.assert_allclose(expected.loc[(repeat,method),k],v,rtol=0,atol=1e-12)
        metrics.append(dict(repeat=int(repeat),method=method,test_n=len(g),test_positive=int(y.sum()),
                            threshold=float(g.threshold.iloc[0]),fnr=fn/(fn+tp),**m))
        fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
        for x, z, t in zip(fpr, tpr, thresholds):
            curve_rows.append(dict(repeat=repeat,method=method,fpr=x,tpr=z,threshold=t))
        # At duplicate FPR values retain the highest attained TPR before interpolation.
        unique = pd.DataFrame(dict(fpr=fpr,tpr=tpr)).groupby("fpr",as_index=False).tpr.max()
        assert np.all(np.diff(unique.fpr)>0) and np.all(np.diff(unique.tpr)>=0)
        interpolated = np.interp(grid, unique.fpr, unique.tpr)
        for x,z in zip(grid,interpolated):
            grid_rows.append(dict(repeat=repeat,method=method,fpr=x,tpr=z))
    metrics = pd.DataFrame(metrics)
    csv(metrics, OUT / "tables" / "test_metrics_verified_by_round.csv")
    csv(metrics[["repeat","method","test_n","threshold","tn","fp","fn","tp","fpr","fnr"]],
        OUT / "tables" / "confusion_matrices_by_round.csv")
    summary = metrics.groupby("method",sort=False).agg(
        roc_auc_mean=("roc_auc","mean"),roc_auc_std=("roc_auc","std"),
        pr_auc_mean=("pr_auc","mean"),pr_auc_std=("pr_auc","std"),
        ap_mean=("ap","mean"),ap_std=("ap","std")).reset_index()
    csv(summary, OUT / "tables" / "roc_auc_summary.csv")
    original_summary = read_csv(SOURCE / "summary_mean_std.csv").set_index("method")
    for row in summary.itertuples(index=False):
        for k in ["roc_auc_mean","roc_auc_std","pr_auc_mean","pr_auc_std","ap_mean","ap_std"]:
            np.testing.assert_allclose(getattr(row,k),original_summary.loc[row.method,k],atol=1e-12,rtol=0)
    csv(pd.DataFrame(curve_rows), OUT / "tables" / "roc_curve_points_by_round.csv")
    grid_df = pd.DataFrame(grid_rows)
    csv(grid_df, OUT / "tables" / "roc_curve_common_grid_by_round.csv")
    roc_mean = grid_df.groupby(["method","fpr"],sort=False).agg(
        mean_tpr=("tpr","mean"),std_tpr=("tpr","std"),rounds=("repeat","nunique")).reset_index()
    csv(roc_mean, OUT / "tables" / "roc_curve_mean_std.csv")
    pooled = metrics.groupby("method",sort=False)[["tn","fp","fn","tp"]].sum().reset_index()
    pooled["prediction_occurrences"] = pooled[["tn","fp","fn","tp"]].sum(axis=1)
    pooled["fpr_pooled"] = pooled.fp/(pooled.fp+pooled.tn)
    pooled["fnr_pooled"] = pooled.fn/(pooled.fn+pooled.tp)
    pooled["recall_pooled"] = 1-pooled.fnr_pooled
    pooled["precision_pooled"] = pooled.tp/(pooled.tp+pooled.fp)
    assert pooled.prediction_occurrences.eq(7460).all()
    csv(pooled, OUT / "tables" / "confusion_matrices_pooled.csv")
    flags = joined.assign(is_FP=joined.outcome.eq("FP").astype(int),is_FN=joined.outcome.eq("FN").astype(int))
    frequency = flags.groupby(["method","transaction_id"],sort=False).agg(
        y_true=("y_true","first"),evaluated_rounds=("repeat","nunique"),
        FP_rounds=("is_FP","sum"),FN_rounds=("is_FN","sum"),
        score_mean=("y_score","mean"),score_min=("y_score","min"),score_max=("y_score","max")).reset_index()
    frequency["error_rounds"] = frequency.FP_rounds+frequency.FN_rounds
    frequency["error_fraction_when_evaluated"] = frequency.error_rounds/frequency.evaluated_rounds
    frequency = frequency.merge(data.drop(columns="is_fraud"),on="transaction_id",validate="many_to_one")
    csv(frequency,OUT/"case_analysis"/"transaction_error_frequency.csv")
    unique_counts = frequency.groupby("method",sort=False).agg(
        unique_test_transactions=("transaction_id","nunique"),
        unique_FP_transactions=("FP_rounds",lambda s:int((s>0).sum())),
        unique_FN_transactions=("FN_rounds",lambda s:int((s>0).sum()))).reset_index()
    error_summary = pooled.merge(unique_counts,on="method",validate="one_to_one")
    csv(error_summary,OUT/"case_analysis"/"error_summary_by_method.csv")
    representatives=[]
    for method in METHODS:
        for outcome in ["FP","FN"]:
            g = errors[(errors.method==method)&(errors.outcome==outcome)]
            g = g.sort_values(["score_minus_threshold","transaction_id","repeat"],
                              ascending=[outcome=="FN",True,True])
            chosen = g.drop_duplicates("transaction_id").head(C["representative_cases_per_method_and_error"]).copy()
            chosen["selection_rule"] = "largest positive margin for FP / most negative margin for FN, within model; unique IDs"
            representatives.append(chosen)
    csv(pd.concat(representatives,ignore_index=True),OUT/"case_analysis"/"representative_fp_fn_cases.csv")
    profiles=[]
    numeric=["amount","transaction_hour","foreign_transaction","location_mismatch",
             "device_trust_score","velocity_last_24h","cardholder_age"]
    for (method,outcome),g in joined.groupby(["method","outcome"],sort=False):
        for feature in numeric:
            profiles.append(dict(method=method,outcome=outcome,feature=feature,
                prediction_occurrences=len(g),unique_transactions=g.transaction_id.nunique(),
                mean=g[feature].mean(),median=g[feature].median(),q25=g[feature].quantile(.25),q75=g[feature].quantile(.75)))
    csv(pd.DataFrame(profiles),OUT/"case_analysis"/"error_numeric_feature_profiles.csv")
    categories=joined.groupby(["method","outcome","merchant_category"],sort=False).size().rename("prediction_occurrences").reset_index()
    categories["within_outcome_fraction"]=categories.prediction_occurrences/categories.groupby(["method","outcome"]).prediction_occurrences.transform("sum")
    csv(categories,OUT/"case_analysis"/"error_merchant_profiles.csv")
    save_json(OUT/"tables"/"verification.json",dict(passed=True,verified_model_rounds=45,
        prediction_rows=len(joined),unique_transactions=joined.transaction_id.nunique(),
        error_occurrences=len(errors),all_original_metrics_reproduced=True,source_files_unchanged=True,
        confusion_sum_per_method=7460,positive_occurrences_per_method=70,
        note="Five repeated test sets overlap; pooled counts describe prediction occurrences, not independent transactions."))
    print(error_summary.to_string(index=False))
    print("ROC, confusion and case source tables complete; all original metrics matched.")


if __name__=="__main__":
    main()
