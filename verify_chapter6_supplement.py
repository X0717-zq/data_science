"""Read-only numerical validation followed by a separate verification record."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from chapter6_common import ROOT, SOURCE, OUT, METHODS, initialize, sha, read_csv, save_json


def metric_values(labels, score):
    p,r,_=precision_recall_curve(labels,score)
    return dict(PR_AUC=auc(r,p),AP=average_precision_score(labels,score),ROC_AUC=roc_auc_score(labels,score))


def main():
    initialize()
    folder=OUT/"learning_curves"
    data=pd.read_csv(ROOT/"fraudulent_card.csv")
    plan=json.loads((folder/"frozen_training_subsets.json").read_text(encoding="utf-8"))
    results=read_csv(folder/"learning_curve_metrics.csv")
    summary=read_csv(folder/"learning_curve_summary.csv").set_index(["method","fraction"])
    assert len(results)==225 and results.status.eq("ok").all()
    assert not results.duplicated(["repeat","method","fraction"]).any()
    assert set(results.method)==set(METHODS)
    predictions=read_csv(SOURCE/"test_predictions_by_round.csv")
    for path,hashed in plan["index_file_hashes"].items():
        assert sha(folder/path)==hashed
    for entry in plan["rounds"]:
        parts={n:read_csv(folder/f) for n,f in entry["partitions"].items()}
        ids={n:set(f.row_position) for n,f in parts.items()}
        assert not ids['train']&ids['validation'] and not ids['train']&ids['test'] and not ids['validation']&ids['test']
        test=predictions[(predictions['repeat']==entry['repeat'])&(predictions.method=='RF-Plain')]
        np.testing.assert_array_equal(parts['test'].transaction_id,test.transaction_id)
        seen=set()
        for subset in entry['subsets']:
            idx=read_csv(folder/subset['file'])
            actual=set(idx.row_position)
            assert seen<=actual<=ids['train'] and not actual&ids['validation'] and not actual&ids['test']
            assert len(idx)==subset['train_n'] and idx.is_fraud.sum()==subset['train_positive']
            assert idx.is_fraud.mean()<.01
            np.testing.assert_array_equal(data.transaction_id.iloc[idx.row_position],idx.transaction_id)
            np.testing.assert_array_equal(data.is_fraud.iloc[idx.row_position],idx.is_fraud)
            seen=actual
        assert seen==ids['train']
    for row in results.itertuples(index=False):
        f=folder/row.scores_file
        assert sha(f)==row.scores_sha256
        with np.load(f,allow_pickle=False) as stored:
            for part,index_key,score_key in [('train','train_row_position','train_score'),
                                              ('validation','validation_row_position','validation_score')]:
                idx=stored[index_key]
                expected_idx=read_csv(folder/(row.subset_file if part=='train' else row.validation_file)).row_position
                np.testing.assert_array_equal(idx,expected_idx)
                labels=data.is_fraud.iloc[idx].to_numpy()
                assert len(labels)==len(stored[score_key])
                for m,value in metric_values(labels,stored[score_key]).items():
                    np.testing.assert_allclose(value,getattr(row,f'{part}_{m}'),rtol=0,atol=1e-12)
    for (method,fraction),block in results.groupby(['method','fraction']):
        assert len(block)==5
        for part in ['train','validation']:
            for m in ['PR_AUC','AP','ROC_AUC']:
                key=f'{part}_{m}'
                for operation in ['mean','std']:
                    value=block[key].mean() if operation=='mean' else block[key].std(ddof=1)
                    np.testing.assert_allclose(summary.loc[(method,fraction),f'{key}_{operation}'],value,rtol=0,atol=1e-12)
    replay=read_csv(folder/'full_size_validation_replay_check.csv')
    assert len(replay)==45 and replay.threshold_matches.all() and replay.validation_F2_matches.all()
    expanded=read_csv(OUT/'tables/test_predictions_with_features.csv')
    errors=read_csv(OUT/'case_analysis/fp_fn_all_occurrences.csv')
    cases=read_csv(OUT/'case_analysis/representative_fp_fn_cases.csv')
    assert len(expanded)==67140 and len(errors)==828
    assert errors.outcome.isin(['FP','FN']).all()
    assert len(cases)==54 and not cases.duplicated(['method','outcome','transaction_id']).any()
    for _,case in cases.iterrows():
        source=errors[(errors['repeat']==case['repeat'])&(errors.method==case.method)&(errors.transaction_id==case.transaction_id)]
        assert len(source)==1
        for field in ['y_true','y_pred','y_score','threshold','amount','device_trust_score','velocity_last_24h']:
            np.testing.assert_allclose(source.iloc[0][field],case[field],rtol=0,atol=1e-12)
    pooled=read_csv(OUT/'tables/confusion_matrices_pooled.csv').set_index('method')
    by_round=read_csv(OUT/'tables/confusion_matrices_by_round.csv')
    for method,g in expanded.groupby('method'):
        for outcome,col in [('TP','tp'),('FP','fp'),('TN','tn'),('FN','fn')]:
            assert pooled.loc[method,col]==g.outcome.eq(outcome).sum()
        assert pooled.loc[method,'prediction_occurrences']==7460
    common=read_csv(OUT/'tables/roc_curve_common_grid_by_round.csv')
    curve_summary=read_csv(OUT/'tables/roc_curve_mean_std.csv')
    check=common.groupby(['method','fpr']).tpr.agg(['mean','std']).reset_index()
    actual=curve_summary.merge(check,on=['method','fpr'],validate='one_to_one')
    np.testing.assert_allclose(actual.mean_tpr,actual['mean'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(actual.std_tpr,actual['std'],rtol=0,atol=1e-12)
    assert curve_summary['rounds'].eq(5).all()
    for _,g in curve_summary.groupby('method'):
        assert np.all(np.diff(g.sort_values('fpr').mean_tpr)>=-1e-12)
    # Re-read source manifest after all verification to prove no original output was replaced.
    initialize()
    result=dict(passed=True,original_results_preserved=True,learning_jobs_verified=225,
        frozen_nested_subsets_verified=True,learning_test_set_disjoint=True,
        full_size_validation_replay_matches=45,learning_metrics_and_SD_recomputed=True,
        enriched_prediction_rows=67140,error_occurrences=828,representative_cases=54,
        pooled_confusion_verified=True,mean_ROC_and_SD_verified=True,
        scope='numerical and provenance checks; figure QA is recorded separately')
    save_json(OUT/'verification.json',result)
    shutil.copy2(__file__,OUT/'code_snapshot'/Path(__file__).name)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
