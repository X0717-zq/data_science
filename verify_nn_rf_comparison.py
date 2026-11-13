"""Independently reload every selected checkpoint and verify held-out results."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_nn_rf_comparison import (ROOT, SOURCE, FraudMLP, predict, rank_metrics,
                                 reference, sha, read_csv, write_json, partition, sampler_for)


def main():
    c=json.loads((ROOT/"nn_comparison_config.json").read_text(encoding="utf-8"))
    out=ROOT/c["output"]
    status=json.loads((out/"status.json").read_text(encoding="utf-8"))
    assert status["phase"]=="completed" and status["completed"]==35
    frozen=json.loads((out/"all_checkpoints_frozen.json").read_text(encoding="utf-8"))
    assert frozen["test_scoring_started"] is False and len(frozen["models"])==35
    for path,digest in json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))["source_hashes"].items():
        assert sha(path)==digest
    torch.set_num_threads(c["num_threads"])
    torch.set_num_interop_threads(1)
    data=pd.read_csv(ROOT/c["input"])
    ref=reference()
    plan_path=ROOT/c["frozen_splits"]
    plan=json.loads(plan_path.read_text(encoding="utf-8"));folder=plan_path.parent
    predictions=read_csv(out/"tables/nn_test_predictions_with_features.csv")
    metrics=read_csv(out/"tables/nn_metrics_by_round.csv")
    original=read_csv(SOURCE/"test_predictions_by_round.csv")
    combined=read_csv(out/"tables/all_metrics_by_round.csv")
    combined_summary=read_csv(out/"tables/all_model_summary_mean_std.csv").set_index("method")
    checks=[];sampler_checks=[]
    for e in plan["rounds"]:
        indices={key:partition(data,e,key,plan,folder) for key in ["train","validation","test"]}
        assert len(set(indices["train"])|set(indices["validation"])|set(indices["test"]))==len(data)
        round_dir=out/"models"/f"round_{e['repeat']:02d}"
        prep=joblib.load(round_dir/"preprocessor.joblib")
        fresh=ref.make_preprocessor()
        fresh.fit(data.iloc[indices["train"]][ref.FEATURES])
        np.testing.assert_array_equal(prep.transform(data[ref.FEATURES]),fresh.transform(data[ref.FEATURES]))
        tx=prep.transform(data.iloc[indices["train"]][ref.FEATURES])
        ty=data.is_fraud.iloc[indices["train"]].to_numpy()
        sx,sy=sampler_for("SMOTE",e["model_seed"],c).fit_resample(tx,ty)
        cx,cy=sampler_for("SMOTE+Tomek",e["model_seed"],c).fit_resample(tx,ty)
        sampler_checks.append(dict(repeat=e["repeat"],smote_n=len(sy),smote_tomek_n=len(cy),
            removed_by_tomek=len(sy)-len(cy),
            identical_resampled_data=bool(np.array_equal(sx,cx) and np.array_equal(sy,cy))))
        for j,v in enumerate(c["variants"]):
            dest=round_dir/f"variant_{j:02d}"
            info=json.loads((dest/"training_complete.json").read_text(encoding="utf-8"))
            for name,digest in info["files"].items():
                assert sha(dest/name)==digest
            h=read_csv(dest/"epoch_history.csv")
            assert list(h.epoch)==list(range(1,info["epochs_run"]+1))
            best=h[np.isclose(h.validation_pr_auc,h.validation_pr_auc.max(),rtol=0,
                              atol=c["improvement_tolerance"])].iloc[0]
            assert int(best.epoch)==info["best_epoch"]
            assert abs(best.validation_pr_auc-info["best_validation_pr_auc"])<1e-12
            assert not any("test" in name for name in h.columns)
            state=torch.load(dest/"best_model.pt",map_location="cpu",weights_only=True)
            model=FraudMLP(state["input_size"],state["hidden_layers"],state["dropout"])
            model.load_state_dict(state["state_dict"])
            val=data.iloc[indices["validation"]]
            vs=predict(model,prep.transform(val[ref.FEATURES]))
            saved=np.load(dest/"train_validation_scores.npz")
            np.testing.assert_array_equal(val.is_fraud,saved["validation_y"])
            np.testing.assert_allclose(vs,saved["validation_score"],rtol=0,atol=1e-12)
            threshold,f2=ref.select_f2_threshold(val.is_fraud.to_numpy(),vs)
            assert abs(threshold-info["threshold_from_validation"])<1e-12
            assert abs(f2-info["validation_f2"])<1e-12
            test=data.iloc[indices["test"]]
            ts=predict(model,prep.transform(test[ref.FEATURES]))
            pred=predictions[(predictions["repeat"]==e["repeat"])&(predictions.method==v["method"])]
            np.testing.assert_array_equal(pred.transaction_id,test.transaction_id)
            np.testing.assert_array_equal(pred.y_true,test.is_fraud)
            np.testing.assert_allclose(pred.y_score,ts,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(pred.y_pred,(ts>=threshold).astype(int))
            rf=original[(original["repeat"]==e["repeat"])&(original.method==v["rf_method"])]
            np.testing.assert_array_equal(pred.transaction_id,rf.transaction_id)
            np.testing.assert_array_equal(pred.y_true,rf.y_true)
            calculated=ref.evaluate_scores(test.is_fraud.to_numpy(),ts,threshold)
            row=metrics[(metrics["repeat"]==e["repeat"])&(metrics.method==v["method"])].iloc[0]
            for metric,value in calculated.items():
                assert abs(row[metric]-value)<1e-12,(v["method"],metric)
            checks.append(dict(repeat=e["repeat"],method=v["method"],passed=True,
                best_epoch=info["best_epoch"],validation_pr_auc=best.validation_pr_auc,
                test_pr_auc=calculated["pr_auc"]))
    for method,g in combined.groupby("method"):
        for metric in ["pr_auc","ap","roc_auc","accuracy","precision","recall","f1","f2","mcc","fpr"]:
            s=combined_summary.loc[method]
            assert np.isclose(g[metric].mean(),s[metric+"_mean"],rtol=0,atol=1e-12)
            assert np.isclose(g[metric].std(ddof=1),s[metric+"_std"],rtol=0,atol=1e-12)
    assert len(predictions)==35*1492 and len(checks)==35 and len(combined)==80
    manifest=json.loads((SOURCE/"chapter6_supplement/source_manifest.json").read_text(encoding="utf-8"))
    for rel,digest in manifest["source_hashes"].items():
        assert sha(ROOT/rel)==digest
    pd.DataFrame(checks).to_csv(out/"qa/checkpoints_verified.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(sampler_checks).to_csv(out/"qa/smote_tomek_equivalence.csv",index=False,encoding="utf-8-sig")
    result=dict(passed=True,verified_checkpoints=35,verified_nn_prediction_rows=len(predictions),
        compared_rf_rounds=35,original_reference_rounds=45,original_sources_preserved=True,
        preprocessing_fitted_on_train_only=True,frozen_indices_exactly_matched=True,
        best_epochs_recomputed_from_validation_history=True,validation_thresholds_recomputed=True,
        test_predictions_reloaded_from_checkpoints=True,all_metrics_and_sample_SD_recomputed=True,
        no_test_columns_in_training_history=True)
    write_json(out/"qa/verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
