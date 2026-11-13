"""Frozen-split, seven-strategy MLP comparison against existing RF results."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "credit_card_fraud_10k_results"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def reference():
    spec = importlib.util.spec_from_file_location("original_rf", ROOT / "credit_card_fraud_10k_experiment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FraudMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout):
        super().__init__()
        layers = []
        last = input_size
        for size in hidden_layers:
            layers.extend([nn.Linear(last, size), nn.ReLU(), nn.Dropout(dropout)])
            last = size
        layers.append(nn.Linear(last, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy().astype(float)


def rank_metrics(y, score):
    p, r, _ = precision_recall_curve(y, score)
    return {"pr_auc": float(auc(r,p)), "ap": float(average_precision_score(y,score)),
            "roc_auc": float(roc_auc_score(y,score))}


def sampler_for(strategy, seed, c):
    base = lambda: SMOTE(sampling_strategy=c["sampling_ratio"], k_neighbors=c["smote_neighbors"],
                         random_state=seed)
    if strategy in ["none", "balanced"]:
        return None
    if strategy == "SMOTE":
        return base()
    if strategy == "ADASYN":
        return ADASYN(sampling_strategy=c["sampling_ratio"], n_neighbors=c["smote_neighbors"], random_state=seed)
    if strategy == "NearMiss":
        return NearMiss(version=1,sampling_strategy=c["sampling_ratio"],n_neighbors=c["nearmiss_neighbors"])
    if strategy == "SMOTE+Tomek":
        return SMOTETomek(smote=base(),random_state=seed)
    if strategy == "SMOTE+ENN":
        return SMOTEENN(smote=base(),random_state=seed)
    raise ValueError(strategy)


def partition(data, round_entry, key, plan, folder):
    rel = round_entry["partitions"][key]
    assert sha(folder/rel) == plan["index_file_hashes"][rel]
    part = read_csv(folder/rel)
    idx = part.row_position.to_numpy(dtype=int)
    np.testing.assert_array_equal(data.transaction_id.iloc[idx], part.transaction_id)
    np.testing.assert_array_equal(data.is_fraud.iloc[idx], part.is_fraud)
    return idx


def train_one(c, variant, entry, train_x, train_y, val_x, val_y, dest, ref):
    dest.mkdir(parents=True,exist_ok=True)
    complete = dest/"training_complete.json"
    if complete.exists():
        result = json.loads(complete.read_text(encoding="utf-8"))
        for name, digest in result["files"].items():
            assert sha(dest/name)==digest, name
        print(f"Reuse completed: round {entry['repeat']} {variant['method']}",flush=True)
        return result
    seed = entry["model_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sampler = sampler_for(variant["strategy"], seed, c)
    # Preserve float64 input for identical neighbour selection as the RF reference.
    fit_x, fit_y = (sampler.fit_resample(train_x, train_y)
                    if sampler is not None else (train_x.copy(), train_y.copy()))
    fit_x, fit_y = np.asarray(fit_x), np.asarray(fit_y)
    weight = float((train_y==0).sum()/(train_y==1).sum()) if variant["strategy"]=="balanced" else 1.0
    model = FraudMLP(train_x.shape[1],c["hidden_layers"],c["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,dtype=torch.float32))
    unweighted = nn.BCEWithLogitsLoss()
    tx = torch.as_tensor(fit_x,dtype=torch.float32)
    ty = torch.as_tensor(fit_y,dtype=torch.float32)
    original_x=torch.as_tensor(train_x,dtype=torch.float32)
    original_y=torch.as_tensor(train_y,dtype=torch.float32)
    vx=torch.as_tensor(val_x,dtype=torch.float32)
    vy=torch.as_tensor(val_y,dtype=torch.float32)
    best_value=-float("inf"); best_epoch=0; best_state=None; history=[]
    started=time.perf_counter()
    for epoch in range(1,c["max_epochs"]+1):
        model.train()
        order=torch.randperm(len(tx))
        running_loss=0.0
        for start in range(0,len(order),c["batch_size"]):
            batch=order[start:start+c["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model(tx[batch]),ty[batch])
            assert torch.isfinite(loss), "Non-finite training objective"
            loss.backward()
            optimizer.step()
            running_loss+=float(loss.item())*len(batch)
        model.eval()
        with torch.no_grad():
            train_logits=model(original_x); val_logits=model(vx)
            train_scores=torch.sigmoid(train_logits).numpy().astype(float)
            val_scores=torch.sigmoid(val_logits).numpy().astype(float)
            train_loss=float(unweighted(train_logits,original_y).item())
            val_loss=float(unweighted(val_logits,vy).item())
        tm=rank_metrics(train_y,train_scores); vm=rank_metrics(val_y,val_scores)
        current=vm["pr_auc"]
        improved=current>best_value+c["improvement_tolerance"]
        if improved:
            best_value=current; best_epoch=epoch
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        history.append(dict(repeat=entry["repeat"],method=variant["method"],epoch=epoch,
            optimization_loss=running_loss/len(tx),train_bce=train_loss,validation_bce=val_loss,
            **{f"train_{k}":v for k,v in tm.items()},**{f"validation_{k}":v for k,v in vm.items()},
            new_best=improved,best_epoch_so_far=best_epoch,best_validation_pr_auc=best_value))
        if epoch%50==0:
            csv(pd.DataFrame(history),dest/"epoch_history.csv")
            print(f"Round {entry['repeat']} {variant['method']} epoch={epoch} "
                  f"val_PR_AUC={current:.4f} best={best_value:.4f}@{best_epoch}",flush=True)
        if epoch>=c["min_epochs"] and epoch-best_epoch>=c["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    train_scores=predict(model,train_x); val_scores=predict(model,val_x)
    assert abs(rank_metrics(val_y,val_scores)["pr_auc"]-best_value)<1e-12
    threshold,f2=ref.select_f2_threshold(val_y,val_scores)
    torch.save({"state_dict":best_state,"input_size":int(train_x.shape[1]),
        "hidden_layers":c["hidden_layers"],"dropout":c["dropout"],
        "best_epoch":best_epoch,"model_seed":seed},dest/"best_model.pt")
    np.savez_compressed(dest/"train_validation_scores.npz",
        train_y=train_y,train_score=train_scores,validation_y=val_y,validation_score=val_scores)
    csv(pd.DataFrame(history),dest/"epoch_history.csv")
    result=dict(repeat=entry["repeat"],method=variant["method"],rf_method=variant["rf_method"],
        strategy=variant["strategy"],model_seed=seed,split_seed=entry["split_seed"],
        train_n=len(train_y),train_positive=int(train_y.sum()),
        fitted_n=len(fit_y),fitted_positive=int(fit_y.sum()),fitted_negative=int((fit_y==0).sum()),
        pos_weight=weight,validation_n=len(val_y),validation_positive=int(val_y.sum()),
        best_epoch=best_epoch,epochs_run=epoch,
        stop_reason="patience" if epoch<c["max_epochs"] else "max_epochs",
        best_validation_pr_auc=best_value,validation_ap=rank_metrics(val_y,val_scores)["ap"],
        validation_f2=f2,threshold_from_validation=threshold,
        training_seconds=time.perf_counter()-started,
        train_selected_pr_auc=rank_metrics(train_y,train_scores)["pr_auc"],
        test_used_for_checkpoint_or_threshold=False,
        model_parameter_count=sum(p.numel() for p in model.parameters()),
        completed_utc=stamp(),status="ok")
    result["files"]={name:sha(dest/name) for name in
                     ["best_model.pt","train_validation_scores.npz","epoch_history.csv"]}
    write_json(complete,result)
    print(f"Completed {entry['repeat']}/5 {variant['method']}: "
          f"best validation PR-AUC {best_value:.4f}, epoch {best_epoch}/{epoch}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"nn_comparison_config.json")
    args=parser.parse_args()
    c=json.loads(args.config.read_text(encoding="utf-8"))
    assert c["selection_metric"]=="validation_pr_auc" and c["threshold_metric"]=="validation_f2"
    assert c["activation"]=="ReLU" and c["optimizer"]=="AdamW"
    out=ROOT/c["output"]
    for name in ["models","tables","figures","qa","code_snapshot"]:
        (out/name).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(c["num_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    ref=reference()
    plan_path=ROOT/c["frozen_splits"]
    plan=json.loads(plan_path.read_text(encoding="utf-8")); folder=plan_path.parent
    data=pd.read_csv(ROOT/c["input"])
    assert len(data)==9943 and data.is_fraud.sum()==94
    previous_manifest=json.loads((SOURCE/"chapter6_supplement/source_manifest.json").read_text(encoding="utf-8"))
    for rel,digest in previous_manifest["source_hashes"].items():
        assert sha(ROOT/rel)==digest,rel
    hashes={str(p.resolve()):sha(p) for p in [Path(__file__),args.config,plan_path,ROOT/c["input"],
                  ROOT/"credit_card_fraud_10k_experiment.py",SOURCE/"metrics_by_round.csv",
                  SOURCE/"test_predictions_by_round.csv"]}
    manifest=out/"run_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding="utf-8"))["source_hashes"]==hashes, \
            "Inputs/config/training code changed. Use a new output directory."
    else:
        write_json(manifest,dict(created_utc=stamp(),source_hashes=hashes,
            frozen_partitions=plan["index_file_hashes"],torch=torch.__version__,
            python=platform.python_version(),platform=platform.platform(),
            cuda_used=False,cpu_threads=c["num_threads"],config=c))
    for path in [Path(__file__),args.config,ROOT/"credit_card_fraud_10k_experiment.py"]:
        shutil.copy2(path,out/"code_snapshot"/path.name)
    training=[]
    for e in plan["rounds"]:
        indices={key:partition(data,e,key,plan,folder) for key in ["train","validation","test"]}
        assert not(set(indices["train"])&set(indices["validation"]) or
                   set(indices["train"])&set(indices["test"]) or set(indices["validation"])&set(indices["test"]))
        tr=data.iloc[indices["train"]]; va=data.iloc[indices["validation"]]
        assert max(tr.is_fraud.mean(),va.is_fraud.mean())<.01
        prep=ref.make_preprocessor()
        xt=prep.fit_transform(tr[ref.FEATURES]); xv=prep.transform(va[ref.FEATURES])
        round_dir=out/"models"/f"round_{e['repeat']:02d}"
        round_dir.mkdir(exist_ok=True)
        joblib.dump(prep,round_dir/"preprocessor.joblib")
        for j,variant in enumerate(c["variants"]):
            result=train_one(c,variant,e,xt,tr.is_fraud.to_numpy(),xv,va.is_fraud.to_numpy(),
                             round_dir/f"variant_{j:02d}",ref)
            training.append(result)
            write_json(out/"status.json",dict(phase="training",completed=len(training),
                expected=len(plan["rounds"])*len(c["variants"]),last_method=variant["method"],
                last_round=e["repeat"],test_scoring_started=False))
            csv(pd.DataFrame([{k:v for k,v in t.items() if k!="files"} for t in training]),
                out/"tables/training_summary.csv")
    assert len(training)==35 and all(t["status"]=="ok" for t in training)
    write_json(out/"all_checkpoints_frozen.json",dict(frozen_utc=stamp(),models=training,
        test_scoring_started=False))
    print("All 35 checkpoints and validation thresholds frozen. Starting test scoring.",flush=True)
    rows=[]; prediction_rows=[]
    for e in plan["rounds"]:
        indices=partition(data,e,"test",plan,folder); test=data.iloc[indices]
        old_test=read_csv(SOURCE/"test_predictions_by_round.csv")
        old_test=old_test[(old_test["repeat"]==e["repeat"])&(old_test.method=="RF-Plain")]
        np.testing.assert_array_equal(test.transaction_id,old_test.transaction_id)
        round_dir=out/"models"/f"round_{e['repeat']:02d}"
        prep=joblib.load(round_dir/"preprocessor.joblib")
        x=prep.transform(test[ref.FEATURES])
        for j,variant in enumerate(c["variants"]):
            dest=round_dir/f"variant_{j:02d}"
            info=json.loads((dest/"training_complete.json").read_text(encoding="utf-8"))
            checkpoint=torch.load(dest/"best_model.pt",map_location="cpu",weights_only=True)
            model=FraudMLP(checkpoint["input_size"],checkpoint["hidden_layers"],checkpoint["dropout"])
            model.load_state_dict(checkpoint["state_dict"])
            score=predict(model,x)
            threshold=info["threshold_from_validation"]
            result=ref.evaluate_scores(test.is_fraud.to_numpy(),score,threshold)
            fpr,tpr,_=roc_curve(test.is_fraud,score,drop_intermediate=False)
            result["recall_at_fpr_1pct_diagnostic"]=float(tpr[fpr<=.01].max())
            rows.append(dict(repeat=e["repeat"],split_seed=e["split_seed"],model_seed=e["model_seed"],
                method=variant["method"],rf_method=variant["rf_method"],family="MLP",
                threshold_from_validation=threshold,validation_f2_at_threshold=info["validation_f2"],
                validation_pr_auc=info["best_validation_pr_auc"],best_epoch=info["best_epoch"],
                epochs_run=info["epochs_run"],test_size=len(test),test_positive=int(test.is_fraud.sum()),
                test_prevalence=float(test.is_fraud.mean()),**result))
            frame=test.copy()
            frame.insert(0,"repeat",e["repeat"]); frame.insert(1,"method",variant["method"])
            frame=frame.rename(columns={"is_fraud":"y_true"})
            frame["y_score"]=score;frame["threshold"]=threshold
            frame["y_pred"]=(score>=threshold).astype(int)
            frame["outcome"]=np.select([
                frame.y_true.eq(1)&frame.y_pred.eq(1),frame.y_true.eq(0)&frame.y_pred.eq(1),
                frame.y_true.eq(0)&frame.y_pred.eq(0)],["TP","FP","TN"],default="FN")
            prediction_rows.append(frame)
    nn_results=pd.DataFrame(rows)
    predictions=pd.concat(prediction_rows,ignore_index=True)
    csv(nn_results,out/"tables/nn_metrics_by_round.csv")
    csv(predictions,out/"tables/nn_test_predictions_with_features.csv")
    csv(predictions[predictions.outcome.isin(["FP","FN"])],out/"tables/nn_fp_fn_all_occurrences.csv")
    original=read_csv(SOURCE/"metrics_by_round.csv")
    original["family"]=np.where(original.method.eq("Isolation Forest"),"Isolation Forest",
                               np.where(original.method.eq("EasyEnsemble"),"RF ensemble","RF"))
    combined=pd.concat([original,nn_results],ignore_index=True)
    csv(combined,out/"tables/all_metrics_by_round.csv")
    columns=["pr_auc","ap","roc_auc","accuracy","precision","recall","f1","f2","mcc","fpr"]
    summary=combined.groupby(["method","family"],sort=False).agg(
        **{f"{col}_{stat}":(col,stat) for col in columns for stat in ["mean","std"]}).reset_index()
    csv(summary,out/"tables/all_model_summary_mean_std.csv")
    paired=[]
    for v in c["variants"]:
        a=original[original.method.eq(v["rf_method"])].set_index("repeat")
        b=nn_results[nn_results.method.eq(v["method"])].set_index("repeat")
        for repeat in range(1,6):
            row=dict(repeat=repeat,strategy=v["strategy"],rf_method=v["rf_method"],nn_method=v["method"])
            for col in columns:
                row.update({f"rf_{col}":a.loc[repeat,col],f"nn_{col}":b.loc[repeat,col],
                            f"delta_{col}":b.loc[repeat,col]-a.loc[repeat,col]})
            paired.append(row)
    csv(pd.DataFrame(paired),out/"tables/paired_differences_by_round.csv")
    write_json(out/"status.json",dict(phase="completed",completed=35,expected=35,
        test_scoring_started=True,nn_prediction_rows=len(predictions),
        original_results_preserved=True,completed_utc=stamp()))
    print(summary[["method","pr_auc_mean","pr_auc_std","f2_mean"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
