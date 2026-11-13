"""Nine methods x five nested training sizes x five original splits.

The original train_and_score function is reused without changing parameters.
Its last argument receives the original training subset, NOT the test set,
so it returns validation and resubstitution-training scores for this analysis.
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from chapter6_common import (C, ROOT, SOURCE, OUT, METHODS, initialize, reference,
                            sha, now, save_json, csv, read_csv, scores)


def prepare_splits(data, ref):
    folder = OUT / "learning_curves"
    plan_path = folder / "frozen_training_subsets.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for path, hashed in plan["index_file_hashes"].items():
            assert sha(folder / path) == hashed
        return plan
    predictions = read_csv(SOURCE / "test_predictions_by_round.csv")
    plan = dict(created_utc=now(), subset_seed_offset=C["subset_seed_offset"],
        fractions=C["learning_fractions"], nested_stratified_subsets=True,
        test_used_for_learning=False, rounds=[], index_file_hashes={})
    all_counts = []
    for repeat, split_seed in enumerate(ref.REPEAT_SEEDS, 1):
        train, rest = train_test_split(np.arange(len(data)), test_size=.30,
                                      stratify=data.is_fraud, random_state=split_seed)
        val, test = train_test_split(rest, test_size=.5, stratify=data.is_fraud.iloc[rest],
                                    random_state=split_seed + 1000)
        previous = predictions[(predictions['repeat'] == repeat) & (predictions.method == 'RF-Plain')]
        np.testing.assert_array_equal(data.transaction_id.iloc[test], previous.transaction_id)
        assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
        round_dir = folder / f"round_{repeat:02d}"
        round_dir.mkdir(exist_ok=True)
        entry = dict(repeat=repeat, split_seed=split_seed, model_seed=split_seed + repeat,
                     subset_seed=split_seed + C["subset_seed_offset"], partitions={}, subsets=[])
        for name, idx in [("train", train), ("validation", val), ("test", test)]:
            f = round_dir / f"{name}_indices.csv"
            csv(pd.DataFrame(dict(row_position=idx, transaction_id=data.transaction_id.iloc[idx].to_numpy(),
                                  is_fraud=data.is_fraud.iloc[idx].to_numpy())), f)
            rel = f.relative_to(folder).as_posix()
            entry["partitions"][name] = rel
            plan["index_file_hashes"][rel] = sha(f)
        rng = np.random.RandomState(entry["subset_seed"])
        positives = rng.permutation(train[data.is_fraud.iloc[train].to_numpy() == 1])
        negatives = rng.permutation(train[data.is_fraud.iloc[train].to_numpy() == 0])
        last = set()
        for fraction in C["learning_fractions"]:
            n = int(round(len(train) * fraction))
            pos = int(round(n * len(positives) / len(train)))
            chosen = set(np.r_[positives[:pos], negatives[:n-pos]].tolist())
            idx = np.array([x for x in train if x in chosen], dtype=int)
            assert len(idx) == n and last <= chosen and pos >= 6
            assert 0 < data.is_fraud.iloc[idx].mean() < .01
            if fraction == 1.0:
                np.testing.assert_array_equal(idx, train)
            tag = f"fraction_{int(round(fraction*100)):03d}"
            f = round_dir / f"{tag}_indices.csv"
            csv(pd.DataFrame(dict(row_position=idx, transaction_id=data.transaction_id.iloc[idx].to_numpy(),
                                  is_fraud=data.is_fraud.iloc[idx].to_numpy())), f)
            rel = f.relative_to(folder).as_posix()
            plan["index_file_hashes"][rel] = sha(f)
            entry["subsets"].append(dict(fraction=fraction, tag=tag, file=rel, train_n=n, train_positive=pos))
            all_counts.append(dict(repeat=repeat, fraction=fraction, train_n=n, train_positive=pos,
                                   train_negative=n-pos, positive_rate=pos/n, validation_n=len(val),
                                   validation_positive=int(data.is_fraud.iloc[val].sum())))
            last = chosen
        plan["rounds"].append(entry)
    save_json(plan_path, plan)
    csv(pd.DataFrame(all_counts), folder / "training_size_counts.csv")
    return plan


def main():
    initialize()
    folder = OUT / "learning_curves"
    copy = OUT / "code_snapshot" / Path(__file__).name
    if copy.exists():
        assert sha(copy) == sha(__file__), "Learning code changed after freezing."
    else:
        shutil.copy2(__file__, copy)
    ref = reference()
    # Read the identical input representation used by the reference run.
    data = pd.read_csv(ROOT / "fraudulent_card.csv")
    plan = prepare_splits(data, ref)
    existing_metrics = read_csv(SOURCE / "metrics_by_round.csv").set_index(["repeat", "method"])
    jobs, matches = [], []
    started = now()
    for e in plan["rounds"]:
        val_idx = read_csv(folder / e["partitions"]["validation"]).row_position.to_numpy(dtype=int)
        val = data.iloc[val_idx]
        for subset in e["subsets"]:
            train_idx = read_csv(folder / subset["file"]).row_position.to_numpy(dtype=int)
            train = data.iloc[train_idx]
            dest = folder / f"round_{e['repeat']:02d}" / subset["tag"]
            dest.mkdir(exist_ok=True)
            for method_index, method in enumerate(METHODS):
                basename = f"model_{method_index:02d}"
                result_file = dest / f"{basename}.json"
                if result_file.exists():
                    row = json.loads(result_file.read_text(encoding="utf-8"))
                    if row["status"] == "ok":
                        assert sha(dest / f"{basename}_scores.npz") == row["scores_sha256"]
                        jobs.append(row)
                        continue
                row = dict(repeat=e["repeat"], method=method, fraction=subset["fraction"],
                    train_n=len(train), train_positive=int(train.is_fraud.sum()),
                    validation_n=len(val), validation_positive=int(val.is_fraud.sum()),
                    model_seed=e["model_seed"], subset_file=subset["file"],
                    validation_file=e["partitions"]["validation"])
                tic = time.perf_counter()
                try:
                    val_score, train_score = ref.train_and_score(method, train[ref.FEATURES], train.is_fraud,
                        val[ref.FEATURES], train[ref.FEATURES], e["model_seed"])
                    assert len(train_score) == len(train) and len(val_score) == len(val)
                    assert np.isfinite(train_score).all() and np.isfinite(val_score).all()
                    for partition, y, score in [("train", train.is_fraud, train_score),
                                                ("validation", val.is_fraud, val_score)]:
                        row.update({f"{partition}_{k}": v for k,v in scores(y, score).items()})
                    threshold, f2 = ref.select_f2_threshold(val.is_fraud.to_numpy(), val_score)
                    row.update(validation_F2_threshold=threshold, validation_F2=f2,
                               fit_and_score_seconds=time.perf_counter()-tic, status="ok")
                    score_file = dest / f"{basename}_scores.npz"
                    np.savez_compressed(score_file, train_score=train_score, validation_score=val_score,
                                        train_row_position=train_idx, validation_row_position=val_idx)
                    row["scores_file"] = score_file.relative_to(folder).as_posix()
                    row["scores_sha256"] = sha(score_file)
                except Exception as ex:
                    row.update(status="failed", error_type=type(ex).__name__, error=str(ex),
                               fit_and_score_seconds=time.perf_counter()-tic)
                save_json(result_file, row)
                jobs.append(row)
                print(f"Round {e['repeat']}/5 | {subset['fraction']:.0%} | {method}: "
                      f"{row['status']} | validation PR-AUC={row.get('validation_PR_AUC', float('nan')):.4f}", flush=True)
                csv(pd.DataFrame(jobs), folder / "learning_curve_metrics_partial.csv")
    results = pd.DataFrame(jobs)
    csv(results, folder / "learning_curve_metrics.csv")
    summary = []
    for (method, fraction), block in results.groupby(["method", "fraction"], sort=False):
        ok = block[block.status == "ok"]
        row = dict(method=method, fraction=fraction, train_n=int(block.train_n.iloc[0]),
                   successful_rounds=len(ok), expected_rounds=len(plan["rounds"]))
        for partition in ["train", "validation"]:
            for m in C["learning_score_metrics"]:
                col = f"{partition}_{m}"
                row[f"{col}_mean"] = ok[col].mean()
                row[f"{col}_std"] = ok[col].std(ddof=1)
        summary.append(row)
    csv(pd.DataFrame(summary), folder / "learning_curve_summary.csv")
    for row in jobs:
        if row["fraction"] == 1.0 and row["status"] == "ok":
            old = existing_metrics.loc[(row["repeat"], row["method"])]
            matches.append(dict(repeat=row["repeat"], method=row["method"],
                old_threshold=old.threshold_from_validation, replay_threshold=row["validation_F2_threshold"],
                old_validation_F2=old.validation_f2_at_threshold, replay_validation_F2=row["validation_F2"],
                threshold_matches=bool(np.isclose(old.threshold_from_validation,row["validation_F2_threshold"],rtol=0,atol=1e-12)),
                validation_F2_matches=bool(np.isclose(old.validation_f2_at_threshold,row["validation_F2"],rtol=0,atol=1e-12))))
    csv(pd.DataFrame(matches), folder / "full_size_validation_replay_check.csv")
    save_json(folder / "status.json", dict(started_utc=started, completed_utc=now(),
        expected_jobs=225, successful_jobs=int((results.status == "ok").sum()),
        failed_jobs=int((results.status != "ok").sum()), test_used=False,
        full_size_validation_replay_mismatches=int(sum(not(r['threshold_matches'] and r['validation_F2_matches']) for r in matches)),
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','imbalanced-learn','matplotlib']},
        python=platform.python_version(), platform=platform.platform()))
    print(f"Learning curves complete: {(results.status=='ok').sum()}/225 successful fits.", flush=True)


if __name__ == "__main__":
    main()
