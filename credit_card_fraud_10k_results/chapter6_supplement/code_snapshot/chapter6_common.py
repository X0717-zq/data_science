"""Shared paths, integrity checks and scoring for the chapter-six supplement."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "chapter6_supplement_config.json"
C = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SOURCE = ROOT / C["source_results"]
OUT = ROOT / C["output"]
METHODS = ["RF-Plain", "RF-Balanced", "SMOTE", "ADASYN", "NearMiss",
           "SMOTE+Tomek", "SMOTE+ENN", "EasyEnsemble", "Isolation Forest"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def reference():
    spec = importlib.util.spec_from_file_location("chapter6_original_experiment", ROOT / C["reference_script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.METHODS == METHODS
    return module


def scores(y, value):
    p, r, _ = precision_recall_curve(y, value)
    return dict(PR_AUC=float(auc(r, p)), AP=float(average_precision_score(y, value)),
                ROC_AUC=float(roc_auc_score(y, value)))


def initialize():
    for sub in ["tables", "figures", "case_analysis", "learning_curves", "code_snapshot", "qa"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    state_path = OUT / "source_manifest.json"
    sources = [p for p in SOURCE.iterdir() if p.is_file()]
    sources += [ROOT / C["reference_script"], ROOT / "fraudulent_card.csv", CONFIG_PATH]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sources}
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        assert existing["source_hashes"] == hashes, "Source data, original results, or supplement config changed."
    else:
        save_json(state_path, dict(created_utc=now(), source_hashes=hashes,
            original_results_preserved=True, scope="nine-method chapter-six supplement"))
    for p in [CONFIG_PATH, ROOT / "chapter6_common.py", ROOT / C["reference_script"]]:
        target = OUT / "code_snapshot" / p.name
        if target.exists():
            assert sha(target) == sha(p), f"Frozen shared source changed: {p.name}"
        else:
            shutil.copy2(p, target)


def markdown(frame):
    rows = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(str(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
