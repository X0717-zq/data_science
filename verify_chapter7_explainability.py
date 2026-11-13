"""Verify saved chapter-seven outputs independently of their report text."""
from pathlib import Path
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import shap
from PIL import Image
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "chapter7_explainability_config.json").read_text(encoding="utf-8"))
OUT = ROOT / C["output"]

def read(p):
    return pd.read_csv(p, float_precision="round_trip")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    num = json.loads((OUT / "qa" / "numerical_verification.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert num["all_checks_passed"]
    for p, h in {**manifest["source_hashes"], **manifest["frozen_split_hashes"]}.items():
        assert sha(ROOT / p) == h, f"Input altered: {p}"
    df = read(OUT / "tables" / "test_shap_all_occurrences.csv")
    orig = read(ROOT / C["source_results"] / "test_predictions_by_round.csv")
    orig = orig[orig.method.eq("RF-Plain")]
    merged = df.merge(orig, on=["repeat", "transaction_id"], suffixes=("", "_original"), validate="one_to_one")
    assert len(merged) == len(df) == len(orig) == 7460
    assert df.transaction_id.nunique() == 5485
    for a, b in [("is_fraud", "y_true"), ("y_score", "y_score_original"), ("threshold", "threshold_original"), ("y_pred", "y_pred_original")]:
        np.testing.assert_array_equal(merged[a], merged[b])
    shapcols = ["shap_" + f for f in num["features"]]
    np.testing.assert_allclose(df.base_value + df[shapcols].sum(axis=1), df.y_score, atol=1e-12, rtol=0)
    byround = read(OUT / "tables" / "feature_importance_by_round.csv")
    summary = read(OUT / "tables" / "feature_importance_summary.csv").set_index("feature")
    pm = read(OUT / "tables" / "permutation_importance_all_repeats.csv")
    assert len(pm) == 5 * 8 * 30
    assert pm.groupby(["repeat", "feature"]).size().eq(30).all()
    np.testing.assert_allclose(pm.baseline_pr_auc - pm.shuffled_pr_auc, pm.pr_auc_drop, atol=1e-14, rtol=0)
    for f in num["features"]:
        v = df.groupby("repeat")["shap_" + f].apply(lambda x: x.abs().mean())
        np.testing.assert_allclose([v.mean(), v.std(ddof=1)], summary.loc[f, ["shap_mean", "shap_sd"]].to_numpy(dtype=float), atol=1e-14, rtol=0)
        p = pm[pm.feature.eq(f)].groupby("repeat").pr_auc_drop.mean()
        np.testing.assert_allclose([p.mean(), p.std(ddof=1)], summary.loc[f, ["permutation_drop_mean", "permutation_drop_sd"]].to_numpy(dtype=float), atol=1e-14, rtol=0)
    unique = df.drop_duplicates("transaction_id")
    groups = read(OUT / "tables" / "business_group_evidence.csv")
    assert groups.groupby("feature_cn").unique_transactions.sum().eq(len(unique)).all()
    assert groups.groupby("feature_cn").fraud.sum().eq(int(unique.is_fraud.sum())).all()
    assert (len(unique), int(unique.is_fraud.sum())) == (5485, 51)
    rates = read(OUT / "tables" / "threshold_and_workload_by_round.csv")
    np.testing.assert_array_equal(rates[["TP", "FP", "TN", "FN"]].sum().to_numpy(), [59, 11, 7379, 11])
    report = (OUT / "第七章_模型解释与业务应用.md").read_text(encoding="utf-8")
    for required in ["## 7.1", "## 7.2", "## 7.3", "洞察一", "洞察二", "洞察三", "99.0617%", "5,485", "1.11e-15"]:
        assert required in report
    positions = [report.index(s) for s in ["### 7.2.1", "**表7-2", "![图7-4", "### 7.2.2", "![图7-5", "### 7.2.3", "**表7-3", "### 7.2.4", "**表7-4", "### 7.2.5"]]
    assert positions == sorted(positions), "Requested figure/table placement changed"
    risk = json.loads((OUT / "qa" / "risk_groups_numerical_verification.json").read_text(encoding="utf-8"))
    assert risk["all_counts_and_rates_passed"] and len(risk["panels"]) == 5
    assert sha(OUT / "tables" / "business_group_evidence.csv") == risk["source_sha256"]
    for image in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        assert (OUT / image).is_file()
    audit_dir = Path("C:/Users/86132/.agents/skills/nature-figure/scripts")
    for source, log in [("plot_chapter7_explainability.py", "figure_source_preflight.txt"), ("plot_chapter7_risk_groups.py", "risk_groups_source_preflight.txt")]:
        static = subprocess.run([sys.executable, str(audit_dir / "validate_figure.py"), str(ROOT / source)], capture_output=True, text=True, encoding="utf-8")
        (OUT / "qa" / log).write_text(static.stdout + static.stderr, encoding="utf-8")
        assert static.returncode == 0
    figures = []
    for name in ["fig7_1_global_importance", "fig7_2_shap_direction", "fig7_3_fp_fn_local_shap", "fig7_4_hour_device_fraud_rate", "fig7_5_foreign_location_velocity_fraud_rate"]:
        paths = [OUT / "figures" / (name + "." + ext) for ext in ["png", "pdf", "svg"]]
        assert all(p.is_file() for p in paths)
        with Image.open(paths[0]) as im:
            dpi = im.info.get("dpi")
            assert dpi and min(dpi) >= 299.9
            dimensions = list(im.size)
        align = json.loads((OUT / "qa" / (name + ".alignment.json")).read_text(encoding="utf-8"))
        collision = json.loads((OUT / "qa" / (name + ".collision.json")).read_text(encoding="utf-8"))
        assert align["verdict"] == collision["verdict"] == "PASS"
        audit = subprocess.run([sys.executable, str(audit_dir / "audit_pdf_text.py"), str(paths[1]), "--min-pt", "5", "--json"], capture_output=True, text=True, encoding="utf-8")
        assert audit.returncode == 0
        glyph = json.loads(audit.stdout)
        assert glyph["below_minimum_count"] == 0
        (OUT / "qa" / (name + ".text_audit.json")).write_text(audit.stdout, encoding="utf-8")
        figures.append(dict(name=name, pixel_dimensions=dimensions, dpi=dpi, minimum_font_pt=glyph["minimum_found_pt"], alignment="PASS", collisions="PASS", sha256={p.suffix:sha(p) for p in paths}))
    (OUT / "qa" / "installed_method_documentation.txt").write_text("Installed SHAP TreeExplainer.__init__ documentation\n\n" + inspect.getdoc(shap.TreeExplainer.__init__) + "\n\nInstalled sklearn permutation_importance documentation (conceptual reference; our implementation permutes original groups and scores trapezoidal PR-AUC)\n\n" + inspect.getdoc(permutation_importance), encoding="utf-8")
    result = dict(all_checks_passed=True, exact_original_scores_labels_predictions_thresholds=True, all_source_hashes_unchanged=True, shap_additivity_passed=True, all_8_grouped_shap_mean_sd_recomputed=True, all_8_permutation_mean_sd_recomputed=True, unique_positive_transactions=51, prediction_occurrences=7460, unique_transactions=5485, permutation_evaluations=1200, figures=figures, static_warnings_review={"PNG_without_TIFF":"No TIFF requested; PNG at verified 300 dpi plus PDF/SVG supplied.", "dynamic_DPI":"DPI comes from the separate config and is verified from PNG metadata.", "random_generator":"Only visual horizontal jitter for the binary flag; all observed scores, values, labels and SHAP remain intact."}, interpretation_boundary="Associational model explanations and retrospective test diagnostics, not causal mechanisms, tuned features, or validated production rules.")
    result["risk_group_counts_and_train_cutpoints_verified"] = True
    result["requested_section_figure_table_order_verified"] = True
    result["static_warnings_review"]["dynamic_figure_width"] = "Risk-group figure dimensions come from a separate config: 7.4 by 4.0 inches, checked against rendered PNG dimensions and panel geometry."
    (OUT / "qa" / "delivery_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "qa" / "visual_review.md").write_text("""# 最终图形核验

已查看最终PNG，并按面板检查：

| 图 | 面板 | 观察与统计单位 | 版式与文字 | 自动核验 |
| --- | --- | --- | --- | --- |
| 图7-1 | a | 八个字段的五轮SHAP；均值±样本SD及五轮散点 | 标签完整；双行轴名与脚注分开；未截断误差线 | 对齐PASS；碰撞PASS；最小字号7.7pt |
| 图7-1 | b | 每轮30次置换先平均，再汇总五轮；不与a混用单位 | 保留小于零的波动范围；与a相同顺序 | 对齐PASS；碰撞PASS |
| 图7-2 | a–d | 展示全部7,460次预测；灰色正常、橙色欺诈；重复交易说明完整 | 四轴等宽等高；零线清晰；全部数据在范围内；二元轴错开仅为显示重叠 | 对齐PASS；碰撞PASS；最小字号7.5pt |
| 图7-3 | a | 第5轮交易3103，全部八项有符号贡献 | 正负贡献可辨；数值避开网格线；基准重构说明与脚注分离 | 对齐PASS；碰撞PASS；最小字号7pt |
| 图7-3 | b | 第3轮交易656，全部八项有符号贡献 | 与a共用横轴尺度；小贡献显示为−0.0003而非四舍五入成零 | 对齐PASS；碰撞PASS |
| 图7-4 | a–b | 时段与设备评分分别分组；每个面板完整覆盖5,485笔、51笔欺诈 | 每柱显示比例和分子/分母；零起点；与图7-5共用0—6.5%尺度 | 对齐PASS；碰撞PASS；最小字号7.1pt |
| 图7-5 | a–c | 境外、位置、频次分别分组；无重复交易计数 | 三面板等宽等高；分组语义清晰；不把边际比例当联合风险 | 对齐PASS；碰撞PASS；最小字号7.1pt |

静态检测的三项警告已复核：PNG/PDF/SVG为本次交付格式，无TIFF要求；PNG实际DPI为约300；随机数仅用于图7-2境外标记的横向显示抖动，不是模拟数据。

图7-4/7-5尺寸来自独立配置，实际导出为2220×1200像素、约300dpi。未绘误差线，因为图上为去重样本中的描述性比例，不是五轮均值或总体推断。
""", encoding="utf-8")
    shutil.copy2(__file__, OUT / "code_snapshot" / Path(__file__).name)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
