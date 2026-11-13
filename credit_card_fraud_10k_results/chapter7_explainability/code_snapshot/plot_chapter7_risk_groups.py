"""Render descriptive fraud rates from the deduplicated test-set union.

Figure contract: a quantitative grid contrasts higher-risk clues and their
complements for each field. Rates are observed sample proportions, not five-run
means or model precision. Every bar shows its exact numerator/denominator;
no row is sampled, no synthetic data or inferential interval is introduced.
All panels and both figures use one linear percentage axis starting at zero.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from figure_alignment_helper import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "chapter7_explainability_config.json").read_text(encoding="utf-8"))
OUT = ROOT / C["output"]
CONFIG = ROOT / "chapter7_risk_groups_config.json"
G = json.loads(CONFIG.read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [C["figure"]["font"], "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": False})

def export(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=OUT / "qa" / (name + ".alignment.json"), overlay_svg=OUT / "qa" / (name + ".alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    if "png" in C["figure"]["formats"]:
        fig.savefig(OUT / "figures" / (name + ".png"), dpi=C["figure"]["dpi"], facecolor="white")
    if "pdf" in C["figure"]["formats"]:
        fig.savefig(OUT / "figures" / (name + ".pdf"), facecolor="white")
    if "svg" in C["figure"]["formats"]:
        fig.savefig(OUT / "figures" / (name + ".svg"), facecolor="white")
    plt.close(fig)

def read(path):
    return pd.read_csv(path, float_precision="round_trip")

def main():
    source = OUT / "tables" / "business_group_evidence.csv"
    groups = read(source)
    tests = read(OUT / "tables" / "test_shap_all_occurrences.csv")
    features = [p["field"] for f in G["figures"] for p in f["panels"]]
    assert tests.groupby("transaction_id")[features + ["is_fraud"]].nunique().le(1).all().all()
    unique = tests.drop_duplicates("transaction_id")
    assert (len(unique), int(unique.is_fraud.sum())) == (5485, 51)
    raw = read(ROOT / C["dataset"])
    train_idx = read(ROOT / C["frozen_splits"] / "round_01" / "train_indices.csv")
    train = raw.iloc[train_idx.row_position.to_numpy()]
    checked = []
    for figspec in G["figures"]:
        panels = figspec["panels"]
        fig, axes = plt.subplots(1, len(panels), figsize=G["figure_size_inches"], sharey=True, squeeze=False)
        fig.subplots_adjust(left=.10, right=.97, bottom=.25, top=.77, wspace=.28)
        source_rows = []
        for letter, ax, panel in zip("abc", axes.flat, panels):
            selected = groups[groups.feature_cn.eq(panel["feature_cn"])].set_index("group").loc[panel["groups"]].reset_index()
            if "train_quantile" in panel:
                assert np.isclose(train[panel["field"]].quantile(panel["train_quantile"]), panel["threshold"])
            first = getattr(unique[panel["field"]], panel["first_operator"])(panel["threshold"])
            assert selected.unique_transactions.sum() == len(unique)
            for idx, mask in enumerate([first, ~first]):
                row = selected.iloc[idx]
                assert row.unique_transactions == int(mask.sum())
                assert row.fraud == int(unique.loc[mask, "is_fraud"].sum())
                np.testing.assert_allclose(row.fraud_rate, row.fraud / row.unique_transactions, atol=1e-15, rtol=0)
            heights = selected.fraud_rate.to_numpy() * 100
            ax.bar([0, 1], heights, width=.54, color=G["colors"], zorder=3)
            ax.set_ylim(0, G["y_limit_percent"])
            ax.set_yticks(G["y_ticks_percent"])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.set_xlim(-.65, 1.65)
            ax.set_xticks([0, 1], panel["display_labels"])
            ax.set_title(f"{letter}  {panel['feature_cn']}", loc="left", pad=12)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            # No grid through the numerical labels above bars.
            for x, (row, h) in enumerate(zip(selected.itertuples(), heights)):
                ax.text(x, h + .20, f"{h:.2f}%\n{row.fraud:,}/{row.unique_transactions:,}", ha="center", va="bottom", fontsize=8.2, linespacing=1.6)
            selected.insert(0, "figure", figspec["filename"])
            selected["display_label"] = panel["display_labels"]
            source_rows.append(selected)
            checked.append(dict(figure=figspec["filename"], field=panel["field"], unique_transactions=int(selected.unique_transactions.sum()), fraud=int(selected.fraud.sum()), counts_match=True, train_cutoff_verified="train_quantile" in panel))
        axes.flat[0].set_ylabel("观察欺诈率（欺诈笔数 / 该组交易数）", labelpad=10)
        fig.suptitle(figspec["title"], fontsize=12, y=.95)
        fig.text(.10, .12, "五轮测试并集按交易编号去重：5,485笔，其中欺诈51笔；柱上为比例及欺诈笔数/组内总数。", fontsize=7.1)
        fig.text(.10, .055, figspec["footer"], fontsize=7.1)
        export(fig, figspec["filename"])
        pd.concat(source_rows, ignore_index=True).to_csv(OUT / "tables" / (figspec["filename"] + "_source.csv"), index=False, encoding="utf-8-sig")
        print(f"Exported {figspec['filename']}: PNG, PDF, SVG", flush=True)
    for p in [Path(__file__), CONFIG]:
        shutil.copy2(p, OUT / "code_snapshot" / p.name)
    verification = dict(all_counts_and_rates_passed=True, unique_transactions=len(unique), unique_fraud=int(unique.is_fraud.sum()), source_file=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), configuration=G, panels=checked)
    (OUT / "qa" / "risk_groups_numerical_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "qa" / "risk_groups_figure_contract.md").write_text("""# 图7-4与图7-5绘图约定

- 图7-4问题：交易时段和设备评分两类分组的观察欺诈率是否不同？两个面板各呈现一个字段的完整互补分组。
- 图7-5问题：境外标记、位置不匹配、24小时交易频次分别提供哪些风险线索？三个面板分别检视一个字段，不能当作联合条件统计。
- 统计单位：五轮测试并集中的不同交易，5,485笔、其中51笔欺诈；每个面板独立覆盖全体5,485笔。
- 图上为观察欺诈比例及精确分子/分母，不是模型Precision，也不是五轮均值。无误差线：此次仅描述既有样本，不假设交易间独立或进行总体推断。
- 数据：与表7-2同源，重新按逐笔真实标签核对十组计数，所有数值分界重新与第一轮训练集四分位核对。不删除观测。
- 架构：等宽并列定量面板；同一0—6.5%纵轴，零起点；每面板高风险线索为橙色，对照为蓝色，标签完整，不靠颜色单独传递含义。
- 导出：沿用Python/matplotlib、Microsoft YaHei、300dpi PNG、PDF及可编辑SVG。执行严格对齐门禁，并另跑PDF文本和碰撞核验。
""", encoding="utf-8")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
