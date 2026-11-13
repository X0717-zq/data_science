"""Publication figures and source tables for the frozen RF explanation."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_alignment_helper import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "chapter7_explainability_config.json").read_text(encoding="utf-8"))
OUT = ROOT / C["output"]
LABELS = {"amount": "交易金额", "transaction_hour": "交易时段", "foreign_transaction": "境外交易标记", "location_mismatch": "位置不匹配", "device_trust_score": "设备信任评分", "velocity_last_24h": "24小时交易次数", "cardholder_age": "持卡人年龄", "merchant_category": "商户类别"}
BLUE, ORANGE, GRAY = "#247BA0", "#CC5A36", "#A3ACB4"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [C["figure"]["font"], "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": False})

def read(name):
    return pd.read_csv(OUT / "tables" / name, float_precision="round_trip")

def write(df, name):
    df.to_csv(OUT / "tables" / name, index=False, encoding="utf-8-sig")

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

def global_importance():
    s = read("feature_importance_summary.csv").sort_values("shap_rank")
    br = read("feature_importance_by_round.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.5), sharey=True)
    fig.subplots_adjust(left=.20, right=.97, bottom=.27, top=.81, wspace=.30)
    y = np.arange(len(s))
    for ax, mean, sd, roundcol, title, xlabel in zip(axes, ["shap_mean", "permutation_drop_mean"], ["shap_sd", "permutation_drop_sd"], ["mean_abs_shap", "permutation_pr_auc_drop"], ["a  全局 SHAP 贡献", "b  特征置换后的性能损失"], ["平均 |SHAP|\n（模型分数单位）", "PR-AUC 下降量\n（原始值 − 置换后）"]):
        ax.barh(y, s[mean], height=.58, color=BLUE, alpha=.22, zorder=1)
        for i, row in enumerate(s.itertuples()):
            vals = br[br.feature.eq(row.feature)].sort_values("repeat")[roundcol].to_numpy()
            ax.scatter(vals, i + np.linspace(-.17, .17, 5), c="#717B85", s=13, edgecolors="none", zorder=3)
        ax.errorbar(s[mean], y, xerr=s[sd], fmt="D", markersize=3.5, color=BLUE, elinewidth=1.1, capsize=3, zorder=4)
        ax.set_title(title, loc="left", pad=14)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y, [LABELS[f] for f in s.feature])
        ax.axvline(0, c="#9CA3AA", lw=.7)
        ax.grid(axis="x", color="#E5E9ED", lw=.6)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=.12)
    axes[0].invert_yaxis()
    fig.suptitle("RF-Plain 特征重要性：五轮一致性与性能依赖", y=.97, fontsize=12)
    fig.text(.20, .075, "菱形及误差线：五轮均值 ± 样本标准差；灰点：各轮结果。", fontsize=7.7)
    fig.text(.20, .035, "每轮测试 1,492 笔；每个原始字段置换 30 次；商户类别按整组处理。", fontsize=7.7)
    export(fig, "fig7_1_global_importance")

def directions():
    df = read("test_shap_all_occurrences.csv")
    features = ["transaction_hour", "device_trust_score", "foreign_transaction", "velocity_last_24h"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), sharey=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.15, top=.83, wspace=.23, hspace=.49)
    for letter, f, ax in zip("abcd", features, axes.flat):
        x = df[f].to_numpy(dtype=float)
        if f == "foreign_transaction":
            x = x + np.random.default_rng(C["figure"]["seed"]).uniform(-.075, .075, len(x))
        for value, color, size, alpha, marker in [(0, GRAY, 6, .25, "o"), (1, ORANGE, 13, .65, "x")]:
            mask = df.is_fraud.eq(value).to_numpy()
            ax.scatter(x[mask], df.loc[mask, "shap_" + f], c=color, s=size, alpha=alpha, marker=marker, linewidths=.7 if value else 0, rasterized=True)
        ax.axhline(0, color="#6D7782", linestyle="--", lw=.7, zorder=0)
        ax.set_title(f"{letter}  {LABELS[f]}", loc="left", pad=9)
        ax.set_xlabel(LABELS[f] + ("（0/1）" if f == "foreign_transaction" else ""))
        ax.grid(axis="y", color="#E9EDF0", lw=.6)
        ax.set_axisbelow(True)
        if f == "foreign_transaction":
            ax.set_xticks([0, 1], ["0：否", "1：是"])
            ax.set_xlim(-.17, 1.17)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP 值（模型分数单位）")
    fig.suptitle("关键字段如何推高或压低欺诈分数", fontsize=12, y=.97)
    handles = [Line2D([], [], ls="", marker="o", color=GRAY, markersize=5, label="正常：7,390 次"), Line2D([], [], ls="", marker="x", color=ORANGE, markersize=5, label="欺诈：70 次")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .93), ncol=2, fontsize=8)
    fig.text(.12, .060, "全部五轮测试预测共 7,460 次，涉及 5,485 笔不同交易；重复出现不视为独立样本。", fontsize=7.5)
    fig.text(.12, .025, "0 上方为提高分数，0 下方为降低分数；境外标记仅横向轻微错开以展示重叠点。", fontsize=7.5)
    export(fig, "fig7_2_shap_direction")

def local_cases():
    df = read("selected_fp_fn_shap_cases.csv")
    order = read("feature_importance_summary.csv").sort_values("shap_rank").feature.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.20, right=.97, bottom=.23, top=.75, wspace=.28)
    for letter, ax, row in zip("ab", axes, df.itertuples(index=False)):
        vals = np.array([getattr(row, "shap_" + f) for f in order])
        y = np.arange(len(order))
        ax.barh(y, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=.62)
        ax.axvline(0, color="#6D7782", lw=.8)
        ax.set_yticks(y, [LABELS[f] for f in order])
        ax.set_xlim(-.09, .15)
        ax.set_xticks([-.05, 0, .05, .10])
        # No grid behind direct value labels: keep their entire area unobstructed.
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        name = "误报 FP" if row.outcome == "FP" else "漏报 FN"
        ax.set_title(f"{letter}  {name} · 交易 {row.transaction_id}\n第 {row.repeat} 轮；分数 {row.y_score:.3f} / 阈值 {row.threshold:.3f}", loc="left", fontsize=8.6, pad=12, linespacing=1.8)
        ax.set_xlabel("SHAP 贡献（模型分数单位）")
        for k, v in enumerate(vals):
            label = f"{v:+.4f}" if abs(v) < .001 else f"{v:+.3f}"
            ax.text(v + (.003 if v >= 0 else -.003), k, label, va="center", ha="left" if v >= 0 else "right", fontsize=7)
        equation = f"基准 {row.base_value:.4f} + 贡献和 {vals.sum():.4f}\n= 模型分数 {row.y_score:.4f}"
        ax.text(.0, -.21, equation, transform=ax.transAxes, ha="left", va="top", fontsize=7.7, linespacing=1.6)
    axes[0].invert_yaxis()
    fig.suptitle("误报与漏报：同一组特征如何形成不同判断", fontsize=12, y=.96)
    handles = [Line2D([], [], color=ORANGE, lw=5, label="提高分数"), Line2D([], [], color=BLUE, lw=5, label="降低分数")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .91), ncol=2, fontsize=8)
    fig.text(.20, .025, "展示全部 8 个原始字段；贡献解释模型判断，不解释真实欺诈的因果机制。", fontsize=7.5)
    export(fig, "fig7_3_fp_fn_local_shap")

def business_tables():
    data = read("test_shap_all_occurrences.csv")
    unique = data.drop_duplicates("transaction_id")
    # Prespecified round-one training quartiles already recorded in the profile table.
    groups = [("交易时段", "≤6时", unique.transaction_hour.le(6)), ("交易时段", ">6时", unique.transaction_hour.gt(6)), ("设备信任评分", "≤44", unique.device_trust_score.le(44)), ("设备信任评分", ">44", unique.device_trust_score.gt(44)), ("境外交易标记", "1", unique.foreign_transaction.eq(1)), ("境外交易标记", "0", unique.foreign_transaction.eq(0)), ("位置不匹配", "1", unique.location_mismatch.eq(1)), ("位置不匹配", "0", unique.location_mismatch.eq(0)), ("24小时交易次数", ">3", unique.velocity_last_24h.gt(3)), ("24小时交易次数", "≤3", unique.velocity_last_24h.le(3))]
    rows = []
    for feature, name, mask in groups:
        part = unique[mask]
        rows.append(dict(feature_cn=feature, group=name, unique_transactions=len(part), fraud=int(part.is_fraud.sum()), normal=int(part.is_fraud.eq(0).sum()), fraud_rate=float(part.is_fraud.mean())))
    write(pd.DataFrame(rows), "business_group_evidence.csv")
    counts = data.groupby("repeat").apply(lambda p: pd.Series({"threshold": p.threshold.iloc[0], "test_n": len(p), "positive": int(p.is_fraud.sum()), "TP": int(p.outcome.eq("TP").sum()), "FP": int(p.outcome.eq("FP").sum()), "FN": int(p.outcome.eq("FN").sum()), "TN": int(p.outcome.eq("TN").sum()), "alerts": int(p.y_pred.sum()), "recall": p.loc[p.is_fraud.eq(1), "y_pred"].mean(), "precision": p.loc[p.y_pred.eq(1), "is_fraud"].mean()}), include_groups=False).reset_index()
    write(counts, "threshold_and_workload_by_round.csv")
    source = ROOT / C["source_results"]
    m = pd.read_csv(source / "metrics_by_round.csv")
    s = pd.read_csv(source / "summary_mean_std.csv")
    pooled = m.groupby("method", as_index=False)[["tp", "fp", "tn", "fn"]].sum()
    pooled["prediction_occurrences"] = pooled[["tp", "fp", "tn", "fn"]].sum(axis=1)
    pooled["alert_occurrences"] = pooled.tp + pooled.fp
    pooled["pooled_precision"] = pooled.tp / pooled.alert_occurrences
    pooled["pooled_recall"] = pooled.tp / (pooled.tp + pooled.fn)
    pooled["pooled_fpr"] = pooled.fp / (pooled.fp + pooled.tn)
    pooled = s[["method", "pr_auc_mean", "pr_auc_std"]].merge(pooled, on="method", validate="one_to_one")
    write(pooled, "business_model_comparison.csv")
    print(pd.DataFrame(rows).to_string(index=False))

def main():
    shutil.copy2(__file__, OUT / "code_snapshot" / Path(__file__).name)
    business_tables()
    global_importance()
    directions()
    local_cases()
    (OUT / "qa" / "figure_contract.md").write_text("""# 第七章图表证据约定

- 后端：Python / matplotlib；全中文报告，非指定期刊投稿。
- 核心问题：本次 RF-Plain 的分数及排序主要依赖哪些信息，这些依赖在哪里出现误判？
- 图7-1：a 为平均绝对 SHAP 的模型贡献；b 为原始字段分组置换对测试 PR-AUC 的影响。两者度量不同，不可相加，也不解释为因果贡献率。
- 图7-2：在全部测试预测中展示时段、设备评分、境外标记、频次与 SHAP 方向；仅境外二元横轴有固定种子视觉抖动。不删除、抽稀或裁剪观测。
- 图7-3：复用此前已选定的一例 FP 和一例 FN，展示八个原始字段的全部有符号贡献，基准加贡献和核对原始分数。
- 样本单位：五轮分层重复留出；7,460 次预测涉及 5,485 笔交易（51 笔欺诈，70 次欺诈预测）。五轮 SD 描述划分波动，非置信区间；不作独立性假设或显著性检验。
- 输入：七数值字段标准化、一类别字段五列独热编码；不加入派生特征。商户类别先对有符号 SHAP 求和，再取绝对值。置换整组五列保留合法独热编码。
- 版式：图7-1及7-3为两个等宽面板，图7-2为2×2等宽等高面板；图示原始数据和派生源表全部保存。
- 交付：300dpi PNG、嵌入可选文本的PDF、可编辑文本SVG；字号不低于5pt；各图运行对齐、PDF文本与碰撞检查，随后人工查看。
""", encoding="utf-8")
    print("Exported 3 figures in PNG/PDF/SVG.")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
