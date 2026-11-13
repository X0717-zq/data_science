"""Build the complete chapter from verified observations; no model fitting."""
from pathlib import Path
import hashlib
import json
import subprocess
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, fbeta_score, matthews_corrcoef, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figure_alignment_helper import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parent
C = json.loads((ROOT / "chapter6_writeup_config.json").read_text(encoding="utf-8"))
SOURCE = ROOT / "credit_card_fraud_10k_results"
SUP = SOURCE / "chapter6_supplement"
OUT = ROOT / C["output"]
METHODS = ["RF-Plain", "RF-Balanced", "SMOTE", "ADASYN", "NearMiss",
           "SMOTE+Tomek", "SMOTE+ENN", "EasyEnsemble", "Isolation Forest"]
COLORS = ["#1F4E79", "#5B8DB8", "#2C7F78", "#68A69F", "#B17A4B",
          "#8267A5", "#B39BC8", "#4E6D46", "#8B6559"]
STYLES = ["-", "--", "-", "--", "-.", "-", "--", "-.", ":"]


def csv(frame, name):
    frame.to_csv(OUT / "tables" / name, index=False, encoding="utf-8-sig")


def md(frame):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(cell, frame.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = ["| " + " | ".join(map(cell, row)) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule] + body)


def link(label, path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def figlink(label, name):
    return "!" + link(label, OUT / "figures" / (name + ".png"))


def export(fig, name):
    fig.canvas.draw()
    path = OUT / "figures" / name
    require_matplotlib_panel_alignment(
        fig, json_out=path.with_suffix(".alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    assert C["figure_dpi"] >= 300
    fig.savefig(path.with_suffix(".png"), dpi=C["figure_dpi"], facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def flowchart():
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(cx, cy, w, h, text, face="#EDF2F7"):
        ax.add_patch(FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=face, edgecolor="#52667A", linewidth=.8))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5,
                linespacing=1.5, color="#192D40")

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops={"arrowstyle": "->", "lw": .9, "color": "#52667A",
                                "shrinkA": 2, "shrinkB": 2})

    box(.5,.925,.83,.09,"低正例率数据与字段核验\n9,943条：94欺诈＋9,849正常；8个原始特征")
    box(.5,.78,.83,.09,"五轮分层留出（同轮九模型共用划分）\n70%训练 / 15%验证 / 15%测试；冻结行索引")
    arrow((.5,.874),(.5,.831))
    box(.17,.615,.27,.115,"训练集\n6,960条 / 66正例\n拟合标准化与独热编码")
    box(.5,.615,.27,.115,"验证集\n1,491条 / 14正例\n仅应用训练所得变换")
    box(.83,.615,.27,.115,"测试集\n1,492条 / 14正例\n仅应用训练所得变换")
    for x in [.17,.5,.83]:
        arrow((x,.729),(x,.679))
    box(.17,.435,.27,.115,"仅训练部分重采样\n或类别权重 / 无监督拟合\n九模型生成连续分数","#E9F2EF")
    box(.5,.435,.27,.115,"验证分数选择阈值\nF2最大；并列取较大阈值\n随后冻结阈值","#E9F2EF")
    box(.83,.435,.27,.115,"冻结模型与阈值\n计算测试指标与曲线\n导出逐笔TP/FP/TN/FN","#E9F2EF")
    for x in [.17,.5,.83]:
        arrow((x,.552),(x,.499))
    arrow((.316,.435),(.354,.435))
    arrow((.646,.435),(.684,.435))
    box(.31,.225,.55,.13,"训练规模学习曲线：20%→100%\n仅使用嵌套训练子集与固定验证集\n9模型×5规模×5轮＝225次拟合")
    box(.80,.225,.32,.13,"五轮汇总与误差分析\n均值±样本SD\n结合原始交易核查FP/FN")
    arrow((.17,.372),(.17,.296))
    arrow((.83,.372),(.83,.296))
    ax.text(.5,.06,"验证集与测试集不重采样；测试结果不回流至参数、特征或阈值选择",
            ha="center", va="center", fontsize=8, color="#394D61")
    export(fig, "fig6_1_workflow")


def pr_family():
    raw = pd.read_csv(SOURCE / "pr_curve_points_by_round.csv")
    grid = np.linspace(0, 1, 201)
    rows = []
    for (repeat, method), g in raw.groupby(["repeat", "method"], sort=False):
        # Consolidate duplicate recall before interpolation so the envelope is well-defined.
        grouped = g.groupby("recall", sort=True).precision.max()
        x = grouped.index.to_numpy()
        envelope = np.maximum.accumulate(grouped.to_numpy()[::-1])[::-1]
        assert np.all(np.diff(x) > 0) and x[0] == 0 and x[-1] == 1
        y = np.interp(grid, x, envelope)
        rows.extend(dict(repeat=repeat, method=method, recall=r, precision=p)
                    for r, p in zip(grid, y))
    per_round = pd.DataFrame(rows)
    assert per_round.groupby(["repeat", "method"]).ngroups == 45
    csv(per_round, "pr_envelope_common_grid_by_round.csv")
    source = per_round.groupby(["method", "recall"], sort=False).precision.agg(
        mean_precision="mean", std_precision="std").reset_index()
    source.to_csv(OUT / "tables" / "pr_curve_family_source.csv",
                  index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    fig.subplots_adjust(left=.11, right=.97, bottom=.30, top=.90)
    for i, method in enumerate(METHODS):
        g = source[source.method.eq(method)].sort_values("recall")
        x, y, sd = g.recall.to_numpy(), g.mean_precision.to_numpy(), g.std_precision.to_numpy()
        assert len(g) == 201 and np.all(np.diff(x) > 0)
        assert np.all((y >= 0) & (y <= 1)) and np.all(sd >= 0)
        ax.plot(x, y, color=COLORS[i], ls=STYLES[i], label=method,
                lw=1.9 if i == 0 else 1.2)
        ax.fill_between(x, np.clip(y-sd, 0, 1), np.clip(y+sd, 0, 1),
                        color=COLORS[i], alpha=.07, lw=0)
    ax.axhline(14/1492, color="#777777", lw=.8, ls="--", label="测试正例率 0.9383%")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="召回率（Recall）",
           ylabel="精确率（Precision）",
           title="九模型PR曲线族：五轮精确率包络的均值与波动")
    ax.grid(alpha=.16, lw=.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52,.055),
               ncol=3, fontsize=7.2, handlelength=2.4, columnspacing=1.7)
    fig.text(.5,.015,"阴影：±1 SD；包络仅用于展示，PR-AUC与AP按各轮原始分数独立计算",
             ha="center", fontsize=7)
    export(fig, "fig6_2_pr_family")


def main():
    for sub in ["tables", "figures", "qa"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SUP / "source_manifest.json").read_text(encoding="utf-8"))
    for rel, digest in manifest["source_hashes"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    assert json.loads((SUP / "verification.json").read_text(encoding="utf-8"))["passed"]
    plan = json.loads((SOURCE / "experiment_plan.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(SOURCE / "summary_mean_std.csv").set_index("method").loc[METHODS]
    predictions = pd.read_csv(SUP / "tables" / "test_predictions_with_features.csv")
    metrics = pd.read_csv(SOURCE / "metrics_by_round.csv")
    data = pd.read_csv(ROOT / "fraudulent_card.csv")
    assert len(predictions) == 67140 and len(data) == 9943 and data.is_fraud.sum() == 94

    # A test ROC operating-point diagnostic, not a validation-selected deployment threshold.
    constrained = []
    dummy = []
    for (repeat, method), g in predictions.groupby(["repeat", "method"], sort=False):
        fpr, tpr, th = roc_curve(g.y_true, g.y_score, drop_intermediate=False)
        valid = np.flatnonzero(fpr <= C["fpr_limit"])
        best_recall = np.max(tpr[valid])
        choices = valid[tpr[valid] == best_recall]
        idx = choices[np.argmax(th[choices])]
        constrained.append(dict(repeat=repeat, method=method,
            recall_at_fpr_1pct=best_recall, achieved_fpr=fpr[idx],
            diagnostic_test_threshold=th[idx]))
        if method == "RF-Plain":
            y = g.y_true.to_numpy(); pred = np.zeros(len(y), dtype=int)
            dummy.append(dict(repeat=repeat, total=len(y), tp=0, fp=0,
                tn=int((y == 0).sum()), fn=int(y.sum()),
                accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                f2=fbeta_score(y, pred, beta=2), mcc=matthews_corrcoef(y, pred), fpr=0))
    constrained = pd.DataFrame(constrained)
    csum = constrained.groupby("method").recall_at_fpr_1pct.agg(["mean", "std"])
    summary["recall_at_fpr_1pct_mean"] = csum["mean"]
    summary["recall_at_fpr_1pct_std"] = csum["std"]
    csv(constrained, "recall_at_fpr_1pct_by_round.csv")
    csv(pd.DataFrame(dummy), "accuracy_counterexample_test_by_round.csv")
    csv(summary.reset_index(), "appendix4_all_metrics_numeric.csv")

    def formatted(columns):
        rows = []
        for method, s in summary.iterrows():
            row = {"模型": method}
            for key, label, percent in columns:
                k = 100 if percent else 1
                digits = 6 if key == "roc_auc" else 4
                row[label] = f"{s[key+'_mean']*k:.{digits}f} ± {s[key+'_std']*k:.{digits}f}"
            rows.append(row)
        return pd.DataFrame(rows)

    table4a = formatted([("pr_auc", "PR-AUC", False), ("ap", "AP", False),
                        ("roc_auc", "ROC-AUC", False), ("accuracy", "Accuracy（%）", True)])
    table4b = formatted([("precision", "Precision", False), ("recall", "Recall", False),
                        ("f1", "F1", False), ("f2", "F2", False), ("mcc", "MCC", False)])
    table4c = formatted([("fpr", "FPR（%）", True),
                        ("recall_at_fpr_1pct", "Recall@FPR≤1%（曲线诊断）", False)])
    for name, frame in [("appendix4a_ranking_metrics.csv",table4a),
                        ("appendix4b_threshold_metrics.csv",table4b),
                        ("appendix4c_low_fpr_metrics.csv",table4c)]:
        csv(frame, name)
    counts = pd.read_csv(SUP / "tables" / "confusion_matrices_pooled.csv")
    count_table = counts.set_index("method").loc[METHODS].reset_index()[["method","tp","fp","tn","fn"]]
    count_table.columns = ["模型","TP","FP","TN","FN"]
    csv(count_table, "appendix4d_confusion_counts.csv")
    seeds = pd.DataFrame([{"轮次": i, "首次划分": s, "验证/测试划分": s+1000,
                           "模型/随机采样器": s+i, "学习子集": s+2000}
                          for i,s in enumerate(plan["split_seeds"], 1)])
    csv(seeds, "random_seeds.csv")
    thresholds = metrics[["repeat","method","split_seed","model_seed",
                          "threshold_from_validation","validation_f2_at_threshold"]]
    csv(thresholds, "validation_thresholds_45.csv")
    threshold_ranges = metrics.groupby("method").threshold_from_validation.agg(["min","max"])
    threshold_ranges = threshold_ranges.loc[METHODS].reset_index()
    threshold_ranges.columns = ["模型","五轮最小阈值","五轮最大阈值"]
    for col in threshold_ranges.columns[1:]:
        threshold_ranges[col] = threshold_ranges[col].map(lambda x: f"{x:.6f}")
    csv(threshold_ranges, "validation_threshold_ranges.csv")

    hardware_cmd = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
        "$cpu=Get-CimInstance Win32_Processor; "
        "$cs=Get-CimInstance Win32_ComputerSystem; "
        "$os=Get-CimInstance Win32_OperatingSystem; "
        "[pscustomobject]@{cpu=$cpu.Name;cores=$cpu.NumberOfCores;"
        "logical_processors=$cpu.NumberOfLogicalProcessors;ram_bytes=$cs.TotalPhysicalMemory;"
        "os_caption=$os.Caption;os_version=$os.Version;architecture=$os.OSArchitecture}|ConvertTo-Json -Compress")
    hardware = json.loads(subprocess.check_output(
        ["powershell","-NoProfile","-Command",hardware_cmd]).decode("utf-8", errors="replace"))
    hardware["captured_utc"] = datetime.now(timezone.utc).isoformat()
    hardware["scope"] = "Current supplementary analysis host; original run did not preserve a CPU/RAM snapshot."
    (OUT / "environment_snapshot.json").write_text(
        json.dumps({"original_logged_software":plan["versions"],"current_hardware":hardware},
                   ensure_ascii=False,indent=2), encoding="utf-8")
    environment = pd.DataFrame([
        ["操作系统（原日志）", plan["versions"]["platform"], "原实验日志记录"],
        ["操作系统（本次复核）", hardware["os_caption"], "本次补充复核机器快照"],
        ["CPU", hardware["cpu"], "本次补充复核机器快照"],
        ["CPU核心/逻辑处理器", f'{hardware["cores"]} / {hardware["logical_processors"]}', "本次补充复核机器快照"],
        ["物理内存", f'{hardware["ram_bytes"]/2**30:.2f} GiB', "本次补充复核机器快照"],
        *[[k, plan["versions"][k], "原实验日志记录"] for k in
          ["python","numpy","pandas","scikit_learn","imbalanced_learn","matplotlib"]],
        ["执行方式", "CPU；模型n_jobs=1", "模型代码；未调用GPU训练"]
    ], columns=["项目","配置","依据"])
    csv(environment, "hardware_software_environment.csv")

    # Inspect the original file only to validate provenance; no new sample construction.
    raw_file = ROOT.parent / "credit_card_fraud_10k.csv"
    provenance = "本章直接读取已准备好的9,943条低正例率数据，本轮没有重新抽取正例。"
    if raw_file.exists():
        raw = pd.read_csv(raw_file)
        if len(raw) == 10000 and raw.is_fraud.sum() == 151:
            retained = raw[raw.transaction_id.isin(data.transaction_id)]
            assert len(retained) == len(data)
            a = retained.set_index("transaction_id").sort_index()
            b = data.set_index("transaction_id").sort_index()
            pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False)
            provenance += "交易编号核验表明，该数据保留原始10,000条中的全部9,849条正常交易及94条欺诈交易，另57条欺诈未纳入本章基准。"
    (OUT / "dataset_provenance.txt").write_text(provenance, encoding="utf-8")

    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":[C["figure_font"],"Arial","DejaVu Sans"],
        "font.size":8, "axes.titlesize":9, "axes.labelsize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.7,"svg.fonttype":"none","pdf.fonttype":42,
        "axes.unicode_minus":False,"figure.facecolor":"white","legend.frameon":False})
    flowchart()
    pr_family()
    for old, new in [
        ("roc_curves_mean_and_low_fpr","fig6_3_roc"),
        ("confusion_matrices_nine_models","fig6_4_confusion"),
        ("learning_curves_pr_auc","fig6_5_learning"),
        ("fp_fn_error_rates","fig6_6_errors")]:
        for ext in C["figure_formats"]:
            shutil.copy2(SUP / "figures" / f"{old}.{ext}", OUT / "figures" / f"{new}.{ext}")

    cases = pd.read_csv(SUP / "case_analysis" / "representative_fp_fn_cases.csv")
    rf_cases = cases[cases.method.eq("RF-Plain")][
        ["outcome","repeat","transaction_id","y_score","threshold","amount",
         "device_trust_score","velocity_last_24h"]].copy()
    for col in ["y_score","threshold"]:
        rf_cases[col] = rf_cases[col].map(lambda x:f"{x:.4f}")
    rf_cases.columns = ["类型","轮次","交易编号","分数","阈值","金额","设备信任评分","近24h交易数"]
    rf_round = pd.read_csv(SUP / "tables" / "confusion_matrices_by_round.csv")
    rf_round = rf_round[rf_round.method.eq("RF-Plain")][["repeat","threshold","tp","fp","tn","fn"]]
    rf_round.threshold = rf_round.threshold.map(lambda x:f"{x:.4f}")
    rf_round.columns = ["轮次","验证阈值","TP","FP","TN","FN"]
    learning = pd.read_csv(SUP / "learning_curves" / "learning_endpoints_for_report.csv")
    d = dummy[0]
    dummy_table = pd.DataFrame([
        ["全部预测为正常",d["tp"],d["fp"],d["tn"],d["fn"],f'{d["accuracy"]:.6%}',0,0,0,0,0],
    ], columns=["规则","TP","FP","TN","FN","Accuracy","Recall","F1","F2","MCC","FPR"])
    substitutions = {
        "TABLE4A":md(table4a),"TABLE4B":md(table4b),"TABLE4C":md(table4c),
        "COUNTS":md(count_table),"SEEDS":md(seeds),"ENV":md(environment),
        "RF_CASES":md(rf_cases),"RF_ROUNDS":md(rf_round),"LEARNING":md(learning),
        "DUMMY":md(dummy_table),"DATA_PROVENANCE":provenance,
        "FIG1":figlink("图6-1 整体技术路线","fig6_1_workflow"),
        "FIG2":figlink("图6-2 PR曲线族","fig6_2_pr_family"),
        "FIG3":figlink("图6-3 ROC曲线","fig6_3_roc"),
        "FIG4":figlink("图6-4 混淆矩阵","fig6_4_confusion"),
        "FIG5":figlink("图6-5 学习曲线","fig6_5_learning"),
        "FIG6":figlink("图6-6 误报率与漏报率","fig6_6_errors"),
        "THRESHOLD_RANGES":md(threshold_ranges),
        "FULL_METRICS":link("附表4完整数值表",OUT / "tables/appendix4_all_metrics_numeric.csv"),
        "PREDICTIONS":link("逐笔预测与交易特征",SUP / "tables/test_predictions_with_features.csv"),
        "CASES":link("九模型FP/FN完整分析",SUP / "case_analysis/FP_FN案例分析.md"),
        "RAW_PR":link("各轮原始PR曲线点",SOURCE / "pr_curve_points_by_round.csv"),
        "THRESHOLDS":link("45组验证阈值与验证F2",OUT / "tables/validation_thresholds_45.csv"),
        "RECALL_DATA":link("低FPR诊断逐轮数据",OUT / "tables/recall_at_fpr_1pct_by_round.csv"),
    }
    template = (ROOT / "chapter6_writeup_template.md").read_text(encoding="utf-8")
    for token, text in substitutions.items():
        template = template.replace("@@"+token+"@@", text)
    assert "@@" not in template, "Unresolved template token"
    (OUT / "第六章_模型构建与实验分析.md").write_text(template,encoding="utf-8")
    guide = pd.DataFrame([
        ["图6-1","6.1第一段后","整体流程图","fig6_1_workflow"],
        ["图6-2","6.5附表4及其解读之后","PR曲线族，必须保留","fig6_2_pr_family"],
        ["图6-3","6.5 PR曲线图之后","ROC及低FPR局部图","fig6_3_roc"],
        ["图6-4","6.5混淆计数表之后","九模型混淆矩阵","fig6_4_confusion"],
        ["图6-5","6.5学习曲线实验说明之后","九模型训练规模学习曲线","fig6_5_learning"],
        ["图6-6","6.6阈值与误差分析段落之后","五轮FPR/FNR波动","fig6_6_errors"],
    ],columns=["编号","插入位置","图名","文件基名"])
    csv(guide, "figure_insertion_guide.csv")
    guide["PNG入口"] = [link("打开图",OUT/"figures"/(n+".png")) for n in guide["文件基名"]]
    (OUT/"图表插入位置与使用说明.md").write_text(
        "# 第六章图表插入位置\n\n"+md(guide.drop(columns="文件基名"))+
        "\n\n每图提供PNG（300 dpi）、PDF及SVG。图6-1与图6-2为本次新增，图6-3至图6-6沿用已核验图表。"
        "\n\n表6-1采样策略放在6.3；表6-2固定参数空间放在6.3；随机种子和环境放在6.4；"
        "Accuracy反例与替代指标体系紧接6.4的评价协议；附表4（a—d）放在6.5。"
        "\n\n本任务为二分类，使用混淆矩阵及FP/FN诊断，不加入线性回归残差图。学习曲线横轴为训练样本量，不是epoch。"
        "\n\n完整正文："+link("第六章完整稿",OUT/"第六章_模型构建与实验分析.md")+"\n",
        encoding="utf-8")
    qa = {
        "source_hashes_unchanged":True,"original_models_retrained":False,
        "model_rounds":len(metrics),"prediction_rows":len(predictions),
        "counterexample_test_n":d["total"],"counterexample_test_fraud":d["fn"],
        "low_fpr_diagnostic_rows":len(constrained),"model_count":len(summary),
        "hyperparameter_search_performed":False,
        "pr_display":"45 raw PR curves: max precision for duplicate recall, reverse cumulative maximum, common-grid interpolation; not used for AUC calculation",
        "claim_boundary":"Repeated holdouts; sample SD, not independent CI; current hardware snapshot identified separately",
    }
    (OUT/"qa/numerical_checks.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    for path in [Path(__file__),ROOT/"chapter6_writeup_config.json",ROOT/"chapter6_writeup_template.md"]:
        shutil.copy2(path,OUT/path.name)
    print("Chapter, appendix tables, six-figure insertion guide and two new figures written.")
    print(csum.loc[METHODS].to_string())


if __name__ == "__main__":
    main()
