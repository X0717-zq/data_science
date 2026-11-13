"""Auditable paired comparison and figures for the completed neural experiment."""
from pathlib import Path
import json
import shutil
import importlib.metadata
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve,roc_curve
from figure_alignment_helper import require_matplotlib_panel_alignment

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/"nn_comparison_config.json").read_text(encoding="utf-8"))
OUT=ROOT/C["output"]
SOURCE=ROOT/"credit_card_fraud_10k_results"
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
COLORS=["#1F4E79","#B87333","#3D8A7F","#8B6BAD","#A65363"]


def read(path):
    return pd.read_csv(path,float_precision="round_trip")


def md(frame):
    def cell(v):return str(v).replace("|","\\|").replace("\n"," ")
    lines=["| "+" | ".join(map(cell,frame.columns))+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines.extend("| "+" | ".join(map(cell,row))+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def link(label,path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def image(label,name):
    return "!"+link(label,OUT/"figures"/f"{name}.png")


def save(fig,name):
    fig.canvas.draw()
    path=OUT/"figures"/name
    require_matplotlib_panel_alignment(fig,json_out=path.with_suffix(".alignment.json"),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    assert C["figure_dpi"]>=300
    fig.savefig(path.with_suffix(".png"),dpi=C["figure_dpi"],facecolor="white")
    fig.savefig(path.with_suffix(".pdf"),facecolor="white")
    fig.savefig(path.with_suffix(".svg"),facecolor="white")
    plt.close(fig)


def panel(ax,letter,title):
    ax.annotate(letter,xy=(0,1),xycoords="axes fraction",xytext=(-14,9),
        textcoords="offset points",ha="left",va="bottom",fontsize=10,fontweight="bold")
    ax.set_title(title,pad=10)


def curves(predictions):
    grid=np.linspace(0,1,201)
    result=[]
    for (method,repeat),g in predictions.groupby(["method","repeat"]):
        p,r,_=precision_recall_curve(g.y_true,g.y_score)
        grouped=pd.DataFrame({"x":r,"y":p}).groupby("x").y.max()
        x=grouped.index.to_numpy()
        yp=np.maximum.accumulate(grouped.to_numpy()[::-1])[::-1]
        assert np.all(np.diff(x)>0)
        pr=np.interp(grid,x,yp)
        fpr,tpr,_=roc_curve(g.y_true,g.y_score,drop_intermediate=False)
        grouped=pd.DataFrame({"x":fpr,"y":tpr}).groupby("x").y.max()
        assert np.all(np.diff(grouped.index.to_numpy())>0)
        roc=np.interp(grid,grouped.index,grouped)
        result.extend(dict(method=method,repeat=repeat,x=x,precision=p,tpr=t)
                      for x,p,t in zip(grid,pr,roc))
    result=pd.DataFrame(result)
    result.to_csv(OUT/"tables/curve_common_grid_by_round.csv",index=False,encoding="utf-8-sig")
    summary=result.groupby(["method","x"],sort=False).agg(
        precision_mean=("precision","mean"),precision_std=("precision","std"),
        tpr_mean=("tpr","mean"),tpr_std=("tpr","std")).reset_index()
    summary.to_csv(OUT/"tables/curve_mean_std.csv",index=False,encoding="utf-8-sig")
    return summary


def paired_plot(paired):
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    fig.subplots_adjust(left=.23,right=.97,bottom=.23,top=.90)
    labels=[]
    for j,v in enumerate(C["variants"]):
        g=paired[paired.nn_method.eq(v["method"])]
        values=g.delta_pr_auc.to_numpy()
        for k,value in enumerate(values):
            ax.scatter(value,j+(k-2)*.045,color=COLORS[k],s=18,zorder=3)
        ax.errorbar(values.mean(),j,xerr=values.std(ddof=1),fmt="D",color="#202C38",
            markersize=4.5,capsize=3,lw=1.0,zorder=4)
        labels.append(v["strategy"].replace("none","不重采样").replace("balanced","类别权重"))
    ax.axvline(0,color="#7F8790",ls="--",lw=.8)
    ax.set(yticks=range(7),yticklabels=labels,xlabel="测试PR-AUC差值（MLP − 对应RF）",
           title="相同划分与采样策略下的配对差值")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.16,lw=.5)
    handles=[Line2D([],[],marker="o",ls="",color=COLORS[k],label=f"第{k+1}轮") for k in range(5)]
    handles.append(Line2D([],[],marker="D",ls="-",color="#202C38",label="均值 ± SD"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.04),ncol=3,fontsize=7.5)
    save(fig,"paired_pr_auc_difference")


def baseline_curves(summary):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.85))
    fig.subplots_adjust(left=.10,right=.98,bottom=.26,top=.84,wspace=.30)
    for j,ax in enumerate(axes):
        panel(ax,chr(97+j),["PR曲线","ROC曲线"][j])
        col=["precision","tpr"][j]
        for k,method in enumerate(["RF-Plain","MLP-Plain"]):
            g=summary[summary.method.eq(method)].sort_values("x")
            x=g.x.to_numpy();y=g[col+"_mean"].to_numpy();sd=g[col+"_std"].to_numpy()
            ax.plot(x,y,color=COLORS[k],lw=1.6,label=method)
            ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=COLORS[k],alpha=.14,lw=0)
        if j==0:ax.axhline(14/1492,color="#999999",ls="--",lw=.7)
        else:ax.plot([0,1],[0,1],color="#999999",ls="--",lw=.7)
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel=["Recall","FPR"][j],ylabel=["Precision","TPR"][j])
        ax.grid(alpha=.15,lw=.5)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.07),ncol=2)
    fig.text(.5,.02,"五轮均值 ± SD；PR使用精确率包络，积分指标由原始分数计算",ha="center",fontsize=7)
    save(fig,"rf_mlp_plain_pr_roc")


def epoch_plot(histories):
    fig,axes=plt.subplots(4,2,figsize=(7.2,10.2))
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.95,hspace=.62,wspace=.28)
    max_epoch=int(histories.epoch.max())
    for j,(ax,v) in enumerate(zip(axes.flat,C["variants"])):
        panel(ax,chr(97+j),v["method"])
        for k in range(1,6):
            g=histories[(histories.method==v["method"])&(histories["repeat"]==k)].sort_values("epoch")
            ax.plot(g.epoch,g.train_pr_auc,color="#A3A8AD",ls="--",alpha=.40,lw=.65)
            ax.plot(g.epoch,g.validation_pr_auc,color=COLORS[k-1],lw=.85,alpha=.85)
            best=g[np.isclose(g.validation_pr_auc,g.validation_pr_auc.max(),rtol=0,
                              atol=C["improvement_tolerance"])].iloc[0]
            ax.scatter(best.epoch,best.validation_pr_auc,color=COLORS[k-1],s=20,marker="o",zorder=4)
        ax.set(xlim=(0,max_epoch),ylim=(0,1.02),xlabel="Epoch",ylabel="PR-AUC")
        ax.grid(alpha=.15,lw=.5)
    fig.delaxes(axes.flat[-1])
    handles=[Line2D([],[],color=COLORS[k],label=f"验证：第{k+1}轮") for k in range(5)]
    handles.append(Line2D([],[],color="#A3A8AD",ls="--",label="真实训练集"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.04),ncol=3,fontsize=7.5)
    fig.text(.5,.012,"圆点：各轮最佳验证PR-AUC对应epoch；线条在早停处结束，未补齐或延长",ha="center",fontsize=7)
    save(fig,"neural_epoch_learning_curves")


def main():
    assert json.loads((OUT/"qa/verification.json").read_text(encoding="utf-8"))["passed"]
    summary=read(OUT/"tables/all_model_summary_mean_std.csv").set_index("method")
    training=read(OUT/"tables/training_summary.csv")
    nnmetrics=read(OUT/"tables/nn_metrics_by_round.csv")
    paired=read(OUT/"tables/paired_differences_by_round.csv")
    predictions=read(OUT/"tables/nn_test_predictions_with_features.csv")
    rf_predictions=read(SOURCE/"test_predictions_by_round.csv")
    combined=pd.concat([rf_predictions,predictions],ignore_index=True)
    histories=[]
    for repeat in range(1,6):
        for j,_ in enumerate(C["variants"]):
            histories.append(read(OUT/"models"/f"round_{repeat:02d}"/f"variant_{j:02d}"/"epoch_history.csv"))
    histories=pd.concat(histories,ignore_index=True)
    histories.to_csv(OUT/"tables/all_epoch_history.csv",index=False,encoding="utf-8-sig")
    display=[]
    for v in C["variants"]:
        rf=summary.loc[v["rf_method"]]; nn=summary.loc[v["method"]]
        delta=paired[paired.nn_method.eq(v["method"])].delta_pr_auc
        display.append({"采样/权重":v["strategy"].replace("none","不重采样").replace("balanced","类别权重"),
            "RF PR-AUC":f"{rf.pr_auc_mean:.4f} ± {rf.pr_auc_std:.4f}",
            "MLP PR-AUC":f"{nn.pr_auc_mean:.4f} ± {nn.pr_auc_std:.4f}",
            "MLP−RF":f"{delta.mean():+.4f} ± {delta.std(ddof=1):.4f}",
            "MLP更高轮数":f"{int((delta>0).sum())}/5"})
    pair_table=pd.DataFrame(display)
    pair_table.to_csv(OUT/"tables/paired_pr_auc_for_report.csv",index=False,encoding="utf-8-sig")
    all_methods=[item for v in C["variants"] for item in [v["rf_method"],v["method"]]]+["EasyEnsemble","Isolation Forest"]
    def metric_table(cols):
        rows=[]
        for method in all_methods:
            s=summary.loc[method];r={"模型":method}
            for key,label,pct in cols:
                factor=100 if pct else 1
                r[label]=f"{s[key+'_mean']*factor:.4f} ± {s[key+'_std']*factor:.4f}"
            rows.append(r)
        return pd.DataFrame(rows)
    ranking=metric_table([("pr_auc","PR-AUC",False),("ap","AP",False),("roc_auc","ROC-AUC",False),
                          ("accuracy","Accuracy（%）",True)])
    operating=metric_table([("precision","Precision",False),("recall","Recall",False),("f1","F1",False),
                            ("f2","F2",False),("mcc","MCC",False),("fpr","FPR（%）",True)])
    ranking.to_csv(OUT/"tables/all_ranking_metrics_for_report.csv",index=False,encoding="utf-8-sig")
    operating.to_csv(OUT/"tables/all_threshold_metrics_for_report.csv",index=False,encoding="utf-8-sig")
    run_table=training[["repeat","method","fitted_n","fitted_positive","best_epoch","epochs_run",
                        "best_validation_pr_auc","threshold_from_validation","validation_f2"]].copy()
    for col in ["best_validation_pr_auc","threshold_from_validation","validation_f2"]:
        run_table[col]=run_table[col].map(lambda x:f"{x:.4f}")
    run_table.columns=["轮次","方法","拟合样本数","拟合正例数","最佳epoch","实际epoch","验证PR-AUC","验证阈值","验证F2"]
    run_table.to_csv(OUT/"tables/best_epochs_and_thresholds.csv",index=False,encoding="utf-8-sig")
    plain_compare=nnmetrics[nnmetrics.method.eq("MLP-Plain")][
        ["repeat","best_epoch","threshold_from_validation","pr_auc","tp","fp","tn","fn"]].copy()
    for col in ["threshold_from_validation","pr_auc"]:
        plain_compare[col]=plain_compare[col].map(lambda x:f"{x:.4f}")
    plain_compare.columns=["轮次","最佳epoch","验证阈值","测试PR-AUC","TP","FP","TN","FN"]
    plain_counts=nnmetrics[nnmetrics.method.eq("MLP-Plain")][["tp","fp","tn","fn"]].sum()
    cases=[]
    for method,g in predictions.groupby("method",sort=False):
        g=g.copy();g["score_minus_threshold"]=g.y_score-g.threshold
        for outcome in ["FP","FN"]:
            selected=g[g.outcome.eq(outcome)].sort_values("score_minus_threshold",ascending=(outcome=="FN"))
            cases.append(selected.drop_duplicates("transaction_id").head(3))
    cases=pd.concat(cases,ignore_index=True)
    cases.to_csv(OUT/"tables/representative_nn_fp_fn_cases.csv",index=False,encoding="utf-8-sig")
    example=cases[cases.method.eq("MLP-Plain")][
        ["outcome","repeat","transaction_id","y_true","y_score","threshold","amount","device_trust_score","velocity_last_24h"]].copy()
    for col in ["y_score","threshold"]:example[col]=example[col].map(lambda x:f"{x:.4f}")
    example.columns=["类型","轮次","交易编号","真实标签","分数","阈值","金额","信任评分","近24h次数"]
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":[C["figure_font"],"Arial","DejaVu Sans"],
        "font.size":8,"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "legend.fontsize":7.5,"axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.7,
        "svg.fonttype":"none","pdf.fonttype":42,"axes.unicode_minus":False,"legend.frameon":False,
        "figure.facecolor":"white","axes.facecolor":"white"})
    curve_summary=curves(combined)
    paired_plot(paired);baseline_curves(curve_summary);epoch_plot(histories)
    neural_methods=[v["method"] for v in C["variants"]]
    rf_plain=summary.loc["RF-Plain"];nn_plain=summary.loc["MLP-Plain"]
    top_nn=summary.loc[neural_methods].sort_values("pr_auc_mean",ascending=False).iloc[0]
    top_name=top_nn.name
    improved=sum(float(summary.loc[v["method"],"pr_auc_mean"])>float(summary.loc[v["rf_method"],"pr_auc_mean"]) for v in C["variants"])
    achieved=[m for m in neural_methods if summary.loc[m,"pr_auc_mean"]>=.8]
    maxed=int(training.stop_reason.eq("max_epochs").sum())
    equivalence=read(OUT/"qa/smote_tomek_equivalence.csv")
    same_rounds=equivalence.loc[equivalence.identical_resampled_data,"repeat"].astype(int).tolist()
    equivalence_note=("核验采样矩阵后确认，SMOTE与SMOTE+Tomek在第"
        +"、".join(map(str,same_rounds))+"轮产生完全相同的训练矩阵（该轮Tomek未删除记录）；"
        "相同初始化和训练协议下轨迹相同属于预期结果，并非重复抄录。其余轮次的清理数量另存核验表。") if same_rounds else \
        "SMOTE与SMOTE+Tomek各轮实际采样矩阵的比较记录已保存。"
    environment={k:importlib.metadata.version(k) for k in
                 ["torch","numpy","pandas","scikit-learn","imbalanced-learn","matplotlib","joblib"]}
    (OUT/"software_versions.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    report=[
        "# 神经网络与随机森林对比实验",
        "## 1. 结论与比较范围",
        f"本次完成七种MLP策略×五轮＝35次训练，并与原来对应的35组RF结果配对比较。"
        f"不重采样时，MLP-Plain的测试PR-AUC为{nn_plain.pr_auc_mean:.4f}±{nn_plain.pr_auc_std:.4f}，"
        f"RF-Plain为{rf_plain.pr_auc_mean:.4f}±{rf_plain.pr_auc_std:.4f}。"
        f"七种策略中有{improved}种MLP的平均测试PR-AUC高于对应RF。",
        f"按平均测试PR-AUC描述，表现最高的神经网络策略为{top_name}："
        f"{top_nn.pr_auc_mean:.4f}±{top_nn.pr_auc_std:.4f}。这是全部结果中的描述性排序，"
        "未按测试成绩重新选择网络、采样器、epoch或阈值。"
        +("平均PR-AUC达到0.80的神经网络策略为："+", ".join(achieved)+"。" if achieved else
          "本次七种神经网络策略的平均测试PR-AUC均未达到0.80。"),
        md(pair_table),
        image("图1 配对PR-AUC差值","paired_pr_auc_difference"),
        "图1：同一轮、相同采样/加权设置下，测试PR-AUC差值为MLP减RF；圆点为五轮配对差值，"
        "黑色菱形与误差线为均值±样本SD。正值表示该轮MLP较高，负值表示RF较高。"
        "五轮并非独立数据集，未将此SD当作置信区间，未进行显著性检验。",
        "## 2. 固定数据与网络设置",
        "使用9,943条数据、94条欺诈和8个原始特征，与第六章相同。每轮训练集6,960条（66正例）、"
        "验证集1,491条（14正例）、测试集1,492条（14正例）。直接读取上一批冻结索引，未重新划分数据；"
        "交易编号不作为输入。7个数值字段标准化、1个商户类别独热编码，共12维；预处理仅拟合训练集。",
        f"网络为{'→'.join(map(str,[12,*C['hidden_layers'],1]))}的MLP，隐藏层采用ReLU和"
        f"Dropout={C['dropout']}，末层输出logit，评价时通过sigmoid得到分数。"
        f"训练使用BCEWithLogitsLoss、AdamW，学习率{C['learning_rate']}，weight_decay={C['weight_decay']}，"
        f"batch_size={C['batch_size']}，最多{C['max_epochs']}个epoch；CPU单线程、固定随机种子及确定性运算。",
        "本批未搜索网络宽度、层数、学习率等超参数。七种策略的网络和优化器设置相同，"
        "区别只在训练采样或类别权重。MLP-Balanced采用pos_weight=6,894/66≈104.4545；"
        "其他策略使用未加权BCE。过采样目标少数/多数数量比为0.5，SMOTE/ADASYN近邻数5，"
        "NearMiss使用version=1和3个近邻，SMOTE+Tomek/ENN沿用原RF脚本的清理设置。"
        "采样前使用与RF相同的双精度变换结果，之后才转为神经网络的float32张量。",
        "原始八组核心实验和额外RF-Balanced均保留。EasyEnsemble和Isolation Forest作为原有参考结果列入总表，"
        "本次没有将其改造成神经网络，因此不将它们列入七组逐项配对。",
        "首次划分种子：20261109、20261110、20261111、20261112、20261113；"
        "模型种子：20261110、20261112、20261114、20261116、20261118。"
        "同一轮的各MLP策略使用相同初始化种子。配置和输入摘要保存在run_manifest.json，"
        "软件版本保存在software_versions.json。",
        "## 3. 每个epoch如何判断最佳模型",
        "**唯一的checkpoint选择指标是验证集PR-AUC（原始PR点的梯形积分），不是Accuracy、训练loss或测试成绩。**"
        "每个epoch结束后关闭Dropout，在真实训练集与固定验证集上计算分数，记录BCE、PR-AUC、AP和ROC-AUC。"
        "验证PR-AUC比此前最佳值增加超过1e-12时更新最佳checkpoint；并列时保留较早epoch。"
        f"至少训练{C['min_epochs']}个epoch，连续{C['early_stopping_patience']}个epoch无新最佳则停止，"
        f"上限{C['max_epochs']}。本次有{maxed}组达到epoch上限，其余按耐心规则早停。",
        "结束后恢复最佳checkpoint，再在验证集选择F2最大的分类阈值，并列时取较大阈值。"
        "35个checkpoint及其验证阈值全部冻结后，才开始生成神经网络测试预测。"
        "F2阈值只影响固定工作点的分类结果，不改变连续分数的PR-AUC。",
        image("图2 神经网络epoch训练曲线","neural_epoch_learning_curves"),
        "图2：灰色虚线为各轮真实训练集PR-AUC，彩色线为对应轮验证PR-AUC，圆点标记最佳epoch。"
        "展示每轮原始轨迹，不平均长度不同的早停轨迹，也不将停止后的结果延长。"
        "这里横轴是epoch，与前一批横轴为训练样本量的学习曲线含义不同。",
        md(run_table),
        "## 4. 全部模型结果",
        "以下均为五轮测试均值±样本SD。保留原九种方法并新增七种MLP，共16种方法。"
        "Accuracy和FPR以百分数表示，其余指标按0—1表示；PR-AUC为梯形积分，AP为平均精确率。",
        md(ranking),md(operating),
        image("图3 不重采样RF与MLP的PR及ROC曲线","rf_mlp_plain_pr_roc"),
        "图3：两种不重采样基线在相同五轮测试集上的曲线。PR在重复Recall处取最大Precision，"
        "构造精确率包络后插值；ROC在重复FPR处取最大TPR后插值。线与阴影为五轮均值±SD，"
        "排序指标从各轮原始分数计算，不使用平均包络面积替代。",
        "## 5. 误差与解释",
        "MLP-Plain各轮工作阈值和误差如下：",md(plain_compare),
        f"五轮合计TP={int(plain_counts.tp)}、FP={int(plain_counts.fp)}、TN={int(plain_counts.tn)}、"
        f"FN={int(plain_counts.fn)}。这些是重复测试中的预测次数，不是互不重复交易数。"
        "每轮只有14个正例，单笔欺诈会改变召回率约7.14个百分点，验证PR-AUC和最佳epoch都可能因少量记录而变化。",
        "下表按MLP-Plain的分数减阈值选择3条FP和3条FN（模型内交易编号去重）。"
        "它们用于观察强错误判断，不代表随机抽样，也不证明某个字段造成误判。",
        md(example),
        "模型差异可能来自函数形式、正则化及有效训练数据的差异。树模型通过分段划分直接学习表格变量阈值，"
        "MLP通过梯度更新学习连续组合；在每轮仅66条真实欺诈的条件下，当前MLP结构和正则化未必适合数据边界。"
        "若某一策略的MLP成绩较低，只能得出本次固定结构与训练协议下的结论，不能据此断言神经网络普遍不适用。",
        "NearMiss只保留198条拟合记录（66条欺诈、132条正常），每epoch的更新次数远少于原始或过采样训练集；"
        "相同epoch上限并不等于相同梯度更新预算。合成采样增加的是训练记录而不是独立欺诈证据。"
        "这些差异必须与采样策略的测试结果同时解释。",
        equivalence_note,
        "RF使用固定120棵树，MLP使用验证集选择epoch；两种训练过程的验证信息使用量并不完全相等。"
        "本比较控制了输入特征、原始划分、采样规则与阈值协议，但没有声称搜索预算完全等价。"
        "当前测试集已用于结果对比，若继续依据这些结果调整网络，应再保留新的最终测试集或采用嵌套评估。",
        "Accuracy仍不适合作为唯一判断标准：每轮全预测正常即可获得1,478/1,492=99.0617%的Accuracy，"
        "却漏掉全部14条欺诈。评价以PR-AUC/AP、Precision/Recall/F1/F2、MCC和FPR为主，并结合误差案例。"
        "保存的Recall@FPR≤1%为测试ROC诊断，未据此回调工作阈值。",
        "## 6. 文件索引",
        link("参数配置",ROOT/"nn_comparison_config.json"),
        link("运行代码",ROOT/"run_nn_rf_comparison.py"),
        link("七组配对PR-AUC对比",OUT/"tables/paired_pr_auc_for_report.csv"),
        link("16种方法全部指标",OUT/"tables/all_model_summary_mean_std.csv"),
        link("35组最佳epoch与阈值",OUT/"tables/best_epochs_and_thresholds.csv"),
        link("全部epoch记录",OUT/"tables/all_epoch_history.csv"),
        link("神经网络逐笔交易预测",OUT/"tables/nn_test_predictions_with_features.csv"),
        link("全部神经网络FP/FN记录",OUT/"tables/nn_fp_fn_all_occurrences.csv"),
        link("神经网络误差选例",OUT/"tables/representative_nn_fp_fn_cases.csv"),
        link("独立数值核验",OUT/"qa/verification.json"),
        "各轮最佳模型权重、训练/验证分数和预处理对象位于models目录；"
        "模型、配置、输入及原始结果摘要保存于run_manifest.json。图片均提供PNG（300 dpi）、PDF和SVG。",
    ]
    (OUT/"神经网络与随机森林对比报告.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    (OUT/"README_结果入口.md").write_text(
        "# 神经网络与随机森林对比\n\n"+link("完整对比报告",OUT/"神经网络与随机森林对比报告.md")+
        "\n\n"+md(pair_table)+"\n\n"+link("参数配置",ROOT/"nn_comparison_config.json")+
        "\n\n"+link("逐笔预测",OUT/"tables/nn_test_predictions_with_features.csv")+
        "\n\n"+link("最佳epoch与阈值",OUT/"tables/best_epochs_and_thresholds.csv")+"\n",encoding="utf-8")
    shutil.copy2(__file__,OUT/"code_snapshot"/Path(__file__).name)
    print(pair_table.to_string(index=False))
    print("Report and three figures completed.")


if __name__=="__main__":
    main()
