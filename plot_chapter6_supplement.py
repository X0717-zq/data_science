"""Report figures from observed scores, paired rounds and newly fitted learning data."""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from chapter6_common import C, OUT, METHODS, initialize, read_csv, save_json
from figure_alignment_helper import require_matplotlib_panel_alignment

COLORS = ["#1F4E79", "#5B8DB8", "#2C7F78", "#68A69F", "#B17A4B",
          "#8267A5", "#B39BC8", "#4E6D46", "#8B6559"]
STYLES = ["-", "--", "-", "--", "-.", "-", "--", "-.", ":"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [C["figure_font"], "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .7,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def panel(ax, letter, title):
    ax.annotate(letter, xy=(0,1), xycoords="axes fraction", xytext=(-12,10),
                textcoords="offset points", ha="left",va="bottom",fontsize=10,fontweight="bold")
    ax.set_title(title,pad=10)
    ax.set_axisbelow(True)


def export(fig, name):
    fig.canvas.draw()
    path=OUT/"figures"/name
    require_matplotlib_panel_alignment(fig,json_out=path.with_suffix(".alignment.json"),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    assert C["figure_dpi"] >= 300
    assert set(C["figure_formats"]) == {"png", "pdf", "svg"}
    fig.savefig(path.with_suffix(".svg"),facecolor="white")
    fig.savefig(path.with_suffix(".pdf"),facecolor="white")
    fig.savefig(path.with_suffix(".png"),dpi=C["figure_dpi"],facecolor="white")
    plt.close(fig)
    print(f"Exported {name}",flush=True)


def roc_overview():
    curves=read_csv(OUT/"tables"/"roc_curve_mean_std.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.25))
    fig.subplots_adjust(left=.10,right=.98,bottom=.32,top=.85,wspace=.30)
    handles=[]
    for i,ax in enumerate(axes):
        panel(ax,chr(97+i),["整体ROC", "低误报区域（FPR ≤ 5%）"][i])
        for j,method in enumerate(METHODS):
            g=curves[curves.method==method].sort_values("fpr")
            x=g.fpr.to_numpy(); y=g.mean_tpr.to_numpy(); sd=g.std_tpr.to_numpy()
            assert np.all(np.diff(x)>0)
            line,=ax.plot(x,y,color=COLORS[j],linestyle=STYLES[j],
                          lw=1.8 if j==0 else 1.15,label=method)
            ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=COLORS[j],alpha=.055,lw=0)
            if i==0:handles.append(line)
        limit=1 if i==0 else .05
        ax.plot([0,limit],[0,limit],color="#9A9A9A",lw=.7,ls="--")
        ax.set(xlim=(0,limit),ylim=(0,1.02),xlabel="假正例率（FPR）")
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_ylabel("真正例率（TPR）" if i==0 else "")
        ax.grid(alpha=.16,lw=.5)
    fig.legend(handles,METHODS,loc="lower center",bbox_to_anchor=(.52,.07),ncol=3,
               columnspacing=2.0,handlelength=2.4)
    fig.text(.5,.025,"曲线和阴影：五轮均值 ± 1 SD；逐模型波动见分面图",ha="center",fontsize=7)
    export(fig,"roc_curves_mean_and_low_fpr")


def roc_by_model():
    curves=read_csv(OUT/"tables"/"roc_curve_mean_std.csv")
    rounds=read_csv(OUT/"tables"/"roc_curve_common_grid_by_round.csv")
    summary=read_csv(OUT/"tables"/"roc_auc_summary.csv").set_index("method")
    fig,axes=plt.subplots(3,3,figsize=(7.2,7.8))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.90,wspace=.38,hspace=.62)
    for j,(ax,method) in enumerate(zip(axes.flat,METHODS)):
        s=summary.loc[method]
        panel(ax,chr(97+j),f"{method}\nROC-AUC {s.roc_auc_mean:.4f} ± {s.roc_auc_std:.4f}")
        for _,g in rounds[rounds.method==method].groupby("repeat"):
            g=g.sort_values("fpr")
            ax.plot(g.fpr,g.tpr,color="#B6BCC2",alpha=.55,lw=.6)
        g=curves[curves.method==method].sort_values("fpr")
        x=g.fpr.to_numpy();y=g.mean_tpr.to_numpy();sd=g.std_tpr.to_numpy()
        ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=COLORS[j],alpha=.18,lw=0)
        ax.plot(x,y,color=COLORS[j],lw=1.4)
        ax.set(xlim=(0,.05),ylim=(0,1.02),xticks=[0,.025,.05],yticks=[0,.5,1])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(alpha=.16,lw=.5)
        if j//3==2:ax.set_xlabel("假正例率（FPR）")
        if j%3==0:ax.set_ylabel("真正例率（TPR）")
    fig.text(.5,.025,"灰线：单轮；彩线与阴影：五轮均值 ± 1 SD。AUC由各轮完整ROC计算。",ha="center",fontsize=7)
    export(fig,"roc_curves_by_model")


def confusion():
    table=read_csv(OUT/"tables"/"confusion_matrices_pooled.csv").set_index("method")
    fig,axes=plt.subplots(3,3,figsize=(7.2,7.6))
    fig.subplots_adjust(left=.11,right=.98,bottom=.10,top=.92,wspace=.42,hspace=.52)
    for j,(ax,method) in enumerate(zip(axes.flat,METHODS)):
        s=table.loc[method]
        counts=np.array([[s.tn,s.fp],[s.fn,s.tp]],dtype=int)
        rates=counts/counts.sum(axis=1,keepdims=True)
        panel(ax,chr(97+j),method)
        ax.imshow(rates,cmap="Blues",vmin=0,vmax=1,aspect="auto",interpolation="nearest")
        for r in range(2):
            for c in range(2):
                ax.text(c,r,f"{counts[r,c]:,}\n({rates[r,c]:.2%})",ha="center",va="center",
                        fontsize=8,color="white" if rates[r,c]>.6 else "#17212B",linespacing=1.6)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["正常0","欺诈1"],yticklabels=["正常0","欺诈1"])
        ax.tick_params(length=0,pad=4)
        for spine in ax.spines.values():spine.set_visible(False)
        if j//3==2:ax.set_xlabel("预测类别",labelpad=6)
        if j%3==0:ax.set_ylabel("真实类别",labelpad=6)
    fig.text(.5,.025,"汇总五轮预测次数（每模型7,460次）；底色与括号内百分比按真实类别归一化。",ha="center",fontsize=7)
    export(fig,"confusion_matrices_nine_models")


def error_rates():
    data=read_csv(OUT/"tables"/"test_metrics_verified_by_round.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.4),sharey=True)
    fig.subplots_adjust(left=.24,right=.98,bottom=.20,top=.86,wspace=.30)
    for i,(ax,metric,title) in enumerate(zip(axes,["fpr","fnr"],["正常交易误报率（FPR）","欺诈交易漏报率（FNR）"])):
        panel(ax,chr(97+i),title)
        for j,method in enumerate(METHODS):
            vals=data[data.method==method].sort_values("repeat")[metric].to_numpy()
            ax.scatter(vals,j+np.linspace(-.13,.13,len(vals)),s=18,color=COLORS[j],alpha=.8,zorder=3)
            ax.scatter([vals.mean()],[j],marker="D",s=25,color="#1A1A1A",zorder=4)
        ax.set_yticks(np.arange(len(METHODS)),METHODS)
        ax.set_xlim(-.001,.065) if i==0 else ax.set_xlim(-.01,.8)
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis="x",alpha=.2,lw=.5)
        ax.set_xlabel("比例")
    axes[0].set_ylim(8.5,-.5)
    fig.text(.55,.055,"圆点：每轮结果；黑色菱形：五轮均值。两面板分母分别为正常与欺诈交易。",ha="center",fontsize=7)
    export(fig,"fp_fn_error_rates")


def learning():
    path=OUT/"learning_curves"/"learning_curve_summary.csv"
    if not path.exists():
        print("Learning results not ready; other figures completed.",flush=True)
        return
    data=read_csv(path)
    assert data.successful_rounds.eq(5).all(),"Incomplete repetitions must be resolved or explicitly reported."
    fig,axes=plt.subplots(3,3,figsize=(7.2,7.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.92,wspace=.34,hspace=.46)
    handles=[]
    for j,(ax,method) in enumerate(zip(axes.flat,METHODS)):
        panel(ax,chr(97+j),method)
        g=data[data.method==method].sort_values("fraction")
        x=g.fraction.to_numpy()*100
        for part,color,style,label in [("train","#7A838C","--","训练集（原始样本）"),
                                       ("validation","#246C9C","-","固定验证集")]:
            y=g[f"{part}_PR_AUC_mean"].to_numpy();sd=g[f"{part}_PR_AUC_std"].to_numpy()
            line,=ax.plot(x,y,color=color,ls=style,lw=1.3,marker="o",ms=3,label=label)
            ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=color,alpha=.15,lw=0)
            if j==0:handles.append(line)
        ax.set(xlim=(17,103),ylim=(-.02,1.04),xticks=[20,40,60,80,100],yticks=[0,.5,1])
        ax.grid(alpha=.16,lw=.5)
        if j//3==2:ax.set_xlabel("训练样本比例（%）")
        if j%3==0:ax.set_ylabel("PR-AUC")
    fig.legend(handles,[h.get_label() for h in handles],loc="lower center",bbox_to_anchor=(.52,.035),ncol=2)
    fig.text(.5,.012,"五轮均值 ± 1 SD；每轮验证集固定。学习曲线未使用测试集。",ha="center",fontsize=7)
    export(fig,"learning_curves_pr_auc")


def main():
    initialize()
    shutil.copy2(__file__,OUT/"code_snapshot"/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("figure_alignment_helper.py"),OUT/"code_snapshot"/"figure_alignment_helper.py")
    save_json(OUT/"qa"/"figure_contract.json",dict(backend="Python / matplotlib",archetype="quantitative comparison grids",
        exports=C["figure_formats"],dpi=C["figure_dpi"],uncertainty="sample SD over five repeated holdouts; not a CI",
        roles={"roc_curves_mean_and_low_fpr":"overall ranking and low-FPR discrimination",
               "roc_curves_by_model":"round-to-round ROC variability for every method",
               "confusion_matrices_nine_models":"operating-threshold error counts and class-conditional rates",
               "fp_fn_error_rates":"paired-round false-alarm and missed-fraud rates",
               "learning_curves_pr_auc":"training-size sensitivity and training-validation gap"},
        omissions="No requested methods or rounds omitted; no significance claims."))
    roc_overview();roc_by_model();confusion();error_rates();learning()


if __name__=="__main__":
    main()
