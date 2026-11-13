"""Chinese evidence-linked supplement, case review and deliverable index."""
from pathlib import Path
import json
import shutil

import pandas as pd
from chapter6_common import C, ROOT, SOURCE, OUT, METHODS, initialize, csv, read_csv, markdown


def link(label,path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def picture(label,name):
    return f"![{label}]({(OUT/'figures'/name).resolve().as_posix()})"


def case_table(frame):
    out=frame[["outcome","repeat","transaction_id","y_score","threshold","amount","device_trust_score","velocity_last_24h"]].copy()
    for col in ["y_score","threshold"]:out[col]=out[col].map(lambda x:f"{x:.4f}")
    out.amount=out.amount.map(lambda x:f"{x:.2f}")
    out.columns=["类型","轮次","交易编号","预测分数","验证阈值","交易金额","设备信任评分","近24小时交易数"]
    return markdown(out)


def main():
    initialize()
    assert json.loads((OUT/'verification.json').read_text(encoding='utf-8'))['passed']
    shutil.copy2(__file__,OUT/'code_snapshot'/Path(__file__).name)
    aucs=read_csv(OUT/'tables/roc_auc_summary.csv').set_index('method')
    errors=read_csv(OUT/'case_analysis/error_summary_by_method.csv').set_index('method')
    rounds=read_csv(OUT/'tables/confusion_matrices_by_round.csv')
    cases=read_csv(OUT/'case_analysis/representative_fp_fn_cases.csv')
    profile=read_csv(OUT/'case_analysis/error_numeric_feature_profiles.csv').set_index(['method','outcome','feature'])
    learning=read_csv(OUT/'learning_curves/learning_curve_summary.csv').set_index(['method','fraction'])
    rf_learning_low=learning.loc[('RF-Plain',.2)]
    rf_learning_full=learning.loc[('RF-Plain',1.)]
    sizes=read_csv(OUT/'learning_curves/training_size_counts.csv').drop_duplicates('fraction')
    original=read_csv(SOURCE/'summary_mean_std.csv').set_index('method')
    summary_rows=[];endpoints=[]
    for method in METHODS:
        a=aucs.loc[method]
        summary_rows.append({'模型':method,'ROC-AUC（均值±SD）':f'{a.roc_auc_mean:.6f} ± {a.roc_auc_std:.6f}',
                             'PR-AUC（均值±SD）':f'{a.pr_auc_mean:.4f} ± {a.pr_auc_std:.4f}'})
        lo,hi=learning.loc[(method,.2)],learning.loc[(method,1.)]
        endpoints.append({'模型':method,'20%验证PR-AUC':f'{lo.validation_PR_AUC_mean:.4f} ± {lo.validation_PR_AUC_std:.4f}',
            '100%验证PR-AUC':f'{hi.validation_PR_AUC_mean:.4f} ± {hi.validation_PR_AUC_std:.4f}',
            '100%训练PR-AUC':f'{hi.train_PR_AUC_mean:.4f}',
            '100%训练−验证差':f'{hi.train_PR_AUC_mean-hi.validation_PR_AUC_mean:.4f}'})
    auc_display=pd.DataFrame(summary_rows)
    learning_display=pd.DataFrame(endpoints)
    csv(auc_display,OUT/'tables/roc_auc_for_report.csv')
    csv(learning_display,OUT/'learning_curves/learning_endpoints_for_report.csv')
    pooled=errors.reset_index()[['method','tn','fp','fn','tp','unique_FP_transactions','unique_FN_transactions']]
    pooled.columns=['模型','TN次数','FP次数','FN次数','TP次数','误报交易去重数','漏报交易去重数']
    learning_sizes=sizes[['fraction','train_n','train_positive','train_negative','validation_n','validation_positive']].copy()
    learning_sizes.fraction=learning_sizes.fraction.map(lambda x:f'{x:.0%}')
    learning_sizes.columns=['训练比例','训练记录数','训练欺诈数','训练正常数','验证记录数','验证欺诈数']
    lines=["# 第六章补充：ROC、混淆矩阵、逐笔误差及学习曲线",
        "本补充以修订版九模型实验为唯一基准：9,943条交易、94条欺诈，五轮70%/15%/15%分层留出。"
        "原测试结果不重训、不替换；ROC和混淆矩阵由已保存测试预测重算。学习曲线另行训练225次，"
        "沿用九模型原参数及划分，未运行超参数搜索。本文件不混入100棵树的A1—A4消融结果。",
        "## 1. ROC-AUC与ROC曲线",
        markdown(auc_display),
        "ROC-AUC逐轮根据连续分数计算，再对五轮取均值和样本标准差。PR-AUC仍作为本任务主要排序指标。"
        "例如Isolation Forest的ROC-AUC为0.946781，而PR-AUC仅为0.1948；这组实测结果说明，"
        "较高ROC-AUC不足以单独证明当前低正例率任务的欺诈识别效果。",
        picture("九模型平均ROC及低误报区域",'roc_curves_mean_and_low_fpr.png'),
        "图注：左图展示完整FPR范围，右图展示FPR≤5%的区域。各轮ROC先在相同FPR处保留可达到的最大TPR，"
        "再线性插值到共同网格，显示五轮均值±1 SD。数值AUC由各轮完整原始ROC计算，不使用平均曲线面积替代。",
        picture("逐模型ROC波动",'roc_curves_by_model.png'),
        "分面图展示每种方法的五条单轮曲线（灰线）与均值及SD（彩色）。阴影反映划分波动，不是独立样本置信区间。",
        "## 2. 混淆矩阵与误报、漏报",
        markdown(pooled),
        "矩阵按行表示真实类别、列表示预测类别，顺序为正常0、欺诈1，即[[TN,FP],[FN,TP]]。"
        "每个模型汇总5×1,492=7,460次测试预测，包含7,390次正常和70次欺诈预测机会。"
        "五轮测试记录去重后为5,485笔交易，因此上表计数不能称为7,460笔独立交易。",
        picture("九模型混淆矩阵",'confusion_matrices_nine_models.png'),
        "图注：单元格给出预测次数和真实类别内的百分比；底色按真实类别归一化，以免正常交易数量掩盖欺诈漏检。",
        picture("逐轮误报率和漏报率",'fp_fn_error_rates.png'),
        "图注：圆点表示五轮各自结果，菱形表示五轮均值。FPR分母为正常交易数，FNR分母为欺诈交易数。",
        "**汇总计数与逐轮均值不能混淆。** RF-Plain的五轮平均Precision为0.9120，"
        "但把五轮TP、FP相加后计算的Precision为59/(59+11)=0.8429。这分别是逐轮宏平均和合并计数比值，"
        "不能将后者替换前者，也不能将差异认定为计算错误。",
        "## 3. FP/FN具体案例分析",
        "已整理67,140条九模型逐笔预测，保留轮次、方法、交易编号、真实标签、连续分数、阈值、预测类别、"
        "错误类型及原始交易特征；其中828条为误分类预测记录。另按方法和交易编号统计出现轮数及误判轮数，"
        "并导出54条选例（每模型3条FP和3条FN，模型内交易编号去重）。选例按分数与阈值差选取："
        "FP取最大正差，FN取最负差，用于展示较强错误判断，不代表随机抽样或整体分布。",
        "### RF-Plain的逐轮误差",
    ]
    rf_round=rounds[rounds.method.eq('RF-Plain')][['repeat','threshold','tn','fp','fn','tp']].copy()
    rf_round.threshold=rf_round.threshold.map(lambda x:f'{x:.4f}')
    lines += [markdown(rf_round),
        "RF-Plain的11次误报全部出现在第5轮，该轮阈值为验证集确定的0.1250，14个欺诈样本均被识别；"
        "前4轮没有误报，但合计漏检11次，涉及10笔不同交易。这一差异说明，轮次和验证阈值对应的误报/漏报取舍"
        "需要与均值同时报告；本次没有根据这些测试误差重新修改阈值。",
        case_table(cases[cases.method.eq('RF-Plain')]),
        "交易3103在第5轮为正常交易，分数0.2000高于阈值0.1250而被误报；其金额442.77、设备信任评分32、"
        "近24小时交易数5。交易656在第3轮为欺诈交易，分数0.1500低于阈值0.3167而被漏报；"
        "虽然境外交易与位置不匹配标记均为1，但设备信任评分为55、金额仅24.57。"
        "这些案例展示误判时的特征组合，不能仅凭案例断言某个字段导致误判。",
        "在RF-Plain的预测记录中，FP与TN的设备信任评分中位数分别为45和62；FN与TP分别为39和31。"
        "FP与TN的近24小时交易次数中位数分别为4和2；FN与TP分别为5和3。"
        "这些是描述性差异，记录可能跨轮重复，且错误样本数量很小，未进行显著性或因果检验。",
        "其余八种方法的同类选例、误判次数及特征概况见"+link('九模型FP/FN案例分析',OUT/'case_analysis/FP_FN案例分析.md')+"。",
        "## 4. 学习曲线：不同训练样本量下的表现",
        "学习曲线沿用五轮原训练、验证划分。在每轮训练集内部按类别固定随机排列，构造嵌套子集；"
        "子集随机种子为该轮划分种子+2000。样本量及正例数按预定比例取整，9种方法共用同一子集。"
        "每个模型的随机种子仍为划分种子+轮次，采样器及模型参数保持原样。",
        markdown(learning_sizes),
        "每个子集独立拟合标准化与类别编码，采样只作用于该子集；训练分数在原始真实子集上计算，"
        "不在合成后的样本上计算。训练曲线是样本内重代入分数，不能作为泛化结果。"
        "验证集固定为1,491条、14个正例，学习曲线全程不使用测试集。总计9×5×5=225次拟合全部成功。"
        "100%训练量下45组验证最优F2及阈值均与原实验一致。",
        picture("九模型PR-AUC学习曲线",'learning_curves_pr_auc.png'),
        "图注：横轴为原训练集使用比例；灰色虚线为真实训练子集上的PR-AUC，蓝色实线为固定验证集PR-AUC；"
        "阴影为五轮样本SD。表内同时保留AP及ROC-AUC学习结果。训练比例变化同时改变正常和欺诈样本数量，"
        "不能将趋势单独归因于正例数量。",
        markdown(learning_display),
        "RF-Plain的平均验证PR-AUC随训练量从20%增加到100%，"
        f"由{rf_learning_low.validation_PR_AUC_mean:.4f}±{rf_learning_low.validation_PR_AUC_std:.4f}"
        f"升至{rf_learning_full.validation_PR_AUC_mean:.4f}±{rf_learning_full.validation_PR_AUC_std:.4f}；"
        "训练PR-AUC在各档约为1.0000。训练—验证差距随样本量增加缩小，当前范围内更多训练样本对应更好的"
        "验证排序表现，但这不保证继续扩大数据后一定保持同样增幅，也不构成独立外部验证。",
        "NearMiss在全量训练时仍存在较大的训练—验证差距；Isolation Forest的训练与验证PR-AUC均较低，"
        "在本次样本范围内没有出现与RF-Plain相同的持续大幅改善。其余方法的逐点均值、SD及单轮结果均完整保留。",
        "## 5. 数据口径与可复现性",
        "所有新增结果保存在chapter6_supplement子目录，原九模型文件保持不变。"
        "ROC-AUC与其他原指标均由67,140条预测重新核验；225组学习曲线的分数、原始行索引及指标均保存并独立重算。"
        "图表导出PNG、PDF和可编辑SVG，并检查字体、面板对齐、文字碰撞和逐面板可读性。",
        "本批没有超参数搜索，也没有利用测试集选择参数或特征。五轮之间有样本重叠，SD表示重复划分波动；"
        "不能写成五个独立数据集的置信区间。Isolation Forest保存的是异常分数，不是欺诈概率；"
        "其他模型的predict_proba分数也未在本补充中进行概率校准。",
    ]
    report='\n\n'.join(lines)+'\n'
    (OUT/'第六章_补充分析报告.md').write_text(report,encoding='utf-8')
    case_lines=['# 九模型FP/FN案例分析',
        '所有计数以五轮预测发生记录为单位，同时列出去重交易数。极端选例按模型内分数减阈值排序，'
        'FP取最大正差、FN取最负差，每类保留3笔不同交易。这些例子用于查看错误判断，不是随机代表样本。'
        '不同模型分数尺度不同，尤其Isolation Forest分数不是概率，不比较跨模型的分数差绝对大小。']
    for method in METHODS:
        s=errors.loc[method]
        case_lines += [f'## {method}',
            f'五轮合计FP={int(s.fp)}次（{int(s.unique_FP_transactions)}笔不同交易），'
            f'FN={int(s.fn)}次（{int(s.unique_FN_transactions)}笔不同交易）；'
            f'合并计数FPR={s.fpr_pooled:.4%}，漏报率FNR={s.fnr_pooled:.4%}。',
            case_table(cases[cases.method.eq(method)])]
        a=profile.loc[(method,'FP','device_trust_score'),'median']
        b=profile.loc[(method,'TN','device_trust_score'),'median']
        c=profile.loc[(method,'FN','device_trust_score'),'median']
        d=profile.loc[(method,'TP','device_trust_score'),'median']
        case_lines += [f'设备信任评分的记录级中位数：FP={a:g}、TN={b:g}；FN={c:g}、TP={d:g}。'
            '比较时分别以同一真实类别中的正确预测作为参照。此处仅描述观察差异，不据此断定错误成因。']
    case_lines += ['## 数据文件',
        link('全部误分类记录及交易特征',OUT/'case_analysis/fp_fn_all_occurrences.csv'),
        link('按交易编号统计各轮错误频次',OUT/'case_analysis/transaction_error_frequency.csv'),
        link('数值特征的均值、中位数和四分位数',OUT/'case_analysis/error_numeric_feature_profiles.csv'),
        link('商户类别分布',OUT/'case_analysis/error_merchant_profiles.csv')]
    (OUT/'case_analysis/FP_FN案例分析.md').write_text('\n\n'.join(case_lines)+'\n',encoding='utf-8')
    items=[
        ('完整补充报告','第六章_补充分析报告.md'),
        ('ROC-AUC均值和SD','tables/roc_auc_summary.csv'),
        ('45组逐轮完整核验指标','tables/test_metrics_verified_by_round.csv'),
        ('ROC原始曲线点和阈值','tables/roc_curve_points_by_round.csv'),
        ('ROC共同网格均值和SD','tables/roc_curve_mean_std.csv'),
        ('逐轮TP/FP/TN/FN','tables/confusion_matrices_by_round.csv'),
        ('五轮汇总混淆矩阵','tables/confusion_matrices_pooled.csv'),
        ('逐笔预测原字段（67,140行）','tables/test_predictions_by_round.csv'),
        ('逐笔预测＋交易特征＋错误类型','tables/test_predictions_with_features.csv'),
        ('全部FP/FN预测记录（828行）','case_analysis/fp_fn_all_occurrences.csv'),
        ('九模型FP/FN文字分析','case_analysis/FP_FN案例分析.md'),
        ('54条FP/FN选例','case_analysis/representative_fp_fn_cases.csv'),
        ('去重交易的误判频次','case_analysis/transaction_error_frequency.csv'),
        ('训练规模和各类数量','learning_curves/training_size_counts.csv'),
        ('225次学习曲线逐轮结果','learning_curves/learning_curve_metrics.csv'),
        ('学习曲线均值和SD','learning_curves/learning_curve_summary.csv'),
        ('原验证结果复现核验','learning_curves/full_size_validation_replay_check.csv'),
        ('独立数据核验','verification.json'),
    ]
    index=pd.DataFrame([{'内容':label,'文件':rel} for label,rel in items])
    csv(index,OUT/'交付物索引.csv')
    readme=['# 第六章补充交付物','已补齐ROC-AUC及ROC图、混淆矩阵、逐笔预测、九模型FP/FN分析和学习曲线。',
            '## 从这里开始',link('阅读补充报告',OUT/'第六章_补充分析报告.md'),
            '## 数据与分析文件',markdown(pd.DataFrame([{'内容':label,'打开文件':link(Path(rel).name,OUT/rel)} for label,rel in items])),
            '## 图表（每张均有PNG、PDF、SVG）']
    for name,label in [('roc_curves_mean_and_low_fpr','平均ROC与低误报区域'),('roc_curves_by_model','逐模型ROC波动'),
        ('confusion_matrices_nine_models','九模型混淆矩阵'),('fp_fn_error_rates','误报率与漏报率'),('learning_curves_pr_auc','九模型学习曲线')]:
        readme.append('- '+label+'：'+link('PNG',OUT/'figures'/f'{name}.png')+' · '+link('PDF',OUT/'figures'/f'{name}.pdf')+' · '+link('SVG',OUT/'figures'/f'{name}.svg'))
    readme += ['## 参数与代码',
        '原九模型参数由credit_card_fraud_10k_experiment.py继承；新增训练规模、子集种子偏移和绘图设置集中在chapter6_supplement_config.json。原结果未被覆盖。',
        link('新增参数配置',ROOT/'chapter6_supplement_config.json'),
        '按顺序运行：run_chapter6_learning_curves.py、build_chapter6_tables.py、verify_chapter6_supplement.py、plot_chapter6_supplement.py、report_chapter6_supplement.py。'
        '学习曲线已完成的模型会校验分数文件并复用；改变配置或原始数据会触发保护，须另设输出目录。',
        '学习曲线分数文件（NPZ）及冻结行索引保存在learning_curves/round_01—round_05中。'
        '混淆矩阵汇总按预测次数计算，不能把重复测试记录当作独立交易。']
    (OUT/'README_交付物索引.md').write_text('\n\n'.join(readme)+'\n',encoding='utf-8')
    print('Chinese supplement report, nine-model case analyses and delivery index written.')


if __name__=='__main__':
    main()
