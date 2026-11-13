# 第六章补充交付物

已补齐ROC-AUC及ROC图、混淆矩阵、逐笔预测、九模型FP/FN分析和学习曲线。

## 从这里开始

[阅读补充报告](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/第六章_补充分析报告.md)

## 数据与分析文件

| 内容 | 打开文件 |
| --- | --- |
| 完整补充报告 | [第六章_补充分析报告.md](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/第六章_补充分析报告.md) |
| ROC-AUC均值和SD | [roc_auc_summary.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/roc_auc_summary.csv) |
| 45组逐轮完整核验指标 | [test_metrics_verified_by_round.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/test_metrics_verified_by_round.csv) |
| ROC原始曲线点和阈值 | [roc_curve_points_by_round.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/roc_curve_points_by_round.csv) |
| ROC共同网格均值和SD | [roc_curve_mean_std.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/roc_curve_mean_std.csv) |
| 逐轮TP/FP/TN/FN | [confusion_matrices_by_round.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/confusion_matrices_by_round.csv) |
| 五轮汇总混淆矩阵 | [confusion_matrices_pooled.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/confusion_matrices_pooled.csv) |
| 逐笔预测原字段（67,140行） | [test_predictions_by_round.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/test_predictions_by_round.csv) |
| 逐笔预测＋交易特征＋错误类型 | [test_predictions_with_features.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/tables/test_predictions_with_features.csv) |
| 全部FP/FN预测记录（828行） | [fp_fn_all_occurrences.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/case_analysis/fp_fn_all_occurrences.csv) |
| 九模型FP/FN文字分析 | [FP_FN案例分析.md](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/case_analysis/FP_FN案例分析.md) |
| 54条FP/FN选例 | [representative_fp_fn_cases.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/case_analysis/representative_fp_fn_cases.csv) |
| 去重交易的误判频次 | [transaction_error_frequency.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/case_analysis/transaction_error_frequency.csv) |
| 训练规模和各类数量 | [training_size_counts.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/learning_curves/training_size_counts.csv) |
| 225次学习曲线逐轮结果 | [learning_curve_metrics.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/learning_curves/learning_curve_metrics.csv) |
| 学习曲线均值和SD | [learning_curve_summary.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/learning_curves/learning_curve_summary.csv) |
| 原验证结果复现核验 | [full_size_validation_replay_check.csv](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/learning_curves/full_size_validation_replay_check.csv) |
| 独立数据核验 | [verification.json](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/verification.json) |

## 图表（每张均有PNG、PDF、SVG）

- 平均ROC与低误报区域：[PNG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/roc_curves_mean_and_low_fpr.png) · [PDF](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/roc_curves_mean_and_low_fpr.pdf) · [SVG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/roc_curves_mean_and_low_fpr.svg)

- 逐模型ROC波动：[PNG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/roc_curves_by_model.png) · [PDF](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/roc_curves_by_model.pdf) · [SVG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/roc_curves_by_model.svg)

- 九模型混淆矩阵：[PNG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/confusion_matrices_nine_models.png) · [PDF](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/confusion_matrices_nine_models.pdf) · [SVG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/confusion_matrices_nine_models.svg)

- 误报率与漏报率：[PNG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/fp_fn_error_rates.png) · [PDF](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/fp_fn_error_rates.pdf) · [SVG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/fp_fn_error_rates.svg)

- 九模型学习曲线：[PNG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/learning_curves_pr_auc.png) · [PDF](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/learning_curves_pr_auc.pdf) · [SVG](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/chapter6_supplement/figures/learning_curves_pr_auc.svg)

## 参数与代码

原九模型参数由credit_card_fraud_10k_experiment.py继承；新增训练规模、子集种子偏移和绘图设置集中在chapter6_supplement_config.json。原结果未被覆盖。

[新增参数配置](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/chapter6_supplement_config.json)

按顺序运行：run_chapter6_learning_curves.py、build_chapter6_tables.py、verify_chapter6_supplement.py、plot_chapter6_supplement.py、report_chapter6_supplement.py。学习曲线已完成的模型会校验分数文件并复用；改变配置或原始数据会触发保护，须另设输出目录。

学习曲线分数文件（NPZ）及冻结行索引保存在learning_curves/round_01—round_05中。混淆矩阵汇总按预测次数计算，不能把重复测试记录当作独立交易。
