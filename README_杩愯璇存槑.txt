信用卡欺诈检测低正例率九方法实验包

一、数据
- fraudulent_card.csv：本实验使用的最终数据，9,943 条记录。
- 正常交易 9,849 条，欺诈交易 94 条，正例率 0.9454%。
- 该文件是已处理数据，脚本会保持记录及行顺序不变。

二、运行
建议 Python 环境安装：
  pip install numpy pandas matplotlib scikit-learn imbalanced-learn

在本目录打开终端执行：
  python credit_card_fraud_10k_experiment.py

也可显式指定输入和输出：
  python credit_card_fraud_10k_experiment.py --input fraudulent_card.csv --output credit_card_fraud_10k_results

若只提供原始 credit_card_fraud_10k.csv（10,000 条、151 条欺诈），脚本会固定随机抽取 94 条欺诈记录，保留全部 9,849 条正常记录，并输出低正例率数据。
脚本不会自动选择 fraudulent.csv，避免与另一个 URL 欺诈项目的数据文件混淆。

三、实验协议
- 五轮重复分层 70%/15%/15% 划分。
- 每轮：训练集 6,960 条（66 个欺诈），验证集 1,491 条（14 个欺诈），测试集 1,492 条（14 个欺诈）。
- 九种方法：RF-Plain、RF-Balanced、SMOTE、ADASYN、NearMiss、SMOTE+Tomek、SMOTE+ENN、EasyEnsemble、Isolation Forest。
- 每轮在验证集上按 F2 选择阈值，冻结后应用于测试集。
- 预处理器只在训练集拟合；重采样仅作用于训练集。

四、运行后输出
- summary_mean_std.csv：五轮指标均值和标准差。
- metrics_by_round.csv：每轮指标、阈值和混淆矩阵计数。
- split_counts.csv：每轮各数据集样本数量和正例率。
- pr_curves_family_mean.png：五轮插值平均 PR 曲线族。
- pr_auc_comparison.png：五轮测试 PR-AUC 均值±标准差图。
- pr_curve_points_by_round.csv、pr_curve_family_mean_std.csv：曲线数据。
- test_predictions_by_round.csv：逐笔测试分数和预测标签，可用于分析误报与漏报。
- experiment_plan.json：实验参数、软件版本及限制。
- report_中文.md：自动生成的结果报告。

五、results 子目录中的对照结果
results 文件夹包含此前完成的 5 轮、9 方法结果，用于和新脚本运行结果核对。新脚本完整运行后会在指定输出目录重新生成结果。

注意：五轮重复划分共享同一份 94 个欺诈样本，均值±标准差反映划分波动，不是独立样本置信区间。
