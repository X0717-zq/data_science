| 模型 | 主要参数 | PR-AUC（主指标） | ROC-AUC | F2 | 训练耗时（秒/轮，含附加步骤） |
| --- | --- | --- | --- | --- | --- |
| 随机森林（RF-Plain） | 120棵树；Gini；max_depth=None；max_features=sqrt；min_samples_leaf=1；bootstrap=True；class_weight=None；n_jobs=1 | 0.9658 ± 0.0243 | 0.9995 ± 0.0005 | 0.8415 ± 0.0532 | 0.78 ± 0.08 |
| 多层感知器（MLP-Plain） | 12→64→32→1；ReLU；Dropout=0.1；BCEWithLogitsLoss；AdamW；learning_rate=0.001；weight_decay=0.0001；batch_size=256；最多400 epoch；early_stopping patience=50 | 0.6598 ± 0.0967 | 0.9879 ± 0.0061 | 0.5815 ± 0.0872 | 19.11 ± 5.31 |

说明：两行均为不重采样、不加类别权重的Plain版本；相同8个原始字段（编码后12维），同一五轮70%/15%/15%分层重复留出。性能指标为五轮测试集均值±样本标准差，F2阈值来自各轮验证集；不是五折交叉验证。

MLP依据验证集PR-AUC选择最优epoch并恢复权重；实际训练epoch分别为142、202、179、126、239，最优epoch分别为92、152、129、76、189。训练至少60个epoch，最多400个epoch。

耗时边界：RF为学习曲线补跑中fraction=1.0（完整6960笔训练样本）的fit_and_score_seconds，包含预处理、拟合、训练/验证预测及指标和阈值计算，不含测试预测和结果文件保存。MLP为training_seconds，包含迭代训练、逐epoch训练/验证评价、恢复最优权重、阈值计算及模型/分数/历史保存，不含预处理、采样、模型与优化器初始化和测试预测。两者不是同一轮同步计时，也不是相同口径的纯fit时间，不能据此计算严格的算法加速倍数。MLP使用CPU单线程，RF n_jobs=1。

来源：all_model_summary_mean_std.csv；training_summary.csv；../../chapter6_supplement/learning_curves/learning_curve_metrics.csv（只取RF-Plain且fraction=1.0）。逐轮计时另外保存于RF_MLP_耗时逐轮来源.csv。
