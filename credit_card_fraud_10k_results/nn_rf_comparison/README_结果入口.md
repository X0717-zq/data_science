# 神经网络与随机森林对比

[完整对比报告](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/nn_rf_comparison/神经网络与随机森林对比报告.md)

| 采样/权重 | RF PR-AUC | MLP PR-AUC | MLP−RF | MLP更高轮数 |
| --- | --- | --- | --- | --- |
| 不重采样 | 0.9658 ± 0.0243 | 0.6598 ± 0.0967 | -0.3060 ± 0.1094 | 0/5 |
| 类别权重 | 0.8879 ± 0.0516 | 0.5981 ± 0.1926 | -0.2897 ± 0.2114 | 0/5 |
| SMOTE | 0.8052 ± 0.0554 | 0.6271 ± 0.1775 | -0.1780 ± 0.1493 | 0/5 |
| ADASYN | 0.8101 ± 0.0807 | 0.6334 ± 0.1886 | -0.1767 ± 0.1908 | 0/5 |
| NearMiss | 0.4235 ± 0.0574 | 0.1714 ± 0.1353 | -0.2521 ± 0.1637 | 1/5 |
| SMOTE+Tomek | 0.8017 ± 0.0502 | 0.6201 ± 0.1706 | -0.1815 ± 0.1470 | 0/5 |
| SMOTE+ENN | 0.7053 ± 0.0673 | 0.5759 ± 0.1924 | -0.1294 ± 0.1582 | 1/5 |

[参数配置](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/nn_comparison_config.json)

[逐笔预测](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/nn_rf_comparison/tables/nn_test_predictions_with_features.csv)

[最佳epoch与阈值](C:/Users/86132/Desktop/026/data_science/test_card_fraud/credit_card_fraud_10k_revised_bundle/credit_card_fraud_10k_results/nn_rf_comparison/tables/best_epochs_and_thresholds.csv)
