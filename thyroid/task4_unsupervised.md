# 任务4：无监督疾病判断

## 4.0 问题形式化描述
- **输入**：甲状腺数据集 `x ∈ ℝ⁶`，其中训练集仅包含正常样本（标记为0），测试集包含正常和患病样本（标记为0/1）。
- **目标**：在缺少患病样本的情况下，学习正常样本的分布 `p(x | y=0)`，并将与该分布显著偏离的样本视为潜在患病（异常）样本。
- **输出**：对测试集中每个样本的异常分数 `s(x)` 以及基于阈值的二分类判断 `ŷ ∈ {0,1}`（1表示患病）。

## 4.1 方法选择与理由
- **算法**：StandardScaler + One-Class SVM（RBF核，`nu=0.05`，`gamma='scale'`）。
- **选择理由**：
  - 训练集中仅有正常样本，适合使用单类学习（One-Class Classification）框架建模正常区域。
  - One-Class SVM 能在低维连续空间中学习非线性分界，适合 6 维特征且样本量适中（≈1.8k）。
  - `nu` 可直接控制异常占比上界（假设 ≤5%），与提示“训练集均为正常样本”一致；`gamma='scale'` 允许模型自适应特征方差。
  - 训练前使用 StandardScaler 消除特征量纲差异，保持核函数稳定。

## 4.2 模型实现与训练
1. **数据预处理**：
   - 读取 `train-set.csv` / `test-set.csv`（共3772条，特征维度6）。
   - 划分：`X_train`=1839条正常样本；`X_test`=1933条混合样本，其中患病占比≈4.86%。
   - `StandardScaler` 拟合于训练集并用同一变换作用于测试集。
2. **模型训练**：
   - 使用 `OneClassSVM` 在训练集上拟合正常区域，默认决策边界为 `f(x)=0`，`f(x)<0` 视为异常。
   - 训练脚本：`python thyroid/task4_unsupervised.py`（已包含数据加载、训练与评估逻辑，便于复现实验）。

## 4.3 判断效果评估
| 指标 | 数值 |
| --- | --- |
| ROC-AUC | 0.9753 |
| PR-AUC (positive) | 0.7818 |
| Accuracy | 0.9302 |
| Precision (positive) | 0.4055 |
| Recall (positive) | 0.9362 |
| F1 (positive) | 0.5659 |
| Confusion Matrix | TN=1710, FP=129, FN=6, TP=88 |

**结果解读**：
- 高 ROC-AUC / PR-AUC 表明异常分数对患病样本具有良好排序能力。
- 召回率 0.94 说明大部分患病样本被检测出来，满足医疗筛查“宁愿多报疑似”的需求。
- 精度 0.41 表示仍有一定误报，需要结合人工复核或进一步特征以降低 FP。
- 若需权衡误报，可根据 `decision_function` 分数重新设定阈值或调节 `nu`。

## 4.4 可视化分析
- `figures/anomaly_score_distribution.png`：展示正常/患病样本的异常分数分布，可观察到患病样本整体向高分（异常）区域偏移。
- `figures/roc_curve.png`：ROC 曲线距离对角线较远，AUC≈0.98，说明不同阈值下真阳率与假阳率区分明显。
- `figures/pr_curve.png`：PR 曲线在高召回区域仍保持较高精度，印证模型对少量阳性样本的排序能力。

**实现文件**：
- 训练&评估脚本：`thyroid/task4_unsupervised.py`
- 说明文档：`thyroid/task4_unsupervised.md`

