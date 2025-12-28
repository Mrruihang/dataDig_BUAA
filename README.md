## 图像异常 / 天气预测 / 甲状腺筛查：三项目总览

1. 图像异常检测项目（任务2）
- **整体结构**  
  ```
  anomaly detection/
  ├─ task2_image_anomaly.md                # 项目说明（本节来源）
  ├─ task2_image_anomaly.py                # hazelnut + zipper 通用脚本
  ├─ figures/                              # 通用可视化
  │   ├─ task2_confusion_matrix.png
  │   ├─ task2_pca_test_features.png
  │   └─ task2_roc_curve.png
  ├─ hazelnut/ & zipper/                   # 各子类的定制脚本/图
  └─ Image_Anomaly_Detection/              # 原始数据（train/test/labels）
  ```
- **项目介绍**  
  - 数据：两个工业视觉类别（hazelnut、zipper），每类 200 正常 + 50 异常训练样本，测试集混合并由 `image_anomaly_labels.json` 提供标签。  
  - 方法：统一灰度化、缩放到 `64×64`、展平到 4096 维并标准化，使用合并的正常样本训练 `One-Class SVM(RBF, nu=0.1, gamma='scale')`，实现无监督异常检测基线。  
  - 评估与输出：脚本一次性跑完两类测试，控制台打印 Accuracy≈0.70、Normal Recall≈0.93、Anomaly Recall≈0.13、ROC-AUC≈0.62，同时生成 ROC 曲线、混淆矩阵、PCA 投影三张图片供复盘。

2. 天气时间序列预测项目（任务3）
- **整体结构**  
  ```
  weather/
  ├─ task3_timeseries.md        # 项目说明
  ├─ task3_timeseries.py        # 读数、建模、评估主脚本
  ├─ weather.csv                # 原始多变量气象序列
  └─ figures/
      ├─ task3_y_true_vs_pred.png
      ├─ task3_timeseries_true_vs_pred.png
      └─ task3_residuals_over_time.png
  ```
- **项目介绍**  
  - 数据：`weather.csv` 内含德国气象站 6 个月、每 10 分钟一次的 21 维特征（26,200 时间点），目标是根据最近 2 小时（12 个时间点）预测下一时刻室外温度 `OT`。  
  - 方法：滑动窗口展开为 252 维特征，按时间顺序 8:2 划分训练/测试，并基于训练集拟合 `StandardScaler`。模型选用 `GradientBoostingRegressor`（400 棵深度 3 的树、lr=0.05、subsample=0.9），在纯 CPU 环境下兼顾速度与非线性表示。  
  - 评估与输出：测试集上 MAE≈9.59℃、RMSE≈38.31℃、MAPE≈2.15%、R²≈-1.87，表明相对误差低但绝对刻度偏差大。脚本生成三张图（真值-预测散点、时间轴对比、残差随时间）帮助定位系统性误差。

3. 甲状腺无监督筛查项目（任务4）
- **整体结构**  
  ```
  thyroid/
  ├─ task4_unsupervised.md      # 项目说明
  ├─ task4_unsupervised.py      # 预处理 + One-Class SVM
  ├─ figures/
  │   ├─ anomaly_score_distribution.png
  │   ├─ pr_curve.png
  │   └─ roc_curve.png
  └─ thyroid/
      ├─ train-set.csv          # 1839 条正常样本
      └─ test-set.csv           # 1933 条混合样本
  ```
- **项目介绍**  
  - 数据：6 维甲状腺指标，训练集仅含正常样本，测试集含正常与患病（患病约 4.86%）。  
  - 方法：`StandardScaler` 统一量纲后，使用 `One-Class SVM(RBF, nu=0.05, gamma='scale')` 拟合正常分布，`decision_function<0` 判为潜在患病，可直接运行 `python thyroid/task4_unsupervised.py` 复现。  
  - 评估与输出：ROC-AUC≈0.975、PR-AUC≈0.782、Accuracy≈0.93、精度≈0.41、召回≈0.94。高召回满足“宁可多报”的筛查需求；若需降低误报，可调节阈值或 `nu`。配套生成异常分数分布、ROC、PR 三张图，直观展示排序能力。


- 三个项目均已形成“脚本 + 说明文档 + 评估图”闭环，    
- 数据路径默认为 repo 相对路径，如迁移请同步更新文档中的命令示例。  



