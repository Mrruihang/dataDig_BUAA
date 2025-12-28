# BUAA数据挖掘大作业总览（4 个项目）

本仓库包含 4 个相互独立的课程/作业项目：**图像聚类（DataMining）**、**图像异常检测（anomaly detection）**、**甲状腺疾病无监督检测（thyroid）**、**天气时间序列预测（weather）**。本文档提供每个项目的目标、数据位置、运行方式与输出说明。

> 环境建议：Python 3.9+（更推荐 3.10/3.11）。下文命令默认在**仓库根目录**执行。  
> Windows 下由于目录名包含空格（`anomaly detection`），建议对路径加引号。

---

## 目录结构

- `DataMining/homework1/`：任务 1（工业图像聚类：ResNet50 特征 + PCA + K-Means）  
- `anomaly detection/`：任务 2（工业图像异常检测：One-Class SVM；含 hazelnut/zipper 子任务与数据集）  
- `thyroid/`：任务 4（甲状腺数据：仅正常样本训练的一类分类/异常检测）  
- `weather/`：任务 3（天气时间序列：过去 2 小时预测下一时刻温度 OT）  

---

## 统一依赖安装（建议）

不同项目依赖高度重合，均为常见科学计算栈：

```bash
python -m pip install -U pip
python -m pip install numpy pandas scikit-learn matplotlib seaborn pillow tqdm
```

其中 **DataMining 聚类**用到 `torch/torchvision`（用于 ResNet50 特征提取）：

```bash
python -m pip install torch torchvision
```

如果你希望严格按项目作者提供的依赖版本，`DataMining/homework1/requirements.txt` 也可直接安装（注意其中含 `git+https://github.com/openai/CLIP.git` 等条目，可能需要可用的 Git/网络环境）：

```bash
python -m pip install -r DataMining/homework1/requirements.txt
```

---

## 项目 1：DataMining（工业图像无监督聚类）

### 目标

对课程提供的工业图像数据集（6 类、600 张）进行**无监督聚类**：

- **特征**：ImageNet 预训练 `ResNet50` 提取 2048 维特征  
- **降维**：PCA（最多 50 维，并额外保存 2D 投影用于可视化）  
- **聚类**：K-Means（`k=6`）  
- **离线评估**：读取 `cluster_labels.json` 仅用于计算 ARI/NMI/映射后 Accuracy，并生成可视化

### 数据位置（重要）

脚本会从 `DataMining/homework1/` 目录向上自动探测以下路径；你也可以通过参数显式指定：

- 数据集目录：`Cluster/dataset/`
- 标准标签：`Cluster/cluster_labels.json`

> 如果你的仓库里暂时没有 `Cluster/` 目录，请把课程数据放到上述位置，或在运行时传入 `--dataset` / `--labels`。

### 运行方式

1）生成聚类结果（会自动保存特征与中间结果）：

```bash
python DataMining/homework1/clustring.py
```

可选：显式指定数据与标签路径：

```bash
python DataMining/homework1/clustring.py --dataset "路径/到/Cluster/dataset" --labels "路径/到/Cluster/cluster_labels.json"
```

2）离线评估 + 可视化：

```bash
python DataMining/homework1/compare_results.py
```

可选：指定预测文件/标签文件：

```bash
python DataMining/homework1/compare_results.py --pred DataMining/homework1/clustering_results.json --labels "路径/到/Cluster/cluster_labels.json"
```

### 输出说明（主要文件）

输出位于 `DataMining/homework1/`：

- `clustering_results.json`：`{"image_name": cluster_id}`  
- `comparison_report.json`：ARI/NMI/Accuracy、簇-类别映射、错误样本列表等  
- `clustering_visualization.png` / `confusion_matrix.png` / `embedding_distribution.png` / `clustering_distribution.png`  
- `features_pca.npy` / `features_2d.npy` / `image_files.json`：用于评估与可视化对齐

---

## 项目 2：anomaly detection（工业图像异常检测：hazelnut / zipper）

### 目标

在**无监督异常检测**设定下，只用训练集的正常样本学习“正常模式”，再对测试样本输出异常判别与评估指标。

- **特征**：灰度化 + resize 到 `64×64` + 展平（4096 维），并在训练集上 `StandardScaler` 标准化  
- **模型**：`OneClassSVM(RBF)`  
- **评估**：混淆矩阵、分类报告、ROC-AUC，并保存 ROC 曲线 / PCA 可视化图

### 数据位置

数据已随仓库提供，位于：

- `anomaly detection/Image_Anomaly_Detection/`
  - `hazelnut/`、`zipper/`（含 `train/good`、`train/bad`、`test`）
  - `image_anomaly_labels.json`（测试标签，字段 `label ∈ {good, bad}`）

### 运行方式（分别跑两个子任务）

Windows PowerShell / CMD 建议加引号：

```bash
python "anomaly detection/hazelnut/task2_image_anomaly_hazelnut.py"
python "anomaly detection/zipper/task2_image_anomaly_zipper.py"
```

### 输出说明

- hazelnut 输出目录：`anomaly detection/hazelnut/figures/`
  - `task2_hazelnut_roc_curve.png`
  - `task2_hazelnut_confusion_matrix.png`
  - `task2_hazelnut_pca_test_features.png`
- zipper 输出目录：`anomaly detection/zipper/figures/`
  - `task2_zipper_roc_curve.png`
  - `task2_zipper_confusion_matrix.png`
  - `task2_zipper_pca_test_features.png`

---

## 项目 3：thyroid（任务 4：无监督疾病判断）

### 目标

训练集仅包含正常样本（label=0），在缺少患病样本的情况下学习正常分布，并将偏离正常分布的样本判为异常/患病（label=1）。

- **数据**：6 维特征  
- **方法**：`StandardScaler + OneClassSVM(RBF, nu=0.05, gamma='scale')`  
- **输出**：异常分数、预测标签、ROC/PR 曲线与分数分布图

### 数据位置

- `thyroid/thyroid/train-set.csv`
- `thyroid/thyroid/test-set.csv`（含 `label` 列作为测试评估用）

### 运行方式

```bash
python thyroid/task4_unsupervised.py
```

### 输出说明

输出位于 `thyroid/figures/`：

- `anomaly_score_distribution.png`
- `roc_curve.png`
- `pr_curve.png`

终端会打印数据集规模与指标（ROC-AUC、PR-AUC、Accuracy、Precision/Recall/F1、混淆矩阵等）。

---

## 项目 4：weather（任务 3：时间序列预测）

### 目标

使用最近 2 小时（12 个时间点，每 10 分钟一次）的全部气象特征，预测下一时间点的室外温度 `OT`。

- **构造样本**：滑动窗口把序列转为回归（`12 × 21 = 252` 维特征）  
- **划分**：按时间顺序 80% 训练 / 20% 测试，避免信息泄露  
- **模型**：`GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.9)`

### 数据位置

- `weather/weather.csv`（脚本按 `ISO-8859-1` 读取，并解析 `date` 列排序）

### 运行方式

```bash
python weather/task3_timeseries.py
```

### 输出说明

终端输出测试集 MAE / RMSE / MAPE / R²；图片输出到 `weather/figures/`：

- `task3_y_true_vs_pred.png`（散点：真实 vs 预测）
- `task3_timeseries_true_vs_pred.png`（前 500 个测试点的时间序列对比）
- `task3_residuals_over_time.png`（残差随时间变化）

---

## 常见问题（Windows）

- **路径含空格**：运行 `anomaly detection` 下脚本时务必对路径加引号，例如：  
  `python "anomaly detection/hazelnut/task2_image_anomaly_hazelnut.py"`
- **图像聚类数据缺失**：如果没有 `Cluster/dataset`，请把课程数据放到该目录，或用 `--dataset/--labels` 指定绝对路径。


