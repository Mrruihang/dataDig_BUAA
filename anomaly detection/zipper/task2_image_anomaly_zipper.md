## 任务2（zipper）：图像异常检测

### 2.0 问题的形式化描述
- 子任务只关注 `zipper/` 这一类工业视觉数据：  
  - 训练集在 `Image_Anomaly_Detection/zipper/train/` 下，包含 **200 张正常图像**（`good/`）和 **50 张异常图像**（`bad/`）；  
  - 测试集在 `Image_Anomaly_Detection/zipper/test/` 下，正常与异常混合分布，具体标签由根目录 `image_anomaly_labels.json` 提供（字段 `label ∈ {good, bad}`，并有 `original_category: "zipper"`）。
- 本子任务仍然采用 **无监督异常检测** 设定：  
  - 仅使用 `zipper/train/good/` 中的图像学习“正常拉链”的图像模式；  
  - 测试阶段根据样本与正常模式的偏离程度判断其是否异常。
- 形式化地，对 zipper 的图像 \(I \in \mathbb{R}^{H\times W\times C}\)，通过特征提取函数 \(\phi_{zip}(\cdot)\) 得到向量 \(x = \phi_{zip}(I) \in \mathbb{R}^d\)。  
  基于正常样本集合 \(\{x_i^{(normal)}\}\) 训练一类模型 \(g_{zip}(\cdot)\) 以覆盖“正常拉链”的高密度区域，对任意测试样本 \(x\) 通过得分 \(s = g_{zip}(x)\) 与阈值比较，给出正常/异常判断。

### 2.1 如何处理图像特征
- **预处理与特征抽取（zipper）**：  
  - 使用 `PIL` 读取 zipper 图像，转为灰度图（`convert("L")`），统一缩放到 `64×64` 像素；  
  - 像素值缩放到 \([0,1]\) 后按行展开，得到长度为 \(64\times64=4096\) 的向量：  
    \[ x = \text{vec}\left(\frac{I_{gray}}{255}\right) \in \mathbb{R}^{4096}. \]
- **标准化**：  
  - 使用 zipper 训练集正常样本上的统计量拟合 `StandardScaler`，并用该 scaler 统一变换 zipper 的测试特征；  
  - 保证模型的决策边界是围绕 zipper 的正常模式建模的。
- **PCA 可视化**：  
  - 评估阶段对 zipper 测试特征使用 `PCA(n_components=2)`，以 2D 投影的方式展示正常与异常拉链样本的分布。

### 2.2 异常检测模型与设计思想（zipper）
- **模型：One-Class SVM（zipper 专用）**  
  - 使用 `sklearn.svm.OneClassSVM`，只在 `zipper/train/good/` 的正常样本上拟合：  
    - 核函数：`RBF`；  
    - `gamma='scale'`；  
    - `nu=0.1`，控制预计异常比例与决策边界紧致程度。
  - 模型在高维特征空间构造一个覆盖 zipper 正常样本的区域，正常样本预测为 +1，异常样本为 -1。
- **设计思路**：  
  - 将 zipper 从整体任务中单独拆出，可以更有针对性地观察拉链类缺陷的检测效果；  
  - 特征与模型设计和 hazelnut 子任务保持一致，便于在相同条件下对比不同类别的异常检测难度；  
  - 若 zipper 子任务明显优于 hazelnut，说明在相同特征下，zipper 的异常图像在像素空间更易与正常区分。
- **实现脚本**：`python "anomaly detection/zipper/task2_image_anomaly_zipper.py"`：  
  - 读取 `zipper/train/good/` 训练 zipper 专用 One-Class SVM；  
  - 从 `image_anomaly_labels.json` 中筛选 `"zipper/"` 开头的条目构造测试集，并将 `good/bad` 转换为 {1, -1}；  
  - 输出 zipper 的混淆矩阵、分类报告、ROC-AUC，并保存对应可视化图片。

### 2.3 评估与可视化结果（zipper）
在 zipper 子任务上，一次脚本运行的示例结果如下：

- **混淆矩阵**（行：真实，列：预测，顺序为 [normal, anomaly]）：  
  \[
  \begin{bmatrix}
  22 & 10 \\
   4 & 11
  \end{bmatrix}
  \]
- **分类指标（把 anomaly 视作“正类”进行解读）**：  
  - **Accuracy** ≈ 0.70：整体约 70% 的 zipper 测试样本被正确识别；  
  - **normal（正常）类**：召回率约 0.69，说明多数正常拉链能被正确识别；  
  - **anomaly（异常）类**：召回率约 0.73、精度约 0.52，模型对异常拉链较敏感且误报控制尚可；  
  - **ROC-AUC（异常 vs 正常）** ≈ 0.80，说明在 zipper 子任务上，模型在区分异常与正常方面有较强的排序能力。

脚本会在 `anomaly detection/zipper/figures/` 目录下输出 zipper 专用的三张图：

- **`task2_zipper_roc_curve.png`：ROC 曲线（zipper）**  
  - 曲线明显高于随机对角线，说明在 zipper 子任务中异常检测效果较好。

- **`task2_zipper_confusion_matrix.png`：混淆矩阵热力图**  
  - 直观呈现 zipper 正常与异常预测情况，可用于观察误报/漏报的类型和数量。

- **`task2_zipper_pca_test_features.png`：PCA 投影（zipper）**  
  - 展示 zipper 的正常样本与异常样本在 2D 特征空间上的分布，相比 hazelnut 通常具有更好的可分性，这与较高的 ROC-AUC 指标相符。

整体上，zipper 子任务在与 hazelnut 相同的特征与模型设定下表现更好，说明拉链类异常在当前特征空间中更易被 One-Class SVM 捕捉，为后续跨类别对比和方法改进提供了有价值的参考基线。


