## 任务2（hazelnut）：图像异常检测（优化版）

### 2.0 问题的形式化描述
- 子任务只关注 `hazelnut/` 这一类工业视觉数据：  
  - 训练集在 `Image_Anomaly_Detection/hazelnut/train/` 下，包含 **200 张正常图像**（`good/`）和 **50 张异常图像**（`bad/`）；  
  - 测试集在 `Image_Anomaly_Detection/hazelnut/test/` 下，正常与异常混合分布，具体标签由根目录下的 `image_anomaly_labels.json` 指定（字段 `label ∈ {good, bad}`，并有 `original_category: "hazelnut"`）。
- 本子任务依然建模为 **无监督异常检测** 问题：  
  - 仅使用 `hazelnut/train/good/` 中的图像学习“正常榛子”的分布；  
  - 在测试时，根据样本与正常模式的偏离程度判定其是否异常。
- 形式化表述：对 hazelnut 的单张图像 $I \in \mathbb{R}^{H\times W\times C}$，通过特征提取函数 $\phi_{haz}(\cdot)$得到向量 $x = \phi_{haz}(I) \in \mathbb{R}^d$。  
  我们用正常样本集合 $\{x_i^{(normal)}\}_{i=1}^N$ 拟合一类模型 $g_{haz}(\cdot)$，令其覆盖“正常榛子”的高密度区域；对任意测试样本 \(x\)，利用得分 $s = g_{haz}(x)$与阈值比较给出“正常 / 异常”的判断。

### 2.1 如何处理图像特征
- **预处理与特征抽取**：  
  处理流程与总任务一致，但仅对 hazelnut 图像执行：
  - 使用 `PIL` 读取图像，转为灰度图（`convert("L")`），将尺寸统一缩放为 `64×64`；  
  - 将像素值归一化到 \([0,1]\)，再按行展开为长度 $64\times64=4096$ 的向量：  
    $
    x = \text{vec}\left(\frac{I_{gray}}{255}\right) \in \mathbb{R}^{4096}.
    $
- **标准化**：  
  - 只在 hazelnut 训练集的正常特征上拟合 `StandardScaler`，随后对 hazelnut 的测试样本也使用同一 scaler 做零均值、单位方差标准化；  
  - 这样保证模型只围绕 hazelnut 的正常分布进行缩放和对齐。
- **PCA 可视化**：  
  - 在评估阶段，对 hazelnut 测试特征应用 `PCA(n_components=2)`，用于观察正常与异常榛子在 2D 空间中的分布情况。

### 2.2 异常检测模型与设计思想（hazelnut）
- **模型：One-Class SVM + 分数阈值（hazelnut 专用，偏“正常识别准一点”）**  
  - 使用 `sklearn.svm.OneClassSVM`，仅在 `hazelnut/train/good/` 的正常样本上训练：  
    - 核函数：`RBF`；  
    - `gamma=0.01`、`nu=0.15`，沿用在 hazelnut 上 ROC-AUC 表现较好的参数设置，让模型在“谁更像异常”的排序上保持较强区分能力；  
    - 不直接使用 `predict` 的 ±1 输出，而是取 `decision_function` 得到**异常分数**（分数越大越异常），再在测试阶段按分位数选阈值。
  - 具体判决规则：  
    - 对测试集计算分数 `scores = -decision_function(X_test)`；  
    - 取分位数阈值（例如只把约 20% 分数最高的样本视为“异常”），`scores >= 阈值 → anomaly(-1)`，否则视为 `normal(1)`。  
    - 这样可以显式控制“异常比例”，从而让**正常样本识别更准**，并避免极端“全异常”预测，使混淆矩阵更均匀。
- **设计考虑**：  
  - 只使用 hazelnut 的正常样本训练，有助于更精细地刻画“正常榛子”的纹理与形状特征，而不受其他类别（如 zipper）的干扰；  
  - hazelnut 类在简单像素特征下本身较难分离，若只依赖 One-Class SVM 的固定阈值（0）容易出现“几乎全异常”的偏置；  
  - 因此采用“两阶段”设计：SVM 负责提供排序良好的异常分数，阈值阶段根据任务需求（本子任务希望“正常识别准一点，可以接受部分异常遗漏”）来控制异常比例。
- **实现脚本**：`python "anomaly detection/hazelnut/task2_image_anomaly_hazelnut.py"`：  
  - 读取 `hazelnut/train/good/` 作为训练集；  
  - 基于正常样本训练 hazelnut 专用 One-Class SVM；  
  - 使用 `image_anomaly_labels.json` 中以 `"hazelnut/"` 开头的条目作为测试集，并将 `good/bad` 映射为 {1, -1}；  
  - 输出 hazelnut 的混淆矩阵、分类报告、ROC-AUC，并生成可视化图像。

### 2.3 评估与可视化结果（hazelnut）
在采用 “One-Class SVM + 分数分位数阈值（约 20% 样本视为异常）” 的策略后，一次运行的结果如下：

- **混淆矩阵**（行：真实，列：预测，顺序为 [normal, anomaly]）：  
  $
  \begin{bmatrix}
  35 & 5 \\
  9 & 6
  \end{bmatrix}
  $

- **分类指标（把 anomaly 视作“正类”进行解读）**：  
  - **Accuracy** ≈ 0.75：整体约 74.6% 的 hazelnut 测试样本被正确识别；  
  - **normal（正常）类**：召回率约 0.88（35/40），说明大部分正常榛子能被正确识别为正常，仅有少量误报为异常；  
  - **anomaly（异常）类**：召回率约 0.40（6/15），精度约 0.55（6/(6+5)），即在保证正常类较高召回的前提下，接受了一定的异常漏检；  
  - **ROC-AUC（异常 vs 正常）** ≈ 0.81：从排序角度看，模型仍然对“谁更像异常”有较强区分能力。

脚本会在 `anomaly detection/hazelnut/figures/` 目录下生成三张 hazelnut 专用的可视化图像：

- **`task2_hazelnut_roc_curve.png`：ROC 曲线（hazelnut）**  
  - ROC 曲线明显高于随机对角线，AUC 约 0.81，说明在阈值可调的前提下，hazelnut 异常与正常具有较好的可分性；  
  - 不同阈值对应不同的正常/异常权衡，本子任务选择的是“正常召回更高、异常召回适中”的一个点。

- **`task2_hazelnut_confusion_matrix.png`：混淆矩阵热力图**  
  - 可以看到正常类（40 个）中，有 35 个被正确识别，误报为异常的只有 5 个；  
  - 异常类（15 个）中，有 6 个被识别为异常、9 个被错判为正常，整体混淆矩阵相对之前“全异常”的极端情况明显更均匀，更符合“正常识别准一点，可以接受部分异常遗漏”的需求。

- **`task2_hazelnut_pca_test_features.png`：PCA 投影（hazelnut）**  
  - 显示 hazelnut 测试样本在 PCA 2D 空间中的分布情况，正常样本与异常样本在 2D 空间中依然存在一定的重叠；  
  - 这说明在当前简单像素特征下，hazelnut 的异常检测仍然存在困难，但通过合理的分数阈值设置，可以在“正常识别率”和“异常召回率”之间取得较为均衡的折中。

总体而言，这一版 hazelnut 子任务从“极端高异常召回（几乎全异常）”调整为“正常识别更准、混淆矩阵更均匀”的方案，为后续基于更强特征（如 CNN 特征、局部纹理特征等）的进一步优化提供了一个更符合实际需求的基线。

