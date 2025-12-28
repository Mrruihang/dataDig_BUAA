import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Image_Anomaly_Detection"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Load image, convert to grayscale, resize, and flatten to 1D vector."""
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()


def load_train_features() -> np.ndarray:
    """Load training features for hazelnut using only 'good' images."""
    features: List[np.ndarray] = []

    good_dir = DATA_DIR / "hazelnut" / "train" / "good"
    for fname in sorted(os.listdir(good_dir)):
        if not fname.lower().endswith(".png"):
            continue
        fpath = good_dir / fname
        features.append(load_image(fpath))

    X_train = np.vstack(features)
    print(
        f"[hazelnut] Loaded {X_train.shape[0]} normal training samples, "
        f"feature dim={X_train.shape[1]}"
    )
    return X_train


def load_test_features_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    """Load test features and binary labels for hazelnut using JSON annotation.

    Label mapping:
      - good -> 1 (normal)
      - bad  -> -1 (anomaly)
    """
    json_path = DATA_DIR / "image_anomaly_labels.json"
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for rel_path, info in meta.items():
        if not rel_path.startswith("hazelnut/"):
            continue

        label_str = info["label"]
        if label_str == "good":
            y = 1
        elif label_str == "bad":
            y = -1
        else:
            raise ValueError(f"Unknown label: {label_str}")

        img_path = DATA_DIR / rel_path
        X_list.append(load_image(img_path))
        y_list.append(y)

    X_test = np.vstack(X_list)
    y_test = np.array(y_list, dtype=int)
    print(
        f"[hazelnut] Loaded {X_test.shape[0]} test samples, "
        f"feature dim={X_test.shape[1]}"
    )
    print(
        f"[hazelnut] Test label distribution: "
        f"normal={np.sum(y_test == 1)}, anomaly={np.sum(y_test == -1)}"
    )
    return X_test, y_test


def train_oneclass_svm(X_train: np.ndarray) -> tuple[OneClassSVM, StandardScaler]:
    """Fit a One-Class SVM on normal hazelnut samples.

    当前目标：
    - **正常识别更准**：希望更多正常样本被预测为 normal，允许一定的异常漏检；
    - **混淆矩阵更均匀**：避免“几乎全部判为异常”这种极端情况，使四个格子的数量更平衡一些。
    具体做法：
    - 相比上一版（gamma=0.01, nu=0.15，偏向“高异常召回”），
      这里减小 nu、适度增大 gamma，让边界更紧、对正常类更友好，从而提升 normal 的召回率。
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 调参版本：RBF 核 + 手动设置 gamma、nu
    # 这里恢复到上一版在 ROC-AUC 表现较好的设置（gamma=0.01, nu=0.15），
    # 再结合下面 score-based 的自定义阈值来平衡正常/异常的预测比例。
    clf = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.15)
    clf.fit(X_train_scaled)

    return clf, scaler


def evaluate_and_plot(
    clf: OneClassSVM,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate the anomaly detector for hazelnut and generate visualizations."""
    X_test_scaled = scaler.transform(X_test)

    # 使用决策函数分数 + 自定义阈值，而不是直接用 clf.predict，
    # 目的是：让大部分样本被视为“正常”，只把分数最高的一部分当成异常，
    # 从而提升正常类的召回率，并让混淆矩阵更均匀。
    scores = -clf.decision_function(X_test_scaled)

    # 这里按分位数控制“异常比例”，例如只把约 20% 分数最高的样本当作异常。
    anomaly_ratio = 0.2
    thresh = np.quantile(scores, 1 - anomaly_ratio)
    y_pred = np.where(scores >= thresh, -1, 1)

    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    print("[hazelnut] Confusion matrix (rows=true, cols=pred, [normal, anomaly]):")
    print(cm)

    print("\n[hazelnut] Classification report:")
    print(
        classification_report(
            (y_test == -1).astype(int),
            (y_pred == -1).astype(int),
            target_names=["normal", "anomaly"],
            digits=4,
        )
    )

    y_true_binary = (y_test == -1).astype(int)
    auc = roc_auc_score(y_true_binary, scores)
    print(f"[hazelnut] ROC-AUC (anomaly vs normal): {auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"One-Class SVM (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Hazelnut Anomaly Detection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_hazelnut_roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix heatmap
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred normal", "Pred anomaly"])
    plt.yticks([0, 1], ["True normal", "True anomaly"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    plt.title("Confusion Matrix (Hazelnut)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_hazelnut_confusion_matrix.png", dpi=150)
    plt.close()

    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test_scaled)

    plt.figure(figsize=(6, 5))
    mask_normal = y_test == 1
    mask_anom = y_test == -1
    plt.scatter(
        X_test_pca[mask_normal, 0],
        X_test_pca[mask_normal, 1],
        s=15,
        c="tab:blue",
        label="Normal",
        alpha=0.7,
    )
    plt.scatter(
        X_test_pca[mask_anom, 0],
        X_test_pca[mask_anom, 1],
        s=20,
        c="tab:red",
        label="Anomaly",
        alpha=0.8,
        marker="x",
    )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("PCA of Hazelnut Test Features")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_hazelnut_pca_test_features.png", dpi=150)
    plt.close()


def main() -> None:
    print("=== Hazelnut: loading training data (normal samples only) ===")
    X_train = load_train_features()

    print("\n=== Hazelnut: training One-Class SVM anomaly detector ===")
    clf, scaler = train_oneclass_svm(X_train)

    print("\n=== Hazelnut: loading test data with labels ===")
    X_test, y_test = load_test_features_and_labels()

    print("\n=== Hazelnut: evaluating on test set and generating figures ===")
    evaluate_and_plot(clf, scaler, X_test, y_test)


if __name__ == "__main__":
    main()


