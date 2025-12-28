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
    """Load training features for zipper using only 'good' images."""
    features: List[np.ndarray] = []

    good_dir = DATA_DIR / "zipper" / "train" / "good"
    for fname in sorted(os.listdir(good_dir)):
        if not fname.lower().endswith(".png"):
            continue
        fpath = good_dir / fname
        features.append(load_image(fpath))

    X_train = np.vstack(features)
    print(
        f"[zipper] Loaded {X_train.shape[0]} normal training samples, "
        f"feature dim={X_train.shape[1]}"
    )
    return X_train


def load_test_features_and_labels() -> Tuple[np.ndarray, np.ndarray]:
    """Load test features and binary labels for zipper using JSON annotation.

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
        if not rel_path.startswith("zipper/"):
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
        f"[zipper] Loaded {X_test.shape[0]} test samples, "
        f"feature dim={X_test.shape[1]}"
    )
    print(
        f"[zipper] Test label distribution: "
        f"normal={np.sum(y_test == 1)}, anomaly={np.sum(y_test == -1)}"
    )
    return X_test, y_test


def train_oneclass_svm(X_train: np.ndarray) -> tuple[OneClassSVM, StandardScaler]:
    """Fit a One-Class SVM on normal zipper samples."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    clf.fit(X_train_scaled)

    return clf, scaler


def evaluate_and_plot(
    clf: OneClassSVM,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluate the anomaly detector for zipper and generate visualizations."""
    X_test_scaled = scaler.transform(X_test)

    y_pred = clf.predict(X_test_scaled)
    scores = -clf.decision_function(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    print("[zipper] Confusion matrix (rows=true, cols=pred, [normal, anomaly]):")
    print(cm)

    print("\n[zipper] Classification report:")
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
    print(f"[zipper] ROC-AUC (anomaly vs normal): {auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"One-Class SVM (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Zipper Anomaly Detection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_zipper_roc_curve.png", dpi=150)
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

    plt.title("Confusion Matrix (Zipper)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_zipper_confusion_matrix.png", dpi=150)
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
    plt.title("PCA of Zipper Test Features")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task2_zipper_pca_test_features.png", dpi=150)
    plt.close()


def main() -> None:
    print("=== Zipper: loading training data (normal samples only) ===")
    X_train = load_train_features()

    print("\n=== Zipper: training One-Class SVM anomaly detector ===")
    clf, scaler = train_oneclass_svm(X_train)

    print("\n=== Zipper: loading test data with labels ===")
    X_test, y_test = load_test_features_and_labels()

    print("\n=== Zipper: evaluating on test set and generating figures ===")
    evaluate_and_plot(clf, scaler, X_test, y_test)


if __name__ == "__main__":
    main()


