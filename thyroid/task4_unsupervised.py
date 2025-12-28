import pathlib
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class EvaluationResult:
    roc_auc: float
    pr_auc: float
    accuracy: float
    f1: float
    recall: float
    precision: float
    confusion: Dict[str, int]
    anomaly_scores: np.ndarray


def load_data(base_dir: pathlib.Path):
    train = pd.read_csv(base_dir / "train-set.csv")
    test = pd.read_csv(base_dir / "test-set.csv")
    X_train = train.values
    X_test = test.drop(columns=["label"]).values
    y_test = test["label"].values.astype(int)
    return X_train, X_test, y_test


def train_model(X_train: np.ndarray) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ocsvm",
                OneClassSVM(
                    kernel="rbf",
                    gamma="scale",
                    nu=0.05,  # assume <=5%异常
                ),
            ),
        ]
    )
    pipeline.fit(X_train)
    return pipeline


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationResult:
    decision_scores = model.decision_function(X_test)
    anomaly_scores = -decision_scores  # 越大越可能异常
    preds = (model.predict(X_test) == -1).astype(int)

    roc_auc = roc_auc_score(y_test, anomaly_scores)
    pr_auc = average_precision_score(y_test, anomaly_scores)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, output_dict=True, zero_division=0
    )
    confusion = confusion_matrix(y_test, preds)

    return EvaluationResult(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=accuracy,
        f1=report["1"]["f1-score"],
        recall=report["1"]["recall"],
        precision=report["1"]["precision"],
        confusion={
            "tn": int(confusion[0, 0]),
            "fp": int(confusion[0, 1]),
            "fn": int(confusion[1, 0]),
            "tp": int(confusion[1, 1]),
        },
        anomaly_scores=anomaly_scores,
    )


def dataset_stats(X_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    return {
        "train_samples": X_train.shape[0],
        "test_samples": y_test.shape[0],
        "test_positive_ratio": float(y_test.mean()),
        "feature_dim": X_train.shape[1],
    }


def save_visualizations(
    anomaly_scores: np.ndarray,
    y_test: np.ndarray,
    roc_auc: float,
    pr_auc: float,
    output_dir: pathlib.Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(
        anomaly_scores[y_test == 0],
        bins=40,
        alpha=0.7,
        label="Normal",
        density=True,
    )
    plt.hist(
        anomaly_scores[y_test == 1],
        bins=40,
        alpha=0.7,
        label="Positive",
        density=True,
    )
    plt.xlabel("Anomaly score (higher = more abnormal)")
    plt.ylabel("Density")
    plt.title("Anomaly score distribution")
    plt.legend()
    plt.tight_layout()
    score_hist_path = output_dir / "anomaly_score_distribution.png"
    plt.savefig(score_hist_path, dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(
        recall,
        precision,
        label=f"PR curve (AUC={pr_auc:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    pr_path = output_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    return score_hist_path, roc_path, pr_path


def main():
    base_dir = pathlib.Path(__file__).resolve().parent / "thyroid"
    X_train, X_test, y_test = load_data(base_dir)
    stats = dataset_stats(X_train, y_test)
    model = train_model(X_train)
    result = evaluate(model, X_test, y_test)
    figures_dir = pathlib.Path(__file__).resolve().parent / "figures"
    figure_paths = save_visualizations(
        anomaly_scores=result.anomaly_scores,
        y_test=y_test,
        roc_auc=result.roc_auc,
        pr_auc=result.pr_auc,
        output_dir=figures_dir,
    )

    print("=== 数据集规模 ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== 评估指标 ===")
    print(f"ROC-AUC: {result.roc_auc:.4f}")
    print(f"PR-AUC: {result.pr_auc:.4f}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision (positive): {result.precision:.4f}")
    print(f"Recall (positive): {result.recall:.4f}")
    print(f"F1 (positive): {result.f1:.4f}")
    print(f"Confusion Matrix: {result.confusion}")
    print("\n=== 可视化输出 ===")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()

