"""任务3：时间序列预测 - 根据过去2小时天气情况预测下一时刻室外温度(OT)。

运行方式：
    python weather/task3_timeseries.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("weather.csv")
FIGURE_DIR = Path(__file__).with_name("figures")
WINDOW_SIZE = 12  # 12 * 10min = 2 hours
TRAIN_RATIO = 0.8
TARGET_COL = "OT"


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


@dataclass
class EvaluationResult:
    mae: float
    rmse: float
    mape: float
    r2: float


def load_weather_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_sliding_windows(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [col for col in df.columns if col != "date"]
    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    target_idx = feature_cols.index(TARGET_COL)

    total_steps = len(df)
    sample_count = total_steps - WINDOW_SIZE
    if sample_count <= 0:
        raise ValueError("样本数量不足，无法构造滑动窗口样本。")

    X = np.zeros((sample_count, WINDOW_SIZE, feature_matrix.shape[1]), dtype=np.float32)
    y = np.zeros(sample_count, dtype=np.float32)

    for idx in range(sample_count):
        start = idx
        end = idx + WINDOW_SIZE
        X[idx] = feature_matrix[start:end]
        y[idx] = feature_matrix[end, target_idx]

    return X, y


def flatten_and_split(X: np.ndarray, y: np.ndarray) -> DatasetSplits:
    sample_count = X.shape[0]
    train_count = int(sample_count * TRAIN_RATIO)
    if train_count == 0 or train_count == sample_count:
        raise ValueError("请调整 TRAIN_RATIO，当前无法进行有效的训练/测试划分。")

    X_flat = X.reshape(sample_count, -1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_flat[:train_count])
    X_test = scaler.transform(X_flat[train_count:])

    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y[:train_count],
        y_test=y[train_count:],
        scaler=scaler,
    )


def train_regressor(splits: DatasetSplits) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    model.fit(splits.X_train, splits.y_train)
    return model


def evaluate_predictions(
    splits: DatasetSplits, model: GradientBoostingRegressor
) -> EvaluationResult:
    y_pred = model.predict(splits.X_test)
    mae = mean_absolute_error(splits.y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(splits.y_test, y_pred))

    # 避免除零，将极小值裁剪
    y_true_safe = np.clip(np.abs(splits.y_test), a_min=1e-3, a_max=None)
    mape = np.mean(np.abs((splits.y_test - y_pred) / y_true_safe)) * 100
    r2 = r2_score(splits.y_test, y_pred)

    return EvaluationResult(mae=mae, rmse=rmse, mape=mape, r2=r2)


def plot_and_save_figures(
    splits: DatasetSplits,
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
) -> None:
    """绘制若干可视化结果并保存到 figures 目录。"""
    FIGURE_DIR.mkdir(exist_ok=True)

    # 重新获得测试集时间索引，便于绘制时间序列
    total_steps = len(df)
    sample_count = total_steps - WINDOW_SIZE
    train_count = int(sample_count * TRAIN_RATIO)

    # 对应于 y 的时间索引是从 WINDOW_SIZE 开始，长度与样本数一致
    # y_train / y_test 是按样本划分的，所以测试集起始索引用 train_count
    time_index = df["date"].iloc[WINDOW_SIZE : WINDOW_SIZE + sample_count].reset_index(drop=True)
    test_time_index = time_index.iloc[train_count:].reset_index(drop=True)

    y_test = splits.y_test
    y_pred = model.predict(splits.X_test)
    residuals = y_test - y_pred

    # 1) Predicted vs True values (scatter plot)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
    plt.xlabel("True OT (°C)")
    plt.ylabel("Predicted OT (°C)")
    plt.title("Test set: true vs predicted OT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "task3_y_true_vs_pred.png", dpi=200)
    plt.close()

    # 2) Time series comparison (first 500 test points for clarity)
    n_points = min(500, len(test_time_index))
    plt.figure(figsize=(10, 4))
    plt.plot(test_time_index.iloc[:n_points], y_test[:n_points], label="True OT")
    plt.plot(test_time_index.iloc[:n_points], y_pred[:n_points], label="Predicted OT", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("OT (°C)")
    plt.title("Test set time series (head): true vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "task3_timeseries_true_vs_pred.png", dpi=200)
    plt.close()

    # 3) Residuals over time
    plt.figure(figsize=(10, 3))
    plt.plot(test_time_index, residuals, label="Residual (y_true - y_pred)")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Residual (°C)")
    plt.title("Test set residuals over time")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "task3_residuals_over_time.png", dpi=200)
    plt.close()


def main() -> None:
    df = load_weather_dataframe()
    print(f"Loaded weather dataframe with {len(df)} rows.")

    X, y = build_sliding_windows(df)
    print(f"Constructed {X.shape[0]} samples with window size {WINDOW_SIZE}.")

    splits = flatten_and_split(X, y)
    print(
        f"Train samples: {len(splits.y_train)}, "
        f"Test samples: {len(splits.y_test)}, "
        f"Feature dim: {splits.X_train.shape[1]}"
    )

    model = train_regressor(splits)
    eval_result = evaluate_predictions(splits, model)
    plot_and_save_figures(splits, model, df)

    print("\n=== Evaluation on Test Set ===")
    print(f"MAE : {eval_result.mae:.3f} °C")
    print(f"RMSE: {eval_result.rmse:.3f} °C")
    print(f"MAPE: {eval_result.mape:.2f}%")
    print(f"R^2 : {eval_result.r2:.3f}")


if __name__ == "__main__":
    main()

