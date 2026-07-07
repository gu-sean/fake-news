import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
BEST_DIR = SOURCE_DIR / "best_txt"
OUTPUT_PATH = SOURCE_DIR.parent / "img" / "model_performance_comparison.png"

MODEL_FILES = {
    "Linear SVM": (
        BEST_DIR / "nbsvm_V1_log.txt",
        "Final test Accuracy",
        "Final test Macro-F1",
    ),
    "Logistic Regression": (BEST_DIR / "best_config_V4_2.txt", "Accuracy", "Macro F1"),
    "XGBoost": (BEST_DIR / "performance_xgboost_V2.txt", "Accuracy", "Macro F1"),
}


def extract_metric(text: str, metric: str) -> float:
    match = re.search(rf"^{re.escape(metric)}:\s*([0-9.]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Metric '{metric}' was not found.")
    return float(match.group(1))


def load_performance() -> tuple[list[str], list[float], list[float]]:
    models = []
    accuracy = []
    macro_f1 = []

    for model, (file_path, accuracy_metric, f1_metric) in MODEL_FILES.items():
        text = file_path.read_text(encoding="utf-8")
        models.append(model)
        accuracy.append(extract_metric(text, accuracy_metric))
        macro_f1.append(extract_metric(text, f1_metric))

    return models, accuracy, macro_f1


def plot_comparison(
    models: list[str],
    accuracy: list[float],
    macro_f1: list[float],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    y = np.arange(len(models))
    bar_height = 0.34

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    })

    fig, ax = plt.subplots(figsize=(11, 6.5))

    accuracy_bars = ax.barh(
        y + bar_height / 2,
        accuracy,
        height=bar_height,
        label="Accuracy",
        color="#3B82F6",
    )
    f1_bars = ax.barh(
        y - bar_height / 2,
        macro_f1,
        height=bar_height,
        label="Macro F1-score",
        color="#F59E0B",
    )

    ax.set_yticks(y, models)
    ax.set_xlim(0.55, 0.81)
    ax.set_xlabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold", pad=16)
    ax.legend(loc="lower right", frameon=True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.45)
    ax.yaxis.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax.bar_label(accuracy_bars, fmt="%.4f", padding=6, fontsize=13)
    ax.bar_label(f1_bars, fmt="%.4f", padding=6, fontsize=13)

    fig.tight_layout(pad=1.2)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    models, accuracy, macro_f1 = load_performance()
    plot_comparison(models, accuracy, macro_f1)

    for model, acc, f1 in zip(models, accuracy, macro_f1):
        print(f"{model}: Accuracy={acc:.4f}, Macro F1={f1:.4f}")
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
