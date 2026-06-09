"""Visualize the strongest TF-IDF feature coefficients as horizontal bars."""

import pickle
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parents[3]

MODEL_PATH = PROJECT_ROOT / "V4_2" / "best_model_V4_2.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "fake-news" / "data" / "vector" / "korean_vectorizer.pkl"
OUTPUT_PATH = SOURCE_DIR.parent / "img" / "tfidf_coefficient_bars.png"

TOP_N = 15
POSITIVE_COLOR = "#E2554F"
NEGATIVE_COLOR = "#3478B9"
DARK = "#253047"
MUTED = "#657087"


def load_top_features() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Load the top positive and negative Logistic Regression coefficients."""
    model = joblib.load(MODEL_PATH)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = np.asarray(model.coef_[0])

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Feature count mismatch: "
            f"vectorizer={len(feature_names):,}, model={len(coefficients):,}"
        )
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], received {model.classes_}.")

    positive_indices = np.argsort(coefficients)[-TOP_N:][::-1]
    negative_indices = np.argsort(coefficients)[:TOP_N]

    # Reverse each list so the strongest feature appears at the top of barh plots.
    positive_indices = positive_indices[::-1]
    negative_indices = negative_indices[::-1]

    return (
        feature_names[positive_indices].tolist(),
        coefficients[positive_indices],
        feature_names[negative_indices].tolist(),
        coefficients[negative_indices],
    )


def style_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color=DARK, linewidth=1.1)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.yaxis.grid(False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=10)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10, length=0, pad=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_axisbelow(True)


def add_value_labels(ax: plt.Axes, bars, values: np.ndarray, padding: float) -> None:
    for bar, value in zip(bars, values):
        x = value + padding if value >= 0 else value - padding
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            # The negative panel reverses its x-axis, so "left" places the
            # label visually outside the right end of the negative bar.
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=DARK,
        )


def plot_coefficients(
    positive_words: list[str],
    positive_values: np.ndarray,
    negative_words: list[str],
    negative_values: np.ndarray,
) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    max_abs = float(max(positive_values.max(), abs(negative_values.min())))
    axis_limit = max_abs * 1.18
    label_padding = max_abs * 0.018

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.13, top=0.78, wspace=0.28)

    positive_bars = axes[0].barh(
        positive_words,
        positive_values,
        color=POSITIVE_COLOR,
        alpha=0.92,
        height=0.68,
    )
    negative_bars = axes[1].barh(
        negative_words,
        negative_values,
        color=NEGATIVE_COLOR,
        alpha=0.92,
        height=0.68,
    )

    axes[0].set_xlim(0, axis_limit)
    # Reverse the negative axis so both panels read outward from zero.
    axes[1].set_xlim(0, -axis_limit)

    axes[0].set_title(
        "Positive Coefficients",
        fontsize=19,
        fontweight="bold",
        color="#C2413B",
        pad=20,
    )
    axes[1].set_title(
        "Negative Coefficients",
        fontsize=19,
        fontweight="bold",
        color="#2563A6",
        pad=20,
    )

    axes[0].text(
        0.5,
        1.01,
        "Top 15 features contributing toward fake news (label 1)",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )
    axes[1].text(
        0.5,
        1.01,
        "Top 15 features contributing toward real news (label 0)",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    for ax in axes:
        style_axis(ax)
        ax.set_xlabel("Logistic Regression coefficient", fontsize=11, color=DARK, labelpad=12)

    add_value_labels(axes[0], positive_bars, positive_values, label_padding)
    add_value_labels(axes[1], negative_bars, negative_values, label_padding)

    fig.suptitle(
        "Top TF-IDF Feature Coefficients",
        fontsize=25,
        fontweight="bold",
        color=DARK,
        y=0.93,
    )
    fig.text(
        0.5,
        0.875,
        "Longer bars indicate a stronger influence on the model prediction.",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.055,
        "Both panels use the same coefficient scale for direct comparison.",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    positive_words, positive_values, negative_words, negative_values = load_top_features()
    plot_coefficients(positive_words, positive_values, negative_words, negative_values)

    print(f"Positive features visualized: {len(positive_words)}")
    print(f"Negative features visualized: {len(negative_words)}")
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
