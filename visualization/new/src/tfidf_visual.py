"""Visualize the most influential TF-IDF features as two word clouds."""

import pickle
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parents[3]

MODEL_PATH = PROJECT_ROOT / "V4_2" / "best_model_V4_2.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "fake-news" / "data" / "vector" / "korean_vectorizer.pkl"
OUTPUT_PATH = SOURCE_DIR.parent / "img" / "tfidf_wordcloud.png"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")

TOP_N = 100


def load_feature_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Load matching feature names and Logistic Regression coefficients."""
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

    return feature_names, coefficients


def select_top_features(
    feature_names: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the strongest positive and negative coefficient features."""
    positive_indices = np.argsort(coefficients)[-TOP_N:][::-1]
    negative_indices = np.argsort(coefficients)[:TOP_N]

    positive = {
        feature_names[index]: float(coefficients[index])
        for index in positive_indices
        if coefficients[index] > 0
    }
    negative = {
        feature_names[index]: float(abs(coefficients[index]))
        for index in negative_indices
        if coefficients[index] < 0
    }
    return positive, negative


def create_wordcloud(frequencies: dict[str, float], colormap: str) -> WordCloud:
    """Create a reproducible Korean word cloud from coefficient magnitudes."""
    if not frequencies:
        raise ValueError("No non-zero features were available for the word cloud.")

    return WordCloud(
        font_path=str(FONT_PATH),
        width=1500,
        height=1100,
        background_color="#FAFCFF",
        colormap=colormap,
        max_words=TOP_N,
        prefer_horizontal=0.92,
        relative_scaling=0.45,
        min_font_size=8,
        max_font_size=155,
        margin=5,
        random_state=42,
        collocations=False,
    ).generate_from_frequencies(frequencies)


def plot_wordclouds(
    positive_cloud: WordCloud,
    negative_cloud: WordCloud,
    positive_count: int,
    negative_count: int,
) -> None:
    """Place positive and negative feature clouds side by side."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.08, top=0.80, wspace=0.08)

    panels = [
        (
            axes[0],
            positive_cloud,
            "Positive Coefficients",
            f"Top {positive_count} features contributing toward fake news (label 1)",
            "#C2413B",
        ),
        (
            axes[1],
            negative_cloud,
            "Negative Coefficients",
            f"Top {negative_count} features contributing toward real news (label 0)",
            "#2563A6",
        ),
    ]

    for ax, cloud, title, subtitle, title_color in panels:
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_facecolor("#FAFCFF")
        ax.set_title(title, fontsize=19, fontweight="bold", color=title_color, pad=28)
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#657087",
        )
        ax.axis("off")

    fig.suptitle(
        "Top TF-IDF Features Learned by Logistic Regression",
        fontsize=25,
        fontweight="bold",
        color="#253047",
        y=0.93,
    )
    fig.text(
        0.5,
        0.875,
        "Word size represents the absolute magnitude of each learned coefficient.",
        ha="center",
        fontsize=12,
        color="#657087",
    )
    fig.text(
        0.5,
        0.035,
        "Positive coefficient → label 1 (fake news)     |     Negative coefficient → label 0 (real news)",
        ha="center",
        fontsize=10.5,
        color="#657087",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Korean font was not found: {FONT_PATH}")

    feature_names, coefficients = load_feature_coefficients()
    positive, negative = select_top_features(feature_names, coefficients)
    positive_cloud = create_wordcloud(positive, "Reds")
    negative_cloud = create_wordcloud(negative, "Blues")
    plot_wordclouds(positive_cloud, negative_cloud, len(positive), len(negative))

    print(f"Positive features visualized: {len(positive)}")
    print(f"Negative features visualized: {len(negative)}")
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
