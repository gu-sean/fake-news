import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from kiwipiepy import Kiwi
from wordcloud import WordCloud


LABEL_COL = "label"
TEXT_COL = "clean_message"

# Kiwi Korean POS tags and NLTK English POS prefixes used for nouns, verbs, and adjectives.
KOREAN_POS_TAGS = {"NNG", "NNP", "VV", "VA"}
ENGLISH_POS_PREFIXES = ("NN", "VB", "JJ")

# Keep Korean-only and English-only tokens separated for clearer visual analysis.
KOREAN_RE = re.compile(r"^[\uac00-\ud7a3]+$")
ENGLISH_RE = re.compile(r"^[A-Za-z]+$")

DEFAULT_FONT_PATHS = [
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
]

# Common words that tend to dominate counts but add little meaning to the analysis.
DEFAULT_KOREAN_STOPWORDS = {
    "\uc788\ub2e4",
    "\uc5c6\ub2e4",
    "\ud558\ub2e4",
    "\ub418\ub2e4",
    "\uc774\ub2e4",
    "\uc544\ub2c8\ub2e4",
    "\uac19\ub2e4",
    "\ub9d0\ud558\ub2e4",
    "\ubc1d\ud788\ub2e4",
    "\uc804\ud558\ub2e4",
    "\uc124\uba85\ud558\ub2e4",
    "\ubcf4\uc774\ub2e4",
    "\ubcf4\ub2e4",
    "\uc704\ud558\ub2e4",
    "\ub300\ud558\ub2e4",
    "\ub530\ub974\ub2e4",
    "\uc9c0\ub098\ub2e4",
    "\ud1b5\ud558\ub2e4",
    "\ub098\uc624\ub2e4",
    "\ub4e4\ub2e4",
    "\uac00\ub2e4",
    "\uc624\ub2e4",
    "\uc8fc\ub2e4",
    "\ubc1b\ub2e4",
    "\ub9ce\ub2e4",
    "\ub192\ub2e4",
    "\ud06c\ub2e4",
    "\uc791\ub2e4",
    "\ub54c\ubb38",
    "\uad00\ub828",
    "\uc774\ubc88",
    "\uc9c0\ub09c",
    "\ucd5c\uadfc",
    "\uc624\ub298",
}

DEFAULT_ENGLISH_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "but",
    "with",
    "from",
    "that",
    "this",
    "you",
    "your",
    "their",
    "they",
    "will",
    "would",
    "could",
    "should",
    "about",
    "into",
    "over",
    "under",
    "than",
    "then",
    "there",
    "here",
    "said",
}


def ensure_nltk_tagger() -> None:
    """Check whether the NLTK English POS tagger data is available."""
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        raise LookupError(
            "NLTK tagger data is missing. Run: "
            "python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\""
        )


def find_korean_font(custom_font_path: str | None = None) -> str | None:
    """Find a Korean font so Korean words render correctly in matplotlib and WordCloud."""
    if custom_font_path:
        font_path = Path(custom_font_path)
        if not font_path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        return str(font_path)

    for font_path in DEFAULT_FONT_PATHS:
        if font_path.exists():
            return str(font_path)

    return None


def configure_plot_style(font_path: str | None) -> None:
    """Apply common chart styling and font settings."""
    sns.set_theme(style="whitegrid", font_scale=1.0)

    if font_path and "malgun" in font_path.lower():
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


def load_stopwords(stopwords_path: str | None) -> set[str]:
    """Load optional user-defined stopwords from a UTF-8 text file."""
    if not stopwords_path:
        return set()

    path = Path(stopwords_path)
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def normalize_korean_token(token: str, tag: str) -> str:
    """Attach 'da' ending to Korean verb/adjective stems so they are easier to read."""
    if tag in {"VV", "VA"}:
        return f"{token}\ub2e4"
    return token


def collect_english_candidates(message: str, min_length: int, stopwords: set[str]) -> list[str]:
    """Collect English-looking words before running POS tagging."""
    words = []
    for raw_word in message.split():
        word = raw_word.lower()
        if len(word) < min_length:
            continue
        if word in stopwords:
            continue
        if ENGLISH_RE.fullmatch(word):
            words.append(word)
    return words


def update_english_counts(
    english_counts: Counter,
    candidates: list[str],
    batch_size: int = 5000,
) -> None:
    """POS-tag English words in batches and count only nouns, verbs, and adjectives."""
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        tagged_words = nltk.pos_tag(batch)
        english_counts.update(
            word
            for word, tag in tagged_words
            if tag.startswith(ENGLISH_POS_PREFIXES)
        )


def count_pos_words(
    csv_path: Path,
    real_label: int,
    chunksize: int,
    min_korean_length: int,
    min_english_length: int,
    korean_stopwords: set[str],
    english_stopwords: set[str],
    max_rows: int | None,
) -> tuple[Counter, Counter]:
    """Count Korean and English words from real-news rows using POS and length filters."""
    kiwi = Kiwi()
    korean_counts: Counter = Counter()
    english_counts: Counter = Counter()

    for chunk in pd.read_csv(
        csv_path,
        usecols=[LABEL_COL, TEXT_COL],
        chunksize=chunksize,
        nrows=max_rows,
        low_memory=False,
    ):
        chunk[LABEL_COL] = pd.to_numeric(chunk[LABEL_COL], errors="coerce")

        # Only real-news rows are used because this visualization focuses on real-news vocabulary.
        real_messages = chunk.loc[chunk[LABEL_COL] == real_label, TEXT_COL].dropna().astype(str)

        english_candidates: list[str] = []
        for message in real_messages:
            # Korean words are analyzed with Kiwi so only meaningful POS categories remain.
            for token in kiwi.tokenize(message):
                if token.tag not in KOREAN_POS_TAGS:
                    continue

                word = normalize_korean_token(token.form, token.tag)
                if len(word) < min_korean_length:
                    continue
                if word in korean_stopwords:
                    continue
                if KOREAN_RE.fullmatch(word):
                    korean_counts[word] += 1

            # English words are collected first, then POS-tagged in batches for speed.
            english_candidates.extend(
                collect_english_candidates(
                    message=message,
                    min_length=min_english_length,
                    stopwords=english_stopwords,
                )
            )

        update_english_counts(english_counts, english_candidates)

    return korean_counts, english_counts


def draw_wordcloud(
    ax: plt.Axes,
    counts: Counter,
    title: str,
    font_path: str | None,
    top_words: int,
    colormap: str,
) -> None:
    """Draw one WordCloud panel from a word-frequency Counter."""
    if not counts:
        ax.text(0.5, 0.5, "No words found", ha="center", va="center", fontsize=14)
        ax.set_title(title, pad=12)
        ax.axis("off")
        return

    wordcloud = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color="white",
        max_words=top_words,
        prefer_horizontal=0.9,
        colormap=colormap,
        random_state=42,
    ).generate_from_frequencies(dict(counts.most_common(top_words)))

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(title, pad=12)
    ax.axis("off")


def draw_barplot(
    ax: plt.Axes,
    counts: Counter,
    title: str,
    top_words: int,
    palette: str,
) -> None:
    """Draw one horizontal bar chart for the top frequent words."""
    if not counts:
        ax.text(0.5, 0.5, "No words found", ha="center", va="center", fontsize=14)
        ax.set_title(title, pad=12)
        ax.set_axis_off()
        return

    top_df = pd.DataFrame(
        counts.most_common(top_words),
        columns=["word", "count"],
    ).sort_values("count", ascending=True)

    sns.barplot(
        data=top_df,
        x="count",
        y="word",
        hue="word",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=4)


def plot_word_frequency(
    korean_counts: Counter,
    english_counts: Counter,
    font_path: str | None,
    output_path: Path,
    top_cloud_words: int,
    top_bar_words: int,
    show: bool,
) -> None:
    """Draw and save four charts: Korean cloud/bar and English cloud/bar."""
    configure_plot_style(font_path)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    draw_wordcloud(
        ax=axes[0][0],
        counts=korean_counts,
        title=f"Real News Korean Top {top_cloud_words} Word Cloud",
        font_path=font_path,
        top_words=top_cloud_words,
        colormap="tab10",
    )
    draw_barplot(
        ax=axes[0][1],
        counts=korean_counts,
        title=f"Real News Korean Top {top_bar_words} Word Frequency",
        top_words=top_bar_words,
        palette="Set2",
    )
    draw_wordcloud(
        ax=axes[1][0],
        counts=english_counts,
        title=f"Real News English Top {top_cloud_words} Word Cloud",
        font_path=None,
        top_words=top_cloud_words,
        colormap="Dark2",
    )
    draw_barplot(
        ax=axes[1][1],
        counts=english_counts,
        title=f"Real News English Top {top_bar_words} Word Frequency",
        top_words=top_bar_words,
        palette="Set3",
    )

    fig.suptitle("Real News Word Frequency by Language", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Define command-line options for data paths, filtering, and output settings."""
    parser = argparse.ArgumentParser(
        description="Visualize noun, verb, and adjective frequencies from real-news clean_message text."
    )
    parser.add_argument(
        "--csv",
        default="unified_news_refined.csv",
        help="Path to unified_news_refined.csv.",
    )
    parser.add_argument(
        "--output",
        default="data_visual4_real_word_frequency.png",
        help="Path where the graph image will be saved.",
    )
    parser.add_argument(
        "--real-label",
        type=int,
        default=0,
        help="Label value treated as real news. Default: 0.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=20_000,
        help="Number of rows to read at a time for large CSV files.",
    )
    parser.add_argument(
        "--min-korean-length",
        type=int,
        default=2,
        help="Minimum Korean word length. Default: 2.",
    )
    parser.add_argument(
        "--min-english-length",
        type=int,
        default=3,
        help="Minimum English word length. Default: 3.",
    )
    parser.add_argument(
        "--top-cloud-words",
        type=int,
        default=100,
        help="Number of high-frequency words to show in each word cloud.",
    )
    parser.add_argument(
        "--top-bar-words",
        type=int,
        default=10,
        help="Number of high-frequency words to show in each bar graph.",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Path to a Korean TrueType font file. Defaults to Windows Malgun Gothic if found.",
    )
    parser.add_argument(
        "--stopwords",
        default=None,
        help="Optional UTF-8 text file with one extra stopword per line.",
    )
    parser.add_argument(
        "--no-default-stopwords",
        action="store_true",
        help="Do not use the built-in Korean and English stopword lists.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick testing. Default: use all rows.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the graph window after saving the image.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full real-news word-frequency visualization pipeline."""
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ensure_nltk_tagger()

    font_path = find_korean_font(args.font)
    extra_stopwords = load_stopwords(args.stopwords)
    korean_stopwords = set()
    english_stopwords = set()

    if not args.no_default_stopwords:
        korean_stopwords.update(DEFAULT_KOREAN_STOPWORDS)
        english_stopwords.update(DEFAULT_ENGLISH_STOPWORDS)

    # User-provided stopwords are applied to both languages.
    korean_stopwords.update(extra_stopwords)
    english_stopwords.update(word.lower() for word in extra_stopwords)

    korean_counts, english_counts = count_pos_words(
        csv_path=csv_path,
        real_label=args.real_label,
        chunksize=args.chunksize,
        min_korean_length=args.min_korean_length,
        min_english_length=args.min_english_length,
        korean_stopwords=korean_stopwords,
        english_stopwords=english_stopwords,
        max_rows=args.max_rows,
    )

    print(f"Korean unique words: {len(korean_counts):,}")
    print(f"English unique words: {len(english_counts):,}")

    print("\nKorean top words")
    for word, count in korean_counts.most_common(args.top_bar_words):
        print(f"{word}\t{count:,}")

    print("\nEnglish top words")
    for word, count in english_counts.most_common(args.top_bar_words):
        print(f"{word}\t{count:,}")

    plot_word_frequency(
        korean_counts=korean_counts,
        english_counts=english_counts,
        font_path=font_path,
        output_path=output_path,
        top_cloud_words=args.top_cloud_words,
        top_bar_words=args.top_bar_words,
        show=args.show,
    )
    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
