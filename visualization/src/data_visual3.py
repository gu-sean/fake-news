import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


LABEL_COL = "label"
TEXT_COL = "clean_message"

# NNG/NNP: Korean nouns, VV: verbs, VA: adjectives.
# NNG/NNP: 한국어 명사, VV: 동사, VA: 형용사
KOREAN_POS_TAGS = {"NNG", "NNP", "VV", "VA"}
ENGLISH_POS_PREFIXES = ("NN", "VB", "JJ")

KOREAN_RE = re.compile(r"^[\uac00-\ud7a3]+$")
ENGLISH_RE = re.compile(r"^[A-Za-z]+$")

DEFAULT_FONT_PATHS = [
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
]

# Extra generic words are removed before TF-IDF so the result focuses on topic-bearing words.
# TF-IDF 계산 전에 불필요한 일반 단어를 제거하여 결과에서 주제와 관련된 단어에 집중합니다.
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
    """Check that NLTK's English POS tagger data is available."""
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        raise LookupError(
            "NLTK tagger data is missing. Run: "
            "python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\""
        )


def find_korean_font(custom_font_path: str | None = None) -> str | None:
    """Find a Korean font for WordCloud and matplotlib rendering."""
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
    """Apply chart style and font settings."""
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
    """Attach the dictionary ending to Korean verb/adjective stems."""
    if tag in {"VV", "VA"}:
        return f"{token}\ub2e4"
    return token


def extract_korean_terms(
    message: str,
    kiwi: Kiwi,
    min_length: int,
    stopwords: set[str],
) -> list[str]:
    """Extract Korean nouns, verbs, and adjectives from one message."""
    terms = []
    for token in kiwi.tokenize(message):
        if token.tag not in KOREAN_POS_TAGS:
            continue

        word = normalize_korean_token(token.form, token.tag)
        if len(word) < min_length:
            continue
        if word in stopwords:
            continue
        if KOREAN_RE.fullmatch(word):
            terms.append(word)

    return terms


def extract_english_terms(message: str, min_length: int, stopwords: set[str]) -> list[str]:
    """Extract English nouns, verbs, and adjectives from one message."""
    candidates = []
    for raw_word in message.split():
        word = raw_word.lower()
        if len(word) < min_length:
            continue
        if word in stopwords:
            continue
        if ENGLISH_RE.fullmatch(word):
            candidates.append(word)

    tagged_words = nltk.pos_tag(candidates) if candidates else []
    return [
        word
        for word, tag in tagged_words
        if tag.startswith(ENGLISH_POS_PREFIXES)
    ]


class TokenDocumentStream:
    """Stream fake-news documents as pre-tokenized strings for TF-IDF vectorization."""

    def __init__(
        self,
        csv_path: Path,
        language: str,
        fake_label: int,
        chunksize: int,
        min_korean_length: int,
        min_english_length: int,
        korean_stopwords: set[str],
        english_stopwords: set[str],
        max_rows: int | None,
    ) -> None:
        self.csv_path = csv_path
        self.language = language
        self.fake_label = fake_label
        self.chunksize = chunksize
        self.min_korean_length = min_korean_length
        self.min_english_length = min_english_length
        self.korean_stopwords = korean_stopwords
        self.english_stopwords = english_stopwords
        self.max_rows = max_rows

    def __iter__(self):
        kiwi = Kiwi() if self.language == "korean" else None

        for chunk in pd.read_csv(
            self.csv_path,
            usecols=[LABEL_COL, TEXT_COL],
            chunksize=self.chunksize,
            nrows=self.max_rows,
            low_memory=False,
        ):
            chunk[LABEL_COL] = pd.to_numeric(chunk[LABEL_COL], errors="coerce")
            fake_messages = chunk.loc[chunk[LABEL_COL] == self.fake_label, TEXT_COL]

            for message in fake_messages.dropna().astype(str):
                if self.language == "korean":
                    terms = extract_korean_terms(
                        message=message,
                        kiwi=kiwi,
                        min_length=self.min_korean_length,
                        stopwords=self.korean_stopwords,
                    )
                else:
                    terms = extract_english_terms(
                        message=message,
                        min_length=self.min_english_length,
                        stopwords=self.english_stopwords,
                    )

                yield " ".join(terms)


def calculate_tfidf_scores(
    documents,
    min_df: int,
    max_features: int | None,
) -> dict[str, float]:
    """Calculate average TF-IDF score per word across fake-news documents."""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    if tfidf_matrix.shape[1] == 0:
        return {}

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1

    return {
        word: float(score)
        for word, score in zip(feature_names, mean_scores)
        if score > 0
    }


def draw_wordcloud(
    ax: plt.Axes,
    scores: dict[str, float],
    title: str,
    font_path: str | None,
    top_words: int,
    colormap: str,
) -> None:
    """Draw a cleaner WordCloud with wider margins between words."""
    if not scores:
        ax.text(0.5, 0.5, "No words found", ha="center", va="center", fontsize=14)
        ax.set_title(title, pad=12)
        ax.axis("off")
        return

    top_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_words])
    wordcloud = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color="white",
        max_words=top_words,
        prefer_horizontal=1.0,
        relative_scaling=0.55,
        margin=12,
        collocations=False,
        repeat=False,
        min_font_size=14,
        colormap=colormap,
        random_state=42,
    ).generate_from_frequencies(top_scores)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(title, pad=12)
    ax.axis("off")


def draw_barplot(
    ax: plt.Axes,
    scores: dict[str, float],
    title: str,
    top_words: int,
    palette: str,
) -> None:
    """Draw the top TF-IDF terms as a horizontal bar chart."""
    if not scores:
        ax.text(0.5, 0.5, "No words found", ha="center", va="center", fontsize=14)
        ax.set_title(title, pad=12)
        ax.set_axis_off()
        return

    top_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_words]
    top_df = pd.DataFrame(top_items, columns=["word", "tfidf"]).sort_values("tfidf")

    sns.barplot(
        data=top_df,
        x="tfidf",
        y="word",
        hue="word",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Average TF-IDF")
    ax.set_ylabel("")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=4)


def plot_tfidf_scores(
    korean_scores: dict[str, float],
    english_scores: dict[str, float],
    font_path: str | None,
    output_path: Path,
    top_cloud_words: int,
    top_bar_words: int,
    show: bool,
) -> None:
    """Draw and save Korean/English TF-IDF WordCloud and bar charts."""
    configure_plot_style(font_path)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    draw_wordcloud(
        ax=axes[0][0],
        scores=korean_scores,
        title=f"Korean Top {top_cloud_words} TF-IDF Word Cloud",
        font_path=font_path,
        top_words=top_cloud_words,
        colormap="tab10",
    )
    draw_barplot(
        ax=axes[0][1],
        scores=korean_scores,
        title=f"Korean Top {top_bar_words} TF-IDF Terms",
        top_words=top_bar_words,
        palette="Set2",
    )
    draw_wordcloud(
        ax=axes[1][0],
        scores=english_scores,
        title=f"English Top {top_cloud_words} TF-IDF Word Cloud",
        font_path=None,
        top_words=top_cloud_words,
        colormap="Dark2",
    )
    draw_barplot(
        ax=axes[1][1],
        scores=english_scores,
        title=f"English Top {top_bar_words} TF-IDF Terms",
        top_words=top_bar_words,
        palette="Set3",
    )

    fig.suptitle("Fake News Important Terms by TF-IDF", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Define command-line options for TF-IDF extraction and visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize important fake-news terms with TF-IDF."
    )
    parser.add_argument("--csv", default="unified_news_refined.csv")
    parser.add_argument("--output", default="data_visual3_tfidf.png")
    parser.add_argument("--fake-label", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--min-korean-length", type=int, default=2)
    parser.add_argument("--min-english-length", type=int, default=3)
    parser.add_argument("--top-cloud-words", type=int, default=50)
    parser.add_argument("--top-bar-words", type=int, default=15)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--font", default=None)
    parser.add_argument("--stopwords", default=None)
    parser.add_argument("--no-default-stopwords", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the full fake-news TF-IDF visualization pipeline."""
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

    korean_stopwords.update(extra_stopwords)
    english_stopwords.update(word.lower() for word in extra_stopwords)

    common_stream_args = {
        "csv_path": csv_path,
        "fake_label": args.fake_label,
        "chunksize": args.chunksize,
        "min_korean_length": args.min_korean_length,
        "min_english_length": args.min_english_length,
        "korean_stopwords": korean_stopwords,
        "english_stopwords": english_stopwords,
        "max_rows": args.max_rows,
    }

    korean_docs = TokenDocumentStream(language="korean", **common_stream_args)
    english_docs = TokenDocumentStream(language="english", **common_stream_args)

    print("Calculating Korean TF-IDF scores...")
    korean_scores = calculate_tfidf_scores(
        documents=korean_docs,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    print("Calculating English TF-IDF scores...")
    english_scores = calculate_tfidf_scores(
        documents=english_docs,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    print(f"Korean TF-IDF terms: {len(korean_scores):,}")
    print(f"English TF-IDF terms: {len(english_scores):,}")

    print("\nKorean top TF-IDF terms")
    for word, score in sorted(korean_scores.items(), key=lambda item: item[1], reverse=True)[
        : args.top_bar_words
    ]:
        print(f"{word}\t{score:.6f}")

    print("\nEnglish top TF-IDF terms")
    for word, score in sorted(english_scores.items(), key=lambda item: item[1], reverse=True)[
        : args.top_bar_words
    ]:
        print(f"{word}\t{score:.6f}")

    plot_tfidf_scores(
        korean_scores=korean_scores,
        english_scores=english_scores,
        font_path=font_path,
        output_path=output_path,
        top_cloud_words=args.top_cloud_words,
        top_bar_words=args.top_bar_words,
        show=args.show,
    )
    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
