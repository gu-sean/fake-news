import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LABEL_COL = "label"

# Fake-news problem columns. Each column is expected to contain 0 or 1.
# 가짜 뉴스 문제 유형 컬럼. 이 컬럼은 0 또는 1의 값을 가집니다.
ISSUE_COLS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

LABEL_NAMES = {
    0: "Real News",
    1: "Fake News",
}

ISSUE_NAMES = {
    "stat_distortion": "Statistical Distortion",
    "causal_error": "Causal Error",
    "emotional_provocation": "Emotional Provocation",
    "source_lack": "Lack of Sources",
    "img_mismatch": "Image Mismatch",
}


def configure_plot_style() -> None:
    """Apply common seaborn/matplotlib style settings for readable charts."""
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Windows Korean font fallback. English labels are used so plots remain readable elsewhere.
    # Windows에서는 한국어 글꼴이 기본 글꼴로 사용됩니다. 그래프를 다른 곳에서도 읽을 수 있도록 영어 레이블이 사용됩니다.
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def count_distributions(csv_path: Path, fake_label: int, chunksize: int) -> tuple[pd.Series, pd.Series]:
    """Count news labels and fake-news issue flags without loading the whole CSV at once."""
    label_counts = pd.Series(dtype="int64")
    issue_counts = pd.Series(0, index=ISSUE_COLS, dtype="int64")

    # Read only the columns needed for this visualization to reduce memory usage.
    # 메모리 사용량을 줄이기 위해 이 시각화에 필요한 열만 사용해 읽겠습니다.
    for chunk in pd.read_csv(
        csv_path,
        usecols=[LABEL_COL, *ISSUE_COLS],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk[LABEL_COL] = pd.to_numeric(chunk[LABEL_COL], errors="coerce")
        label_counts = label_counts.add(chunk[LABEL_COL].value_counts(dropna=False), fill_value=0)

        # Issue-type distribution is calculated only among rows marked as fake news.
        # 가짜 뉴스인 행에서만 가짜뉴스 문제 유형 분포를 계산합니다.
        fake_news = chunk[chunk[LABEL_COL] == fake_label]
        for issue_col in ISSUE_COLS:
            issue_counts[issue_col] += pd.to_numeric(
                fake_news[issue_col],
                errors="coerce",
            ).fillna(0).astype(int).sum()

    return label_counts.astype(int), issue_counts.astype(int)


def build_label_dataframe(label_counts: pd.Series) -> pd.DataFrame:
    """Convert raw label counts into a plotting-friendly DataFrame."""
    rows = []
    for label_value, count in label_counts.sort_index().items():
        if pd.isna(label_value):
            label_name = "Missing Label"
        else:
            label_value = int(label_value)
            label_name = LABEL_NAMES.get(label_value, f"Label {label_value}")

        rows.append({"label": label_name, "count": int(count)})

    return pd.DataFrame(rows)


def build_issue_dataframe(issue_counts: pd.Series) -> pd.DataFrame:
    """Convert issue counts into a plotting-friendly DataFrame with readable names."""
    return pd.DataFrame(
        {
            "issue": [ISSUE_NAMES[col] for col in ISSUE_COLS],
            "count": [int(issue_counts[col]) for col in ISSUE_COLS],
        }
    )


def plot_distributions(label_df: pd.DataFrame, issue_df: pd.DataFrame, output_path: Path) -> None:
    """Draw and save two bar charts: label distribution and fake-news issue distribution."""
    configure_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Chart 1: compare real news and fake news counts.
    # 차트 1: 진짜 뉴스와 가짜 뉴스 카운트
    sns.barplot(
        data=label_df,
        x="label",
        y="count",
        hue="label",
        palette=["#2f6f8f", "#c94f4f"],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Real vs Fake News Distribution", pad=14)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.0f", padding=4)

    # Chart 2: compare problem-type counts within fake-news rows.
    # 차트 2: 가짜 뉴스 문제 유형 수를 비교
    sns.barplot(
        data=issue_df,
        x="issue",
        y="count",
        hue="issue",
        palette="Set2",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Problem Type Distribution in Fake News", pad=14)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=25)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def parse_args() -> argparse.Namespace:
    """Define command-line options so paths, labels, and chunk size can be changed easily."""
    parser = argparse.ArgumentParser(
        description="Visualize news label distribution and fake-news issue distribution."
    )
    parser.add_argument(
        "--csv",
        default="unified_news_refined.csv",
        help="Path to unified_news_refined.csv.",
    )
    parser.add_argument(
        "--output",
        default="data_visual1_class_distribution.png",
        help="Path where the graph image will be saved.",
    )
    parser.add_argument(
        "--fake-label",
        type=int,
        default=1,
        help="Label value treated as fake news. Default: 1.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of rows to read at a time for large CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full visualization pipeline."""
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    label_counts, issue_counts = count_distributions(
        csv_path=csv_path,
        fake_label=args.fake_label,
        chunksize=args.chunksize,
    )
    label_df = build_label_dataframe(label_counts)
    issue_df = build_issue_dataframe(issue_counts)

    print("Label distribution")
    print(label_df.to_string(index=False))
    print("\nFake-news issue distribution")
    print(issue_df.to_string(index=False))

    plot_distributions(label_df, issue_df, output_path)
    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
