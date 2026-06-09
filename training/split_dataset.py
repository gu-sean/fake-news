# 전처리 완료된 CSV를 30,000건 단위 subset으로 분할


import argparse
from pathlib import Path
import pandas as pd

SUBSET_SIZE = 30_000
RANDOM_SEED = 42


def split(input_path: str, output_dir: str) -> None:
    df = pd.read_csv(input_path, low_memory=False)

    real = df[df["label"] == 0].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    fake = df[df["label"] == 1].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    half = SUBSET_SIZE // 2  # subset당 각 클래스 15,000건
    per_class = min(len(real), len(fake))
    n_full    = per_class // half
    remainder = per_class % half

    # 나머지가 half보다 작으면 마지막 subset과 합쳐서 파일 수를 줄임
    if 0 < remainder < half:
        n_full -= 1

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_full):
        subset = pd.concat([
            real.iloc[i * half:(i + 1) * half],
            fake.iloc[i * half:(i + 1) * half],
        ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        subset["id"] = range(1, len(subset) + 1)

        out_path = out_dir / f"subset_{i + 1:02d}.csv"
        subset.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  subset_{i + 1:02d}.csv  {len(subset):,}건  (진짜 {(subset.label==0).sum():,} / 가짜 {(subset.label==1).sum():,})")

    # 나머지 행으로 마지막 subset 생성
    real_tail = real.iloc[n_full * half:]
    fake_tail = fake.iloc[n_full * half:]
    tail_size  = min(len(real_tail), len(fake_tail))

    if tail_size > 0:
        last = pd.concat([
            real_tail.iloc[:tail_size],
            fake_tail.iloc[:tail_size],
        ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        last["id"] = range(1, len(last) + 1)

        idx = n_full + 1
        out_path = out_dir / f"subset_{idx:02d}.csv"
        last.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  subset_{idx:02d}.csv  {len(last):,}건  (진짜 {(last.label==0).sum():,} / 가짜 {(last.label==1).sum():,})")

    print(f"\n완료: {out_dir}  총 {idx}개 파일")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋 subset 분할")
    parser.add_argument("--input",      default="data/processed/unified_news_refined.csv")
    parser.add_argument("--output_dir", default="data/processed/subsets")
    args = parser.parse_args()

    split(args.input, args.output_dir)
