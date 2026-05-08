import re
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 설정 ───────────────────────────────────────────────────────────────────────
DVL_COLS = ["stat_distortion", "causal_error", "emotional_provocation", "source_lack", "img_mismatch"]

MIN_CONTENT_LEN = 150    
MAX_CONTENT_LEN = 10_000  
SPECIAL_CHAR_RATIO_MAX = 0.35
TITLE_SIM_THRESHOLD = 0.90
FOREIGN_CHAR_RATIO_MAX = 0.10
RANDOM_SEED = 42

BOILERPLATE_PATTERNS = [
    r"무단\s*전재.{0,20}금지",
    r"저작권.{0,20}무단.{0,20}금지",
    r"copyright\s*©?\s*\d{4}.{0,30}all rights reserved",
    r"all rights reserved\.?\s*$",
    r"\(c\)\s*\d{4}\s+\w[\w\s]+\.",
]

_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)
_STANDARD_CHAR_RE = re.compile(r'[가-힣a-zA-Z0-9\s.,!?;:\'"()\[\]{}\-/\\@#%&*+=<>|~`^_]')

# ── 필터 함수 ──────────────────────────────────────────────────────────────────

def _special_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    special = sum(1 for ch in text if not _STANDARD_CHAR_RE.match(ch))
    return special / len(text)


def _foreign_char_ratio(text: str) -> float:
    """중국어·일본어 문자 비중 (한국어 제외)"""
    if not text:
        return 0.0
    foreign = sum(
        1 for ch in text
        if ('一' <= ch <= '鿿')  
        or ('぀' <= ch <= 'ゟ')   
        or ('゠' <= ch <= 'ヿ')  
    )
    return foreign / len(text)


def _is_boilerplate_dominant(content: str) -> bool:
    """보일러플레이트 제거 후 50자 미만이면 실질 내용 없음으로 판단"""
    stripped = _BOILERPLATE_RE.sub("", content).strip()
    return len(stripped) < 50


def _title_ngrams(text: str, n: int = 3) -> set:
    norm = re.sub(r'[^가-힣a-z0-9]', '', text.lower())
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}


def _jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ── STEP별 필터 ────────────────────────────────────────────────────────────────

def step1_length_filter(df: pd.DataFrame) -> pd.DataFrame:
    """하한선: 150자 미만 제거 / 상한선: 10,000자 초과"""
    before = len(df)
    df = df[df["content"].str.len() >= MIN_CONTENT_LEN].copy()
    removed = before - len(df)

    long_mask = df["content"].str.len() > MAX_CONTENT_LEN
    truncated = long_mask.sum()
    df.loc[long_mask, "content"] = df.loc[long_mask, "content"].str[:MAX_CONTENT_LEN]
    df.loc[long_mask, "clean_message"] = df.loc[long_mask, "clean_message"].str[:MAX_CONTENT_LEN]

    logger.info(
        f"[STEP 1] 길이 필터: 하한({MIN_CONTENT_LEN}자) 제거 {removed}행 | "
        f"상한({MAX_CONTENT_LEN}자) truncate {truncated}행 → 남은 행 {len(df)}"
    )
    return df.reset_index(drop=True)


def step2_special_char_filter(df: pd.DataFrame) -> pd.DataFrame:
    """특수 문자 비중 초과 제거 (주식 시황표·도표형 기사)"""
    before = len(df)
    ratios = df["content"].apply(_special_char_ratio)
    df = df[ratios <= SPECIAL_CHAR_RATIO_MAX].reset_index(drop=True)
    logger.info(f"[STEP 2] 특수 문자 과다 제거: {before} → {len(df)} ({before - len(df)}행 제거)")
    return df


def step3_boilerplate_filter(df: pd.DataFrame) -> pd.DataFrame:
    """보일러플레이트 위주 기사 제거"""
    before = len(df)
    mask = df["content"].apply(_is_boilerplate_dominant)
    df = df[~mask].reset_index(drop=True)
    logger.info(f"[STEP 3] 보일러플레이트 위주 제거: {before} → {len(df)} ({before - len(df)}행 제거)")
    return df


def step4_foreign_lang_filter(df: pd.DataFrame) -> pd.DataFrame:
    """중국어·일본어 혼입 비율 초과 제거"""
    before = len(df)
    ratios = df["content"].apply(_foreign_char_ratio)
    df = df[ratios <= FOREIGN_CHAR_RATIO_MAX].reset_index(drop=True)
    logger.info(f"[STEP 4] 외국어(중·일) 혼입 제거: {before} → {len(df)} ({before - len(df)}행 제거)")
    return df


def step5_title_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    제목 유사도 중복 제거.
    - 정규화된 제목으로 정렬 후 인접 쌍 비교 
    - 같은 label끼리만 비교해 서로 다른 관점의 기사 보존
    """
    before = len(df)
    df = df.copy()
    df["_norm"] = df["title"].str.lower().str.strip()
    df["_ngrams"] = df["_norm"].apply(_title_ngrams)
    df_sorted = df.sort_values(["label", "_norm"]).reset_index(drop=True)

    keep = [True] * len(df_sorted)
    labels: list[int] = df_sorted["label"].tolist()
    ngrams_list: list[set] = df_sorted["_ngrams"].tolist()
    prev_ngrams: set = set()
    prev_label: int = -1

    for i, (curr_label, curr_ngrams) in enumerate(
        tqdm(zip(labels, ngrams_list), total=len(df_sorted), desc="유사 제목 중복 제거", miniters=10000)
    ):
        if curr_label == prev_label and prev_ngrams and _jaccard(prev_ngrams, curr_ngrams) >= TITLE_SIM_THRESHOLD:
            keep[i] = False
        else:
            prev_ngrams = curr_ngrams
            prev_label = curr_label

    result = df_sorted[keep].drop(columns=["_norm", "_ngrams"]).reset_index(drop=True)
    logger.info(f"[STEP 5] 유사 제목 중복 제거: {before} → {len(result)} ({before - len(result)}행 제거)")
    return result


def step6_dvl_fake_filter(df: pd.DataFrame) -> pd.DataFrame:
    """DVL 패턴이 단 하나도 없는 가짜 뉴스 제거"""
    before = len(df)
    dvl_score = df[DVL_COLS].sum(axis=1)
    no_dvl_fake = (df["label"] == 1) & (dvl_score == 0)
    df = df[~no_dvl_fake].reset_index(drop=True)
    real = (df["label"] == 0).sum()
    fake = (df["label"] == 1).sum()
    logger.info(
        f"[STEP 6] DVL 패턴 없는 가짜뉴스 제거: {before} → {len(df)} ({before - len(df)}행 제거)\n"
        f"          현재 분포: label=0(진짜)={real:,}  label=1(가짜)={fake:,}"
    )
    return df


def step7_undersample(df: pd.DataFrame) -> pd.DataFrame:
    """클래스 불균형 해소 (1:1)
    - 가짜 뉴스가 많을 때: DVL 점수 높은 것 우선 보존
    - 진짜 뉴스가 많을 때: 무작위 샘플링
    """
    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    target = min(real_count, fake_count)

    df_real = df[df["label"] == 0].copy()
    df_fake = df[df["label"] == 1].copy()

    if fake_count > target:
        df_fake["_dvl_score"] = df_fake[DVL_COLS].sum(axis=1)
        df_fake = (
            df_fake.sort_values("_dvl_score", ascending=False)
            .head(target)
            .drop(columns=["_dvl_score"])
        )
    elif real_count > target:
        df_real = df_real.sample(n=target, random_state=RANDOM_SEED)

    df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info(
        f"[STEP 7] 클래스 균형(1:1): label=0={target:,}  label=1={target:,}  총={len(df):,}"
    )
    return df


# ── 메인 ───────────────────────────────────────────────────────────────────────

def refine(input_path: str, output_path: str) -> None:
    inp = Path(input_path)
    out = Path(output_path)

    logger.info(f"=== 추가 전처리 시작: {inp} ===")
    df = pd.read_csv(inp, low_memory=False)
    logger.info(
        f"원본: {len(df):,}행  "
        f"(label=0: {(df.label==0).sum():,}  label=1: {(df.label==1).sum():,})"
    )

    df = step1_length_filter(df)
    df = step2_special_char_filter(df)
    df = step3_boilerplate_filter(df)
    df = step4_foreign_lang_filter(df)
    df = step5_title_dedup(df)
    df = step6_dvl_fake_filter(df)
    df = step7_undersample(df)

    df["id"] = range(1, len(df) + 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"=== 완료: {out}  ({len(df):,}행) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unified_news_processed.csv 추가 정제")
    parser.add_argument("--input",  default="data/processed/unified_news_processed.csv")
    parser.add_argument("--output", default="data/processed/unified_news_refined.csv")
    args = parser.parse_args()
    refine(args.input, args.output)
