# 전처리 통합 스크립트 

import csv
import hashlib
import json
import logging
import re
import argparse
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────
DVL_KEYWORD_SEEDS = {
    "stat_distortion":       ["100%", "모든", "항상", "절대", "never", "always"],
    "causal_error":          ["때문에", "원인", "증명", "결과적으로", "causes", "proves"],
    "emotional_provocation": ["충격", "경악", "분노", "shocking", "outrage", "unbelievable"],
    "source_lack":           ["관계자", "소식통", "익명", "sources say", "reportedly"],
    "img_mismatch":          ["사진 속", "이 사진은", "photo shows", "pictured here"],
}
DVL_COLS = list(DVL_KEYWORD_SEEDS.keys())

MIN_CONTENT_LEN        = 150
MAX_CONTENT_LEN        = 10_000
SPECIAL_CHAR_RATIO_MAX = 0.35
TITLE_SIM_THRESHOLD    = 0.90
FOREIGN_CHAR_RATIO_MAX = 0.10
RANDOM_SEED            = 42

BOILERPLATE_PATTERNS = [
    r"무단\s*전재.{0,20}금지",
    r"저작권.{0,20}무단.{0,20}금지",
    r"copyright\s*©?\s*\d{4}.{0,30}all rights reserved",
    r"all rights reserved\.?\s*$",
    r"\(c\)\s*\d{4}\s+\w[\w\s]+\.",
]
_BOILERPLATE_RE   = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)
_STANDARD_CHAR_RE = re.compile(r'[가-힣a-zA-Z0-9\s.,!?;:\'"()\[\]{}\-/\\@#%&*+=<>|~`^_]')


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def _strip(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip()

def _parse_date(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    for fmt in ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.search(r"(\d{4})", str(raw))
    return f"{m.group(1)}-01-01" if m else ""

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ════════════════════════════════════════════════════════════════════════════
# 1단계: 원본 파일 통합
# ════════════════════════════════════════════════════════════════════════════
def _add_row(rows: list, seen: set, title, content, media, date, label) -> None:
    t, c = _strip(title), _strip(content)
    if not t and not c:
        return
    if len(c) < 20:
        return
    h = hashlib.md5((t + c).lower().encode()).hexdigest()
    if h in seen:
        return
    seen.add(h)
    rows.append({
        "title": t, "content": c,
        "media": _strip(media) or "Unknown",
        "date": _parse_date(date),
        "label": label,
    })


def _load_english_sources(raw_dir: Path, rows: list, seen: set) -> None:
    """영어 CSV 3종 로드: Fake.csv / True.csv / Fake_Real_News_Data.csv"""
    for fname, lbl in [("Fake.csv", 1), ("True.csv", 0)]:
        p = raw_dir / fname
        if not p.exists():
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                _add_row(rows, seen,
                         row.get("title", ""), row.get("text", ""),
                         row.get("subject", ""), row.get("date", ""), lbl)
        logger.info(f"  [영어] {fname} 로드 완료")

    p_extra = raw_dir / "Fake_Real_News_Data.csv"
    if p_extra.exists():
        with open(p_extra, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                lbl = 1 if "FAKE" in row.get("label", "").upper() else 0
                _add_row(rows, seen,
                         row.get("title", ""), row.get("text", ""),
                         "Unknown", row.get("date", ""), lbl)
        logger.info("  [영어] Fake_Real_News_Data.csv 로드 완료")


def _load_korean_zip(zip_path: Path, rows: list, seen: set) -> int:
    """
    한국어 zip 내부 JSON 파싱.
    JSON 구조: sourceDataInfo.newsTitle/newsContent/newsCategory
               labeledDataInfo.clickbaitClass (0=클릭베이트/가짜→label=1, 1=정상→label=0)
    """
    count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for jf in (n for n in zf.namelist() if n.endswith(".json")):
                try:
                    data = json.loads(zf.read(jf).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                src      = data.get("sourceDataInfo", {})
                lbl_info = data.get("labeledDataInfo", {})

                title   = _strip(src.get("newsTitle", ""))
                content = _strip(src.get("newsContent", ""))
                media   = _strip(src.get("newsCategory", "")) or "Unknown"
                # clickbaitClass=0 → 클릭베이트(가짜) → label=1
                # clickbaitClass=1 → 정상기사(진짜) → label=0
                label   = 1 - int(lbl_info.get("clickbaitClass", 1))

                before = len(rows)
                _add_row(rows, seen, title, content, media, "", label)
                if len(rows) > before:
                    count += 1
    except zipfile.BadZipFile:
        logger.warning(f"  손상된 zip 건너뜀: {zip_path.name}")
    return count


def load_all_sources(raw_dir: Path) -> pd.DataFrame:
    rows: list = []
    seen: set  = set()

    logger.info("── 영어 데이터셋 로드 ──────────────────────────────────")
    _load_english_sources(raw_dir, rows, seen)

    logger.info("── 한국어 zip 데이터셋 로드 ────────────────────────────")
    for subdir in ["Training/02.라벨링데이터", "Validation/02.라벨링데이터"]:
        zip_dir = raw_dir / subdir
        if not zip_dir.exists():
            logger.warning(f"  폴더 없음 (건너뜀): {zip_dir}")
            continue
        zips  = sorted(zip_dir.glob("*.zip"))
        total = 0
        for zp in tqdm(zips, desc=f"  {subdir.split('/')[0]}"):
            total += _load_korean_zip(zp, rows, seen)
        logger.info(f"  {subdir}: {total:,}건 로드")

    df = pd.DataFrame(rows)
    logger.info(f"전체 원본 통합: {len(df):,}행")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2단계: DVL 특징 추출 + clean_message 생성
# ════════════════════════════════════════════════════════════════════════════
def extract_features(df: pd.DataFrame, min_content_len: int = 10) -> pd.DataFrame:
    logger.info("── DVL 특징 추출 + clean_message 생성 ─────────────────")

    df["content"] = df["content"].fillna("").astype(str)
    df["title"]   = df["title"].fillna("").astype(str)

    # content가 너무 짧으면 title로 대체
    mask = df["content"].str.len() < min_content_len
    df.loc[mask, "content"] = df.loc[mask, "title"]
    df = df[df["content"].str.len() >= min_content_len].reset_index(drop=True)

    for col in ["media", "date", "label"]:
        if df[col].isnull().any():
            fill_val = df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else "unknown"
            df[col]  = df[col].fillna(fill_val)

    df["clean_message"] = (df["title"] + " " + df["content"]).apply(clean_text)

    for pattern, keywords in tqdm(DVL_KEYWORD_SEEDS.items(), desc="DVL 패턴 추출"):
        escaped   = [re.escape(kw) for kw in keywords]
        df[pattern] = df["clean_message"].str.contains(
            "|".join(escaped), case=False, na=False
        ).astype(int)

    return df


# ════════════════════════════════════════════════════════════════════════════
# 3단계: 품질 필터링 (7단계)
# ════════════════════════════════════════════════════════════════════════════
def _special_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    special = sum(1 for ch in text if not _STANDARD_CHAR_RE.match(ch))
    return special / len(text)

def _foreign_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    foreign = sum(
        1 for ch in text
        if ('一' <= ch <= '鿿') or ('぀' <= ch <= 'ゟ') or ('゠' <= ch <= 'ヿ')
    )
    return foreign / len(text)

def _is_boilerplate_dominant(content: str) -> bool:
    return len(_BOILERPLATE_RE.sub("", content).strip()) < 50

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


def _step_length_filter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["content"].str.len() >= MIN_CONTENT_LEN].copy()
    long_mask = df["content"].str.len() > MAX_CONTENT_LEN
    df.loc[long_mask, "content"]       = df.loc[long_mask, "content"].str[:MAX_CONTENT_LEN]
    df.loc[long_mask, "clean_message"] = df.loc[long_mask, "clean_message"].str[:MAX_CONTENT_LEN]
    logger.info(f"  [1] 길이 필터: 제거 {before - len(df):,}행 | truncate {long_mask.sum():,}행 → 잔존 {len(df):,}행")
    return df.reset_index(drop=True)

def _step_special_char_filter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["content"].apply(_special_char_ratio) <= SPECIAL_CHAR_RATIO_MAX].reset_index(drop=True)
    logger.info(f"  [2] 특수문자 필터: {before:,} → {len(df):,} ({before - len(df):,}행 제거)")
    return df

def _step_boilerplate_filter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df["content"].apply(_is_boilerplate_dominant)].reset_index(drop=True)
    logger.info(f"  [3] 보일러플레이트 필터: {before:,} → {len(df):,} ({before - len(df):,}행 제거)")
    return df

def _step_foreign_lang_filter(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["content"].apply(_foreign_char_ratio) <= FOREIGN_CHAR_RATIO_MAX].reset_index(drop=True)
    logger.info(f"  [4] 외국어 필터: {before:,} → {len(df):,} ({before - len(df):,}행 제거)")
    return df

def _step_title_dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["_norm"]   = df["title"].str.lower().str.strip()
    df["_ngrams"] = df["_norm"].apply(_title_ngrams)
    df_sorted     = df.sort_values(["label", "_norm"]).reset_index(drop=True)

    keep = [True] * len(df_sorted)
    prev_ngrams, prev_label = set(), -1
    for i, (curr_label, curr_ngrams) in enumerate(
        tqdm(zip(df_sorted["label"].tolist(), df_sorted["_ngrams"].tolist()),
             total=len(df_sorted), desc="  [5] 제목 중복 제거", miniters=10000)
    ):
        if curr_label == prev_label and prev_ngrams and _jaccard(prev_ngrams, curr_ngrams) >= TITLE_SIM_THRESHOLD:
            keep[i] = False
        else:
            prev_ngrams = curr_ngrams
            prev_label  = curr_label

    result = df_sorted[keep].drop(columns=["_norm", "_ngrams"]).reset_index(drop=True)
    logger.info(f"  [5] 제목 중복 제거: {before:,} → {len(result):,} ({before - len(result):,}행 제거)")
    return result

def _step_dvl_fake_filter(df: pd.DataFrame) -> pd.DataFrame:
    before      = len(df)
    no_dvl_fake = (df["label"] == 1) & (df[DVL_COLS].sum(axis=1) == 0)
    df = df[~no_dvl_fake].reset_index(drop=True)
    logger.info(
        f"  [6] DVL 없는 가짜뉴스 제거: {before:,} → {len(df):,} ({before - len(df):,}행 제거)\n"
        f"      분포: label=0(진짜)={(df.label==0).sum():,}  label=1(가짜)={(df.label==1).sum():,}"
    )
    return df

def _step_undersample(df: pd.DataFrame) -> pd.DataFrame:
    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    target     = min(real_count, fake_count)
    df_real    = df[df["label"] == 0].copy()
    df_fake    = df[df["label"] == 1].copy()
    if fake_count > target:
        df_fake["_dvl_score"] = df_fake[DVL_COLS].sum(axis=1)
        df_fake = df_fake.sort_values("_dvl_score", ascending=False).head(target).drop(columns=["_dvl_score"])
    elif real_count > target:
        df_real = df_real.sample(n=target, random_state=RANDOM_SEED)
    df = pd.concat([df_real, df_fake]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info(f"  [7] 클래스 균형(1:1): label=0={target:,}  label=1={target:,}  총={len(df):,}")
    return df


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("── 품질 필터링 (7단계) ─────────────────────────────────")
    df = _step_length_filter(df)
    df = _step_special_char_filter(df)
    df = _step_boilerplate_filter(df)
    df = _step_foreign_lang_filter(df)
    df = _step_title_dedup(df)
    df = _step_dvl_fake_filter(df)
    df = _step_undersample(df)
    return df


# ════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════════════════════════
def run(raw_dir: str, output_path: str, min_content_len: int = 10) -> None:
    raw_path = Path(raw_dir)
    out_path = Path(output_path)

    logger.info(f"{'='*60}")
    logger.info(f"전처리 시작: {raw_path} → {out_path}")
    logger.info(f"{'='*60}")

    df = load_all_sources(raw_path)
    if df.empty:
        logger.error("로드된 데이터가 없습니다. 경로를 확인하세요.")
        return

    df.insert(0, "id", range(1, len(df) + 1))
    logger.info(f"원본 통합 완료: {len(df):,}행  (label=0: {(df.label==0).sum():,}  label=1: {(df.label==1).sum():,})")

    df = extract_features(df, min_content_len)
    df = apply_quality_filters(df)

    df["id"] = range(1, len(df) + 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info(f"{'='*60}")
    logger.info(f"완료: {out_path}  (최종 {len(df):,}행)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전처리 통합 스크립트 (1차+2차)")
    parser.add_argument("--raw_dir",         default="data/raw",
                        help="원본 데이터 폴더 경로")
    parser.add_argument("--output",          default="data/processed/unified_news_refined.csv",
                        help="최종 출력 CSV 경로")
    parser.add_argument("--min_content_len", type=int, default=10,
                        help="content 최소 길이 (미만 시 title로 대체)")
    args = parser.parse_args()
    run(args.raw_dir, args.output, args.min_content_len)
