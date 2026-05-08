import csv
import hashlib
import logging
import re
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── 로거 설정 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 표준 설정 ──────────────────────────────────────────────────────────────────
STANDARD_COLS = ["id", "title", "content", "media", "date", "label"]
DVL_KEYWORD_SEEDS = {
    "stat_distortion": ["100%", "모든", "항상", "절대", "never", "always"],
    "causal_error": ["때문에", "원인", "증명", "결과적으로", "causes", "proves"],
    "emotional_provocation": ["충격", "경악", "분노", "shocking", "outrage", "unbelievable"],
    "source_lack": ["관계자", "소식통", "익명", "sources say", "reportedly"],
    "img_mismatch": ["사진 속", "이 사진은", "photo shows", "pictured here"],
}

def _parse_date(raw: str) -> str:
    if not raw or not isinstance(raw, str): return ""
    fmts = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]
    for fmt in fmts:
        try: return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError: continue
    m = re.search(r"(\d{4})", str(raw))
    return f"{m.group(1)}-01-01" if m else ""

def _strip(t) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip()

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _load_raw_sources(raw_dir: Path) -> list:
    raw_dir = Path(raw_dir)
    rows, seen = [], set()

    def add(title, content, media, date, label):
        t, c = _strip(title), _strip(content)
        if not t and not c: return
        if len(c) < 20: return
        h = hashlib.md5((t + c).lower().encode()).hexdigest()
        if h in seen: return
        seen.add(h)
        rows.append({"title": t, "content": c, "media": _strip(media) or "Unknown", "date": _parse_date(date), "label": label})

    # 파일 존재 확인 및 로드
    for fname, lbl in [("Fake.csv", 1), ("True.csv", 0)]:
        p = raw_dir / fname
        if p.exists():
            with open(p, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    add(row.get("title", ""), row.get("text", ""), row.get("subject", ""), row.get("date", ""), lbl)

    p_extra = raw_dir / "Fake_Real_News_Data.csv"
    if p_extra.exists():
        with open(p_extra, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                lbl = 1 if "FAKE" in row.get("label", "").upper() else 0
                add(row.get("title", ""), row.get("text", ""), "Unknown", row.get("date", ""), lbl)

    return rows

def preprocess_logic(raw_dir: str, output_path: str, min_content_len: int = 10):
    raw_path = Path(raw_dir)     
    out_path = Path(output_path)

    logger.info(f"=== STEP 1: 파일 통합 ({raw_path}) ===")
    df = pd.DataFrame(_load_raw_sources(raw_path))
    if df.empty:
        logger.error("데이터를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    df.insert(0, "id", range(1, len(df) + 1))

    logger.info("=== STEP 2: DVL 전처리 시작 ===")
    df["content"] = df["content"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    # content가 짧으면 title로 대체, 둘 다 짧으면 해당 행 삭제
    mask = df["content"].str.len() < min_content_len
    df.loc[mask, "content"] = df.loc[mask, "title"]
    df = df[df["content"].str.len() >= min_content_len].reset_index(drop=True)

    # 나머지 결측치 → 컬럼별 최빈값
    for col in ["media", "date", "label"]:
        if df[col].isnull().any():
            fill_val = df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else "unknown"
            df[col] = df[col].fillna(fill_val)

    df["clean_message"] = (df["title"] + " " + df["content"]).apply(clean_text)

    for pattern, keywords in tqdm(DVL_KEYWORD_SEEDS.items(), desc="패턴 추출"):
        escaped = [re.escape(kw) for kw in keywords]
        df[pattern] = df["clean_message"].str.contains("|".join(escaped), case=False, na=False).astype(int)

    logger.info("=== STEP 3: 저장 ===")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"완료! 저장 경로: {out_path} (총 {len(df)}행)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/unified_news_processed.csv")
    parser.add_argument("--min_content_len", type=int, default=10)
    args = parser.parse_args()
    preprocess_logic(args.raw_dir, args.output, args.min_content_len)