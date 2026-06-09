# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
팩트체크 가짜뉴스 데이터 수집 크롤러 (가짜뉴스 전용)

수집 소스:
  - KBS 팩트체크K:    news.kbs.co.kr
  - 연합뉴스 팩트체크: www.yna.co.kr
  - 뉴스톱:           www.newstof.com

출력:
  data/factcheck/factcheck_raw.csv   — 수집된 원본 (가짜뉴스만)
  data/factcheck/factcheck_label.csv — 파이프라인 형식 변환본

사용법:
  python src/factcheck_crawler.py
  python src/factcheck_crawler.py --max 200   # 소스별 최대 200건
  python src/factcheck_crawler.py --source kbs newstof
"""

import argparse
import asyncio
import os
import random
import re
import time
from datetime import datetime

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "factcheck")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 판정 → 레이블 매핑 ────────────────────────────────────────────────────────
# 0 = 진짜(사실)  1 = 가짜(거짓/오해)
VERDICT_MAP = {
    # 거짓 계열 → 가짜(1)
    "전혀 사실 아님":   1,
    "대체로 사실 아님": 1,
    "사실 아님":        1,
    "사실이 아닙":      1,
    "사실과 다릅":      1,
    "사실과 달리":      1,
    "근거가 없":        1,
    "근거 없음":        1,
    "근거없음":         1,
    "확인되지 않":      1,
    "확인할 수 없":     1,
    "과장됐":           1,
    "과장되었":         1,
    "왜곡됐":           1,
    "왜곡되었":         1,
    "틀렸":             1,
    "맞지 않":          1,
    "거짓":             1,
    "오해":             1,
    "왜곡":             1,
    "false":            1,
    "mostly false":     1,
    "pants on fire":    1,
    # 사실 계열 → 진짜(0)
    "대체로 사실":      0,
    "절반의 사실":      0,
    "사실입니다":       0,
    "사실로 확인":      0,
    "사실이 맞":        0,
    "확인됐습니다":     0,
    "확인되었습니다":   0,
    "맞습니다":         0,
    "사실":             0,
    "true":             0,
    "mostly true":      0,
    "half true":        0,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}


async def _fetch(session: aiohttp.ClientSession, url: str, referer: str = "") -> str | None:
    try:
        headers = {**HEADERS}
        if referer:
            headers["Referer"] = referer
        await asyncio.sleep(random.uniform(0.8, 2.0))
        async with session.get(url, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status == 200:
                return await r.text(errors="replace")
    except Exception as e:
        print(f"  [fetch 실패] {url[:60]} → {e}")
    return None


def _extract_verdict(text: str) -> tuple[str, int]:
    """본문에서 판정 결과 추출 → (판정문자열, 0/1/-1)."""
    # "판정:" 패턴 우선
    m = re.search(r"판정\s*[:：]\s*([^\n\r]{2,20})", text)
    if m:
        vstr = m.group(1).strip()
        for kw, label in VERDICT_MAP.items():
            if kw in vstr:
                return vstr, label
    # 키워드 순차 탐색
    for kw, label in VERDICT_MAP.items():
        if kw in text:
            return kw, label
    return "", -1


# ── KBS 팩트체크K ─────────────────────────────────────────────────────────────

KBS_SEED_NCDS = [
    7934085, 7921034, 7910287, 7898234, 7881029,
    7860123, 7834578, 7812349, 7790234, 7765432,
    7743210, 7720987, 7698765, 7670234, 7645678,
    7620345, 7598234, 7570123, 7545678, 7520987,
]


def _parse_kbs_article(html: str, ncd: int) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    full_text = soup.get_text()
    if "팩트체크K" not in full_text and "팩트체크" not in full_text:
        return None

    title_tag = (soup.find("h4", class_=re.compile("headline|title|news-title"))
                 or soup.find("p",  class_=re.compile("headline|title"))
                 or soup.find("h1") or soup.find("title"))
    title = title_tag.get_text(strip=True) if title_tag else ""
    if not title:
        return None

    body_div = (soup.find("div", class_=re.compile(r"view-article|article-body|news-content|cont_newstext"))
                or soup.find("div", id=re.compile(r"cont_newstext|article")))
    body = body_div.get_text(separator=" ", strip=True) if body_div else ""
    if len(body) < 100:
        body = " ".join(p.get_text(strip=True) for p in soup.find_all("p")
                        if len(p.get_text(strip=True)) > 30)
    if len(body) < 100:
        return None

    verdict_raw, label = _extract_verdict(body)
    if label == -1:
        verdict_raw, label = _extract_verdict(full_text)

    date_tag = (soup.find("em",   class_=re.compile("date|time|pub"))
                or soup.find("span", class_=re.compile("date|time|pub")))
    date_str = ""
    if date_tag:
        m = re.search(r"\d{4}\.\d{2}\.\d{2}", date_tag.get_text())
        date_str = m.group(0).replace(".", "-") if m else ""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    return dict(source="KBS팩트체크K",
                url=f"https://news.kbs.co.kr/news/pc/view/view.do?ncd={ncd}",
                title=title, content=body[:5000],
                verdict=verdict_raw, label=label, date=date_str)


def _find_related_ncds(html: str) -> list[int]:
    return [int(n) for n in set(re.findall(r"ncd=(\d{7,8})", html))]


async def crawl_kbs(max_fake: int = 200) -> list[dict]:
    print(f"\n[KBS 팩트체크K] 가짜뉴스 수집 시작 (목표: {max_fake}건)")
    results, visited = [], set()
    queue = list(KBS_SEED_NCDS)
    connector = aiohttp.TCPConnector(limit=5, resolver=aiohttp.ThreadedResolver())

    async with aiohttp.ClientSession(connector=connector) as session:
        while queue and len(results) < max_fake:
            batch, queue = queue[:10], queue[10:]
            tasks = [(ncd, _fetch(session,
                                  f"https://news.kbs.co.kr/news/pc/view/view.do?ncd={ncd}",
                                  "https://news.kbs.co.kr/"))
                     for ncd in batch if ncd not in visited and not visited.add(ncd)]

            for ncd, coro in tasks:
                html = await coro
                if not html:
                    continue
                article = _parse_kbs_article(html, ncd)
                if article and article["label"] == 1:   # 가짜뉴스만 수집
                    results.append(article)
                    print(f"  [KBS {len(results):3d}] 가짜 | {article['title'][:50]}")

                queue.extend(n for n in _find_related_ncds(html) if n not in visited)

            if not queue and KBS_SEED_NCDS:
                base = random.choice(KBS_SEED_NCDS)
                queue.extend(base + random.randint(-500, 500) for _ in range(20))

    print(f"[KBS] 완료: 가짜뉴스 {len(results)}건")
    return results


# ── 연합뉴스 팩트체크 ─────────────────────────────────────────────────────────

async def crawl_yna(max_fake: int = 200) -> list[dict]:
    print(f"\n[연합뉴스] 팩트체크 크롤링 시작 (목표: {max_fake}건)")
    results = []
    connector = aiohttp.TCPConnector(limit=3, resolver=aiohttp.ThreadedResolver())

    async with aiohttp.ClientSession(connector=connector) as session:
        for page_from in range(0, max_fake * 5, 20):   # 가짜만 필터하므로 더 많이 시도
            if len(results) >= max_fake:
                break
            url = (f"https://www.yna.co.kr/search/index"
                   f"?query=팩트체크+사실확인&ctype=A&period=3y&from={page_from}&size=20")
            html = await _fetch(session, url, "https://www.yna.co.kr/")
            if not html:
                break

            soup = BeautifulSoup(html, "lxml")
            links = [a["href"] for a in soup.find_all("a", href=re.compile(r"/view/AKR\d+"))]

            for link in links[:20]:
                if len(results) >= max_fake:
                    break
                art_url = f"https://www.yna.co.kr{link}" if link.startswith("/") else link
                art_html = await _fetch(session, art_url, "https://www.yna.co.kr/")
                if not art_html:
                    continue

                art_soup  = BeautifulSoup(art_html, "lxml")
                title_tag = art_soup.find("h1", class_=re.compile("tit|title")) or art_soup.find("h1")
                body_div  = art_soup.find("div", class_=re.compile("story-news|article|content"))
                if not title_tag or not body_div:
                    continue

                title = title_tag.get_text(strip=True)
                body  = body_div.get_text(separator=" ", strip=True)
                if len(body) < 100:
                    continue

                verdict_raw, label = _extract_verdict(body + " " + title)
                if label != 1:          # 가짜뉴스만
                    continue

                results.append(dict(source="연합뉴스팩트체크",
                                    url=art_url, title=title,
                                    content=body[:5000],
                                    verdict=verdict_raw, label=1,
                                    date=datetime.now().strftime("%Y-%m-%d")))
                print(f"  [연합 {len(results):3d}] 가짜 | {title[:50]}")

    print(f"[연합뉴스] 완료: 가짜뉴스 {len(results)}건")
    return results


# ── 뉴스톱 (newstof.com) ──────────────────────────────────────────────────────

NEWSTOF_BASE = "https://www.newstof.com"
NEWSTOF_LIST = NEWSTOF_BASE + "/news/articleList.html?sc_section_code=S1N0&view_type=sm&page={page}"


async def crawl_newstof(max_fake: int = 300) -> list[dict]:
    """뉴스톱 팩트체크 — 거짓 판정 기사만 수집."""
    print(f"\n[뉴스톱] 크롤링 시작 (목표: {max_fake}건)")
    results  = []
    visited  = set()
    connector = aiohttp.TCPConnector(limit=3, resolver=aiohttp.ThreadedResolver())

    async with aiohttp.ClientSession(connector=connector) as session:
        page = 1
        while len(results) < max_fake:
            list_url = NEWSTOF_LIST.format(page=page)
            html = await _fetch(session, list_url, NEWSTOF_BASE)
            if not html:
                break

            soup  = BeautifulSoup(html, "lxml")
            links = [a["href"] for a in soup.find_all("a", href=re.compile(r"articleView\.html\?idxno=\d+"))]
            links = [l if l.startswith("http") else NEWSTOF_BASE + l for l in links]
            links = list(dict.fromkeys(links))  # 중복 제거 (순서 유지)

            if not links:
                break

            for art_url in links:
                if len(results) >= max_fake or art_url in visited:
                    continue
                visited.add(art_url)

                art_html = await _fetch(session, art_url, list_url)
                if not art_html:
                    continue

                art_soup  = BeautifulSoup(art_html, "lxml")

                # 제목
                title_tag = (art_soup.find("h3", class_=re.compile("heading|title|article"))
                             or art_soup.find("h1")
                             or art_soup.find("title"))
                title = title_tag.get_text(strip=True) if title_tag else ""
                if not title:
                    continue

                # 본문
                body_div = (art_soup.find("div", id=re.compile("article-view-content|articleBody"))
                            or art_soup.find("div", class_=re.compile("article-view-content|view-con|article_txt")))
                body = body_div.get_text(separator=" ", strip=True) if body_div else ""
                if len(body) < 100:
                    paras = [p.get_text(strip=True) for p in art_soup.find_all("p") if len(p.get_text(strip=True)) > 30]
                    body  = " ".join(paras)
                if len(body) < 100:
                    continue

                # 판정
                full_text = art_soup.get_text()
                verdict_raw, label = _extract_verdict(body)
                if label == -1:
                    verdict_raw, label = _extract_verdict(full_text)
                if label != 1:          # 가짜뉴스만
                    continue

                # 날짜
                date_str = ""
                date_tag = (art_soup.find("em",   class_=re.compile("date|time"))
                            or art_soup.find("span", class_=re.compile("date|time"))
                            or art_soup.find("li",   class_=re.compile("date|time")))
                if date_tag:
                    m = re.search(r"\d{4}[.\-]\d{2}[.\-]\d{2}", date_tag.get_text())
                    date_str = re.sub(r"\.", "-", m.group(0)) if m else ""
                if not date_str:
                    date_str = datetime.now().strftime("%Y-%m-%d")

                results.append(dict(source="뉴스톱",
                                    url=art_url, title=title,
                                    content=body[:5000],
                                    verdict=verdict_raw, label=1,
                                    date=date_str))
                print(f"  [뉴스톱 {len(results):3d}] 가짜 | {title[:50]}")

            page += 1

    print(f"[뉴스톱] 완료: 가짜뉴스 {len(results)}건")
    return results


# ── DVL 플래그 자동 태깅 ─────────────────────────────────────────────────────

def _add_dvl_flags(df: pd.DataFrame) -> pd.DataFrame:
    text = df["content"].fillna("").astype(str)

    def _flag(col, keywords):
        return col.apply(lambda x: 1 if any(w in x for w in keywords) else 0)

    df["stat_distortion"]        = _flag(text, ["100%", "모든", "항상", "절대", "never", "always"])
    df["causal_error"]           = _flag(text, ["때문에", "원인", "증명", "결과적으로", "causes", "proves"])
    df["emotional_provocation"]  = _flag(text, ["충격", "경악", "분노", "shocking", "outrage"])
    df["source_lack"]            = _flag(text, ["관계자", "소식통", "익명", "sources say", "reportedly"])
    df["img_mismatch"]           = _flag(text, ["사진 속", "이 사진은", "photo shows"])
    return df


# ── 메인 ──────────────────────────────────────────────────────────────────────

async def main_async(sources: list[str], max_per_source: int):
    all_results = []

    if "kbs" in sources:
        all_results.extend(await crawl_kbs(max_per_source))
    if "yna" in sources:
        all_results.extend(await crawl_yna(max_per_source))
    if "newstof" in sources:
        all_results.extend(await crawl_newstof(max_per_source))

    if not all_results:
        print("\n[경고] 수집된 가짜뉴스 데이터가 없습니다.")
        return

    # 원본 저장
    raw_path = os.path.join(OUT_DIR, "factcheck_raw.csv")
    # 기존 raw 파일이 있으면 합치기
    if os.path.exists(raw_path):
        existing = pd.read_csv(raw_path, encoding="utf-8-sig")
        df_raw   = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True)
        df_raw   = df_raw.drop_duplicates(subset=["url"], keep="first")
    else:
        df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n원본 저장: {raw_path}  ({len(df_raw)}건)")

    # 파이프라인 형식으로 변환 (가짜뉴스만이므로 label=1 전체)
    df_fake = df_raw[df_raw["label"] == 1].copy()
    df_fake = _add_dvl_flags(df_fake)

    df_out = pd.DataFrame({
        "title":                df_fake["title"],
        "content":              df_fake["content"],
        "media":                df_fake["source"],
        "date":                 df_fake["date"],
        "label":                df_fake["label"],
        "url":                  df_fake["url"],
        "clean_message":        df_fake["content"].str.replace(
                                    r"[^가-힣a-zA-Z0-9\s]", "", regex=True).str.lower(),
        "verdict":              df_fake["verdict"],
        "stat_distortion":      df_fake["stat_distortion"],
        "causal_error":         df_fake["causal_error"],
        "emotional_provocation":df_fake["emotional_provocation"],
        "source_lack":          df_fake["source_lack"],
        "img_mismatch":         df_fake["img_mismatch"],
    })

    out_path = os.path.join(OUT_DIR, "factcheck_label.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n최종 데이터: {out_path}")
    print(f"  가짜(1): {len(df_out)}건")
    print(f"  소스 분포: {df_fake['source'].value_counts().to_dict()}")
    print(f"\n다음 단계: python src/merge_factcheck.py")


def main():
    parser = argparse.ArgumentParser(description="팩트체크 가짜뉴스 크롤러 (가짜뉴스 전용)")
    parser.add_argument("--source", nargs="+",
                        choices=["kbs", "yna", "newstof"],
                        default=["kbs", "yna", "newstof"],
                        help="크롤링 소스 (기본: 전체)")
    parser.add_argument("--max", type=int, default=200,
                        help="소스별 최대 수집 건수 (기본: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print(" 팩트체크 가짜뉴스 크롤러 (거짓 판정 기사 전용)")
    print(f" 소스: {args.source}  |  소스별 최대: {args.max}건")
    print("=" * 60)

    asyncio.run(main_async(args.source, args.max))


if __name__ == "__main__":
    main()
