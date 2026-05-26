# 네이버 뉴스 크롤러 

import os
import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://news.naver.com/',
    'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
}

CRAWL_WORKERS  = 20   # 동시 요청 수 
MAX_CRAWL_PAGES = 20  # 수집할 최대 페이지 수

BASE_FLASH_URL = 'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=001'

CATEGORY_BASE_URLS: dict[str, str] = {
    '001':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=001',
    '100':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=100',
    '101':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=101',
    '102':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=102',
    '103':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=103',
    '104':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=104',
    '105':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=105',
    '110':    'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=110',
    'yonhap': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=001&sid2=140&oid=001&isYeonhapFlash=Y',
}


# ── 내부 유틸 ──────────────────────────────────────────────────────────────

def _make_connector_timeout() -> tuple[aiohttp.TCPConnector, aiohttp.ClientTimeout]:
    """
    aiohttp 커넥터 + 타임아웃 공통 생성.
    ThreadedResolver: Windows ProactorEventLoop에서 async DNS 실패 문제 해결.
    """
    resolver  = aiohttp.ThreadedResolver()       
    connector = aiohttp.TCPConnector(limit=CRAWL_WORKERS, resolver=resolver)
    timeout   = aiohttp.ClientTimeout(total=15)
    return connector, timeout


async def _get_soup(session: aiohttp.ClientSession, url: str) -> BeautifulSoup | None:
    """단일 URL 비동기 fetch → BeautifulSoup 반환. 실패 시 None."""
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            text = await resp.text(errors='replace')
            return await asyncio.to_thread(BeautifulSoup,text, 'lxml')
    except Exception as e:
        print(f'[crawler] request failed: {url} -> {e}')
        return None


async def _fetch_article_body(session: aiohttp.ClientSession, url: str) -> str:
    """기사 상세 페이지 본문 비동기 크롤링 (네트워크 다운로드만 비동기 처리)"""
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            text = await resp.text(errors='replace')
            
            # [최적화 3] 다운로드된 텍스트를 스레드 풀로 넘겨 백그라운드에서 정제 및 추출합니다.
            return await asyncio.to_thread(_extract_body_sync, text)
    except Exception as e:
        print(f'[crawler] body fetch failed: {url} -> {e}')
        return ''


def _extract_body_sync(text: str) -> str:
    """
    CPU 소모가 큰 본문 파싱, 태그 탐색, decompose(제거) 연산을 
    하나의 동기 함수로 묶어 멀티스레드 환경에서 처리하도록 분리합니다.
    """
    soup = BeautifulSoup(text, 'lxml')
    for sel in [
        '#dic_area',
        '._article_body_contents',
        '#articleBodyContents',
        '.go_trans._article_content',
        '.article_body',
    ]:
        el = soup.select_one(sel)
        if el:
            for tag in el.select('script, style, .reporter_area, .copyright, .article-sns-group'):
                tag.decompose()
            return el.get_text(separator=' ', strip=True)
    return ''



def _parse_listing_page(soup: BeautifulSoup, articles: list[dict], idx: int) -> int:
    """목록 페이지 soup에서 기사 메타데이터를 파싱해 articles에 추가. 갱신된 idx 반환.

    지원 구조:
      - ul.type06_headline / ul.type06 : 요약형 (dt 기반, dd 안에 press/date)
      - ul.type02                       : 제목형 (li 안에 a·span 직접 배치)
        → 연합뉴스 속보(LPOD 모드) 등에서 사용
    """
    items = (soup.select('.list_body ul.type06_headline li')
             + soup.select('.list_body ul.type06 li')
             + soup.select('.list_body ul.type02 li'))   # ← 연합뉴스 LPOD 구조 추가

    for item in items:
        try:
            img_tag   = item.select_one('dt.photo img')
            image_url = img_tag['src'] if img_tag else None

            dts = item.select('dt')
            if dts:
          
                a_tag = dts[-1].select_one('a')
            else:
               
                a_tag = item.select_one('a')
            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            url   = a_tag.get('href', '')

            summary_el = item.select_one('dd span.lede')
            summary    = summary_el.get_text(strip=True) if summary_el else ''

            press_el = (item.select_one('dd span.writing')
                        or item.select_one('span.writing'))
            press    = press_el.get_text(strip=True) if press_el else '알 수 없음'

            time_el  = (item.select_one('dd span.date')
                        or item.select_one('span.date'))
            pub_time = time_el.get_text(strip=True) if time_el else datetime.now().strftime('%H:%M')

            if not title or not url:
                continue
            if any(x['url'] == url for x in articles):   # 중복 방지
                continue

            idx += 1
            safe_press = press.encode('ascii', 'ignore').decode() or f'press_{idx}'
            print(f'[crawler] ({idx}) {safe_press} - {title[:25]}...')

            articles.append({
                'title':      title,
                'press':      press,
                'category':   '정치/속보',
                'time':       pub_time,
                'url':        url,
                'image_url':  image_url,
                'summary':    summary,
                'text':       summary,    # Phase 2에서 본문으로 교체
                'body_ready': False,
            })
        except Exception:
            continue

    return idx


# ── 공개 API ───────────────────────────────────────────────────────────────

async def fetch_listing_pages(max_pages: int = MAX_CRAWL_PAGES,
                               base_url: str = None) -> list[dict]:
    """Phase 1: 목록 페이지를 max_pages개 동시 요청으로 수집 (aiohttp + asyncio.gather).
    app.py에서는 메인=3, 메인섹션=2, 세부섹션=1 로 호출."""
    if base_url is None:
        base_url = BASE_FLASH_URL

    page_nums = list(range(1, max_pages + 1))
    connector, timeout = _make_connector_timeout()

    print(f'[crawler] {max_pages}페이지 동시 fetch 시작 (workers={CRAWL_WORKERS})...')
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector, timeout=timeout
    ) as session:
        soups_list = await asyncio.gather(
            *[_get_soup(session, f"{base_url}&page={p}") for p in page_nums],
            return_exceptions=True,
        )

    # 실패(None / Exception)를 걸러내고 페이지 번호와 매핑
    soups: dict[int, BeautifulSoup] = {
        p: s
        for p, s in zip(page_nums, soups_list)
        if isinstance(s, BeautifulSoup)
    }

    articles: list[dict] = []
    idx = 0
    empty_streak = 0
    for page_no in sorted(soups):
        before = len(articles)
        idx = _parse_listing_page(soups[page_no], articles, idx)
        if len(articles) == before:
            empty_streak += 1
            print(f'[crawler] {page_no} 페이지 기사 없음 (연속 {empty_streak}회)')
            if empty_streak >= 2:
                print('[crawler] 빈 페이지 연속 → 이후 페이지 무시')
                break
        else:
            empty_streak = 0

    print(f'[crawler] 수집 완료: 총 {len(articles)}건 ({len(soups)}페이지)')
    return articles if articles else _dummy_articles()


async def enrich_bodies(articles: list[dict]) -> None:
    """
    Phase 2: 기사 본문을 비동기로 병렬 수집.
    Semaphore로 동시 요청 수를 CRAWL_WORKERS로 제한.
    """
    to_enrich = [a for a in articles if not a.get('body_ready')]
    total = len(to_enrich)
    done  = 0
    sem   = asyncio.Semaphore(CRAWL_WORKERS)

    async def _fetch_one(article: dict, session: aiohttp.ClientSession) -> None:
        nonlocal done
        async with sem:
            try:
                body = await _fetch_article_body(session, article['url'])
                if body:
                    article['text'] = body
            except Exception:
                pass
            finally:
                article['body_ready'] = True
                done += 1
                if done % 10 == 0:
                    print(f'[crawler-enrich] 본문 수집 {done}/{total}건 완료')

    connector, timeout = _make_connector_timeout()
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector, timeout=timeout
    ) as session:
        await asyncio.gather(*[_fetch_one(a, session) for a in to_enrich])

    save_to_csv(articles)
    print(f'[crawler-enrich] 전체 본문 수집 완료: {len(articles)}건')


# ── CSV 저장 ──────────────────────────────────────────────────────────────

def save_to_csv(articles: list[dict], filename: str = "data/realtime/realtime_news.csv"):
    """
    ── 표준 스키마 컬럼  ─────────────────────────────
    id                   : int   — 행 고유 식별자 (1부터 순번)
    title                : str   — 기사 제목 (원문)
    content              : str   — 기사 본문 (원문, 최대 10,000자). Phase1=요약, Phase2=전문
    media                : str   — 출처 매체명 (불명 시 Unknown)
    date                 : str   — 수집일 (YYYY-MM-DD)
    label                : int   — 정답 레이블 (0=진짜/1=가짜). 추론용 CSV는 NaN
    clean_message        : str   — 학습용 정제 텍스트 (소문자화·특수문자 제거)
    stat_distortion      : int   — [DVL 1] 통계 왜곡 패턴 (0/1)
    causal_error         : int   — [DVL 2] 인과 오류 패턴 (0/1)
    emotional_provocation: int   — [DVL 3] 감정 자극 패턴 (0/1)
    source_lack          : int   — [DVL 4] 출처 불명 패턴 (0/1)
    img_mismatch         : int   — [DVL 5] 이미지 불일치 패턴 (0/1)

    ── 실시간 크롤링 추가 컬럼 ──────────────────────────
    url                  : str   — 기사 원문 URL
    category             : str   — 기사 카테고리
    pub_time             : str   — 목록 페이지 발행 시각
    image_url            : str   — 대표 썸네일 URL (없으면 None)
    summary              : str   — 목록 리드 요약문
    fake_score           : float — AI 탐지 점수 (0.0~100.0)
    body_ready           : bool  — Phase 2 본문 수집 완료 여부
    """
    if not articles:
        print("[crawler] 저장할 뉴스 데이터가 없습니다.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    raw      = pd.DataFrame(articles)
    text_col = raw['text'].fillna('').astype(str)
    today    = datetime.now().strftime('%Y-%m-%d')

    def _dvl(col, keywords):
        return col.apply(lambda x: 1 if any(w in x for w in keywords) else 0)

    out = pd.DataFrame()
    out['id']      = range(1, len(raw) + 1)
    out['title']   = raw['title']
    out['content'] = text_col.str[:10_000]
    out['media']   = raw['press'].fillna('Unknown').replace('', 'Unknown')
    out['date']    = today
    out['label']   = None
    out['clean_message'] = (
        text_col.str.replace(r'[^가-힣a-zA-Z0-9\s]', '', regex=True).str.lower()
    )
    out['stat_distortion']         = _dvl(text_col, ['100%', '모든', '항상', '절대', 'never', 'always'])
    out['causal_error']            = _dvl(text_col, ['때문에', '원인', '증명', '결과적으로', 'causes', 'proves'])
    out['emotional_provocation']   = _dvl(text_col, ['충격', '경악', '분노', 'shocking', 'outrage', 'unbelievable'])
    out['source_lack']             = _dvl(text_col, ['관계자', '소식통', '익명', 'sources say', 'reportedly'])
    out['img_mismatch']            = _dvl(text_col, ['사진 속', '이 사진은', 'photo shows', 'pictured here'])

    out['url']        = raw['url']
    out['category']   = raw['category']   if 'category'   in raw.columns else '속보'
    out['pub_time']   = raw['time']       if 'time'       in raw.columns else ''
    out['image_url']  = raw['image_url']  if 'image_url'  in raw.columns else None
    out['summary']    = raw['summary']    if 'summary'    in raw.columns else ''
    out['fake_score'] = raw['fake_score'] if 'fake_score' in raw.columns else None
    out['body_ready'] = raw['body_ready'] if 'body_ready' in raw.columns else False

    out.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"[crawler] 총 {len(out)}개 뉴스 CSV 저장 완료: {filename}")

    flag_cols = ['stat_distortion', 'causal_error', 'emotional_provocation',
                 'source_lack', 'img_mismatch']
    for article, flags in zip(articles, out[flag_cols].to_dict('records')):
        article.update({col: int(v) for col, v in flags.items()})


def _dummy_articles() -> list[dict]:
    """크롤링 완전 실패 시 화면에 표시할 예시 기사."""
    return [
        {
            'title':     '네이버 뉴스 크롤링에 실패했습니다. 잠시 후 다시 시도해주세요.',
            'press':     '시스템',
            'category':  '안내',
            'time':      datetime.now().strftime('%H:%M'),
            'url':       'https://news.naver.com',
            'image_url': None,
            'summary':   '크롤러가 기사를 가져오지 못했습니다.',
            'text':      '',
        },
    ]