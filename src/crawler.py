# 네이버 뉴스 크롤러

import os
import asyncio
import random
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# ── DVL 플래그 키워드 ──────────────────────────────────────────────────────────
# 단일 단어 대신 가짜뉴스에서만 자주 쓰이는 다중 단어 패턴 위주로 구성
# 진짜 뉴스에도 흔한 단어(때문에, 모든, 관계자 등)는 제외

# 통계 왜곡: 근거 없이 수치를 절대화·과장하는 표현
_KW_STAT_DISTORTION = [
    '100% 효과', '100% 안전', '100% 확실', '100% 증명',
    '단 한 명도', '예외 없이', '전원 사망', '전원 감염',
    '완전히 사라졌다', '완전 치료', '기적의 치료',
    '완벽하게 차단', '100% 예방',
    '100% effective', '100% proven', 'without exception', '100% safe',
    'guaranteed results', 'zero exceptions',
]

# 인과 오류: 근거 없는 단정적 인과관계 주장
_KW_CAUSAL_ERROR = [
    '이것이 원인', '직접적인 원인은', '유일한 원인',
    '먹으면 낫는다', '하면 무조건 낫는', '이것만 먹으면',
    '기적의 식품', '기적의 약', '만병통치',
    '의학적으로 증명됐다', '과학적으로 완전히 증명',
    'directly causes', 'proven to cure', 'miracle cure',
    'this one trick', 'doctors hate', 'guaranteed to work',
]

# 감정 자극: 독자의 공포·분노·충격을 유발하는 선동적 표현
_KW_EMOTIONAL_PROVOCATION = [
    '충격 고백', '충격 폭로', '충격적인 진실', '충격 반전',
    '경악스러운', '경악을 금치 못',
    '분노 폭발', '공분을 사', '국민적 공분',
    '당신만 모르는', '이것을 숨겼다', '쉬쉬하던',
    '절대 알려주지 않는', '믿기지 않는 사실',
    '상상도 못한 진실', '충격! 충격', '경고!!!',
    'shocking truth', "they don't want you to know",
    'what they hide', 'wake up sheeple', 'exposed!!!',
]

# 출처 불명: 익명·불확실 출처에만 의존하는 패턴
_KW_SOURCE_LACK = [
    '소식통에 따르면', '익명의 관계자', '익명을 요구한',
    '카더라', '~라는 말이 있다', '알 수 없는 소식통',
    '정체불명의', '확인되지 않은 보도', '출처 불명',
    '누군가에 따르면', '업계 일각에서는 주장',
    'anonymous sources claim', 'sources say without evidence',
    'unconfirmed reports', 'rumor has it',
]

# 이미지 불일치: 이미지 조작·오용을 본문에서 직접 언급하는 패턴
_KW_IMG_MISMATCH = [
    '사진 속 인물은', '이 사진은 실제로', '조작된 사진',
    '합성 사진', '가짜 사진', '이미지를 도용',
    '실제 사진이 아닌', '다른 나라 사진을', '오래된 사진을 사용',
    '사진이 조작됐다', '이미지 합성',
    'photo is fake', 'image was manipulated', 'old photo used as',
    'photo from different', 'doctored image',
]

# ── UA 풀: 다양한 브라우저/버전 로테이션으로 차단 회피 ───────────────────────
_USER_AGENTS = [
    # Chrome Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    # Chrome Mac
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    # Edge Windows
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
    # Firefox
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0',
]

_BASE_HEADERS = {
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer':         'https://news.naver.com/',
    'Connection':      'keep-alive',
    'Cache-Control':   'no-cache',
}

def _random_headers() -> dict:
    """매 요청마다 UA를 랜덤 선택해 반환."""
    return {**_BASE_HEADERS, 'User-Agent': random.choice(_USER_AGENTS)}

# 목록 수집용 고정 헤더 (세션 단위 사용)
HEADERS = _random_headers()

CRAWL_WORKERS   = 8    # 동시 요청 수 (차단 방지)
MAX_CRAWL_PAGES = 20

_BODY_DELAY_MIN = 0.3  # 본문 요청 간 최소 대기(초)
_BODY_DELAY_MAX = 0.9  # 본문 요청 간 최대 대기(초)
_RETRY_COUNT    = 2    # 실패 시 재시도 횟수
_RETRY_DELAY    = 1.5  # 재시도 전 대기(초)

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
    """aiohttp 커넥터 + 타임아웃 공통 생성.
    ThreadedResolver: Windows ProactorEventLoop에서 async DNS 실패 문제 해결."""
    resolver  = aiohttp.ThreadedResolver()
    connector = aiohttp.TCPConnector(limit=CRAWL_WORKERS, resolver=resolver)
    timeout   = aiohttp.ClientTimeout(total=20)
    return connector, timeout


async def _get_soup(session: aiohttp.ClientSession, url: str) -> BeautifulSoup | None:
    """단일 URL 비동기 fetch → BeautifulSoup 반환. 실패 시 재시도 후 None."""
    for attempt in range(_RETRY_COUNT):
        try:
            hdrs = _random_headers()
            async with session.get(url, headers=hdrs) as resp:
                if resp.status == 429:
                    wait = _RETRY_DELAY * (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                text = await resp.text(errors='replace')
                return await asyncio.to_thread(BeautifulSoup, text, 'lxml')
        except Exception as e:
            if attempt < _RETRY_COUNT - 1:
                await asyncio.sleep(_RETRY_DELAY)
            else:
                print(f'[crawler] request failed: {url} -> {e}')
    return None


async def _fetch_article_body(session: aiohttp.ClientSession, url: str) -> str:
    """기사 상세 페이지 본문 비동기 크롤링. 랜덤 딜레이 + 재시도 포함."""
    await asyncio.sleep(random.uniform(_BODY_DELAY_MIN, _BODY_DELAY_MAX))

    for attempt in range(_RETRY_COUNT):
        try:
            hdrs = _random_headers()
            async with session.get(url, headers=hdrs) as resp:
                if resp.status == 429:
                    await asyncio.sleep(_RETRY_DELAY * (attempt + 2))
                    continue
                resp.raise_for_status()
                text = await resp.text(errors='replace')
                return await asyncio.to_thread(_extract_body_sync, text)
        except Exception as e:
            if attempt < _RETRY_COUNT - 1:
                await asyncio.sleep(_RETRY_DELAY)
            else:
                print(f'[crawler] body fetch failed: {url} -> {e}')
    return ''


def _extract_body_sync(text: str) -> str:
    """CPU 소모가 큰 본문 파싱을 스레드풀에서 처리."""
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
    """
    items = (soup.select('.list_body ul.type06_headline li')
             + soup.select('.list_body ul.type06 li')
             + soup.select('.list_body ul.type02 li'))

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
            if any(x['url'] == url for x in articles):
                continue
            if any(x['title'] == title for x in articles):
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
                'text':       summary,
                'body_ready': False,
            })
        except Exception:
            continue

    return idx


# ── 공개 API ───────────────────────────────────────────────────────────────

async def fetch_listing_pages(max_pages: int = MAX_CRAWL_PAGES,
                               base_url: str = None) -> list[dict]:
    """Phase 1: 목록 페이지를 max_pages개 동시 요청으로 수집."""
    if base_url is None:
        base_url = BASE_FLASH_URL

    page_nums = list(range(1, max_pages + 1))
    connector, timeout = _make_connector_timeout()

    print(f'[crawler] {max_pages}페이지 동시 fetch 시작 (workers={CRAWL_WORKERS})...')
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:
        soups_list = await asyncio.gather(
            *[_get_soup(session, f"{base_url}&page={p}") for p in page_nums],
            return_exceptions=True,
        )

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
    """Phase 2: 기사 본문을 비동기로 병렬 수집.
    Semaphore로 동시 요청 수를 CRAWL_WORKERS로 제한."""
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
        connector=connector, timeout=timeout
    ) as session:
        await asyncio.gather(*[_fetch_one(a, session) for a in to_enrich])

    save_to_csv(articles)
    print(f'[crawler-enrich] 전체 본문 수집 완료: {len(articles)}건')


# ── CSV 저장 ──────────────────────────────────────────────────────────────

def save_to_csv(articles: list[dict], filename: str = "data/realtime/realtime_news.csv"):
    if not articles:
        print("[crawler] 저장할 뉴스 데이터가 없습니다.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    raw      = pd.DataFrame(articles)
    text_col  = raw['text'].fillna('').astype(str)
    title_col = raw['title'].fillna('').astype(str)
    # 제목 + 본문을 합쳐서 검사 (가짜뉴스 패턴은 제목에서도 자주 나타남)
    full_col  = title_col + ' ' + text_col
    today     = datetime.now().strftime('%Y-%m-%d')

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
    out['stat_distortion']       = _dvl(full_col, _KW_STAT_DISTORTION)
    out['causal_error']          = _dvl(full_col, _KW_CAUSAL_ERROR)
    out['emotional_provocation'] = _dvl(full_col, _KW_EMOTIONAL_PROVOCATION)
    out['source_lack']           = _dvl(full_col, _KW_SOURCE_LACK)
    out['img_mismatch']          = _dvl(full_col, _KW_IMG_MISMATCH)

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
