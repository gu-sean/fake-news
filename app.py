"""
네이버 뉴스 클론 — AI 가짜뉴스 탐지 속보 페이지
접속: http://localhost:8000
"""

import sys
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 크롤링 전용 스레드풀 — FastAPI 이벤트루프와 분리해 페이지 응답 지연 방지
# aiohttp는 이 풀의 스레드 안에서 asyncio.run()으로 독립 이벤트루프 생성
_crawl_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='crawler')

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.crawler import fetch_listing_pages, enrich_bodies, CATEGORY_BASE_URLS
from src.detector import get_sorted_timeline


# ── 카테고리 설정 ──────────────────────────────────────────────────────────
SECTIONS: dict = {
    '100': {
        'name': '정치',
        'sub_categories': [
            {'name': '전체',      'sid2': None},
            {'name': '청와대',    'sid2': '269'},
            {'name': '국회/정당', 'sid2': '273'},
            {'name': '행정',      'sid2': '274'},
            {'name': '국방/외교', 'sid2': '275'},
            {'name': '북한',      'sid2': '276'},
            {'name': '정치일반',  'sid2': '271'},
        ],
    },
    '101': {
        'name': '경제',
        'sub_categories': [
            {'name': '전체',      'sid2': None},
            {'name': '부동산',    'sid2': '260'},
            {'name': '금융',      'sid2': '258'},
            {'name': '증권',      'sid2': '259'},
            {'name': '산업/재계', 'sid2': '261'},
            {'name': '글로벌경제','sid2': '262'},
            {'name': '경제일반',  'sid2': '263'},
            {'name': '중기/벤처', 'sid2': '771'},
        ],
    },
    '102': {
        'name': '사회',
        'sub_categories': [
            {'name': '전체',      'sid2': None},
            {'name': '사건사고',  'sid2': '249'},
            {'name': '교육',      'sid2': '250'},
            {'name': '노동',      'sid2': '251'},
            {'name': '환경',      'sid2': '252'},
            {'name': '언론',      'sid2': '257'},
            {'name': '인권/복지', 'sid2': '426'},
            {'name': '지역',      'sid2': '438'},
            {'name': '인물',      'sid2': '800'},
            {'name': '사회일반',  'sid2': '248'},
        ],
    },
    '103': {
        'name': '생활/문화',
        'sub_categories': [
            {'name': '전체',          'sid2': None},
            {'name': '여행/레저',     'sid2': '237'},
            {'name': '자동차/시승기', 'sid2': '240'},
            {'name': '도로/교통',     'sid2': '241'},
            {'name': '건강정보',      'sid2': '243'},
            {'name': '공연/전시',     'sid2': '238'},
            {'name': '책',            'sid2': '239'},
            {'name': '종교',          'sid2': '244'},
            {'name': '생활/문화일반', 'sid2': '245'},
        ],
    },
    '104': {
        'name': '세계',
        'sub_categories': [
            {'name': '전체',        'sid2': None},
            {'name': '아시아/호주', 'sid2': '231'},
            {'name': '미국/중남미', 'sid2': '232'},
            {'name': '유럽',        'sid2': '233'},
            {'name': '중동/아프리카','sid2': '234'},
            {'name': '영문',        'sid2': '322'},
            {'name': '일문',        'sid2': '429'},
        ],
    },
    '105': {
        'name': 'IT/과학',
        'sub_categories': [
            {'name': '전체',       'sid2': None},
            {'name': '인터넷/SNS', 'sid2': '226'},
            {'name': '과학일반',   'sid2': '228'},
            {'name': '게임/리뷰',  'sid2': '229'},
            {'name': '컴퓨터',     'sid2': '230'},
            {'name': 'IT일반',     'sid2': '227'},
        ],
    },
    '110': {
        'name': '오피니언',
        'sub_categories': [],
    },
}

YONHAP_URL = CATEGORY_BASE_URLS['yonhap']


# ── 메인 페이지 캐시 ──────────────────────────────────────────────────────
_cache: dict = {
    'articles':   [],
    'fetched_at': 0,
    'enriching':  False,
}
CACHE_TTL = 300          # 5분
PAGE_SIZE  = 30           # 한 페이지에 보여줄 기사 수
_crawl_in_progress = False

# ── 섹션별 캐시 ───────────────────────────────────────────────────────────
_section_caches: dict[str, dict] = {}
_section_crawling: set[str] = set()

# ── 서버 시작 시 사전 크롤링할 섹션 목록 ────────────────────────────────────
# (cache_key, base_url) — 탭 전환 즉시 기사 표시를 위해 전체 섹션 사전 워밍
PREFETCH_SECTIONS: list[tuple[str, str]] = [
    (f"{sid1}:", CATEGORY_BASE_URLS[sid1])
    for sid1 in SECTIONS
    if sid1 in CATEGORY_BASE_URLS
] + [
    ('yonhap', YONHAP_URL),
]


def _get_section_cache(cache_key: str) -> dict:
    if cache_key not in _section_caches:
        _section_caches[cache_key] = {'articles': [], 'fetched_at': 0, 'enriching': False}
    return _section_caches[cache_key]


# ── 공통 크롤링 로직 ──────────────────────────────────────────────────────

async def _run_crawl():
    """
    aiohttp 크롤링을 FastAPI 이벤트루프와 분리해 실행.
    run_in_executor → 스레드 생성 → asyncio.run() → 독립 이벤트루프에서 aiohttp 동작
    결과: 크롤 중에도 탭 전환·페이지 응답이 지연되지 않음.
    """
    global _crawl_in_progress
    if _crawl_in_progress:
        return
    _crawl_in_progress = True
    loop = asyncio.get_event_loop()
    try:
        print('[app] Phase 1 시작: 목록 페이지 수집 (독립 루프)...')
        raw = await loop.run_in_executor(
            _crawl_executor,
            lambda: asyncio.run(fetch_listing_pages()) 
        )
        articles = get_sorted_timeline(raw)
        _cache['articles']   = articles
        _cache['fetched_at'] = time.time()
        _cache['enriching']  = True
        print(f'[app] Phase 1 완료: {len(articles)}건 → 즉시 화면 표시 가능')

        print('[app] Phase 2 시작: 기사 본문 백그라운드 수집...')
        await loop.run_in_executor(
            _crawl_executor,
            lambda: asyncio.run(enrich_bodies(raw))      #
        )
        _cache['fetched_at'] = time.time()
        _cache['enriching']  = False
        print('[app] Phase 2 완료')
    except Exception as e:
        print(f'[app] 크롤링 실패: {e}')
        _cache['enriching'] = False
    finally:
        _crawl_in_progress = False


async def _run_section_crawl(cache_key: str, base_url: str):
    if cache_key in _section_crawling:
        return
    _section_crawling.add(cache_key)
    cache = _get_section_cache(cache_key)
    loop  = asyncio.get_event_loop()
    try:
        print(f'[app] [{cache_key}] Phase 1 시작...')
        raw = await loop.run_in_executor(
            _crawl_executor,
            lambda: asyncio.run(fetch_listing_pages(base_url=base_url))
        )
        articles = get_sorted_timeline(raw)
        cache['articles']   = articles
        cache['fetched_at'] = time.time()
        cache['enriching']  = True
        print(f'[app] [{cache_key}] Phase 1 완료: {len(articles)}건')

        print(f'[app] [{cache_key}] Phase 2 시작...')
        await loop.run_in_executor(
            _crawl_executor,
            lambda: asyncio.run(enrich_bodies(raw))
        )
        cache['fetched_at'] = time.time()
        cache['enriching']  = False
        print(f'[app] [{cache_key}] Phase 2 완료')
    except Exception as e:
        print(f'[app] [{cache_key}] 크롤링 실패: {e}')
        cache['enriching'] = False
    finally:
        _section_crawling.discard(cache_key)


async def _staggered_warmup():
    """
    서버 시작 시 모든 섹션을 순차적으로(3초 간격) 사전 크롤링.
    동시 요청 과부하 방지를 위해 살짝 지연해서 시작.
    """
    for cache_key, base_url in PREFETCH_SECTIONS:
        await asyncio.sleep(3)   
        if cache_key not in _section_crawling:
            print(f'[app] 사전 워밍 시작: [{cache_key}]')
            asyncio.create_task(_run_section_crawl(cache_key, base_url))


async def _refresh_loop():
    """
    CACHE_TTL마다 모든 캐시를 자동 갱신.
    탭을 열지 않아도 항상 최신 기사 유지.
    """
    while True:
        await asyncio.sleep(CACHE_TTL)
        now = time.time()
        print('[app] 자동 갱신 주기 도래 — 스테일 캐시 갱신 시작')

        # 메인(속보) 캐시
        if (now - _cache['fetched_at']) > CACHE_TTL and not _crawl_in_progress:
            asyncio.create_task(_run_crawl())

        # 섹션 캐시
        for cache_key, base_url in PREFETCH_SECTIONS:
            sec_cache = _get_section_cache(cache_key)
            if (now - sec_cache['fetched_at']) > CACHE_TTL and cache_key not in _section_crawling:
                asyncio.create_task(_run_section_crawl(cache_key, base_url))
                await asyncio.sleep(2)   


# ── 서버 시작 시 즉시 첫 크롤링 시작 ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[app] 서버 시작 — 속보(메인) + 전체 섹션 사전 크롤링 시작...')
    asyncio.create_task(_run_crawl())        
    warmup_task  = asyncio.create_task(_staggered_warmup())   
    refresh_task = asyncio.create_task(_refresh_loop())       
    yield
    warmup_task.cancel()
    refresh_task.cancel()
    _crawl_executor.shutdown(wait=False)           


app = FastAPI(title="속보 - AI 가짜뉴스 탐지", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")


# ── 공통 유틸 ─────────────────────────────────────────────────────────────
VALID_VIEWS = {'title', 'summary', 'photo', 'paper'}


def _today_str() -> str:
    return (datetime.now().strftime('%Y.%m.%d. ')
            + ['월', '화', '수', '목', '금', '토', '일'][datetime.now().weekday()] + '요일')


def _paginate(all_articles: list, page: int):
    total_pages  = max(1, (len(all_articles) + PAGE_SIZE - 1) // PAGE_SIZE)
    current_page = min(page, total_pages)
    start        = (current_page - 1) * PAGE_SIZE
    return all_articles[start: start + PAGE_SIZE], current_page, total_pages


# ─────────────────────────────────────────────────────────────────────────────
# 라우트 — 메인 속보
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
async def index(
    request: Request,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
):
    if view not in VALID_VIEWS:
        view = 'title'

    now = time.time()
    is_stale = (not _cache['articles']) or ((now - _cache['fetched_at']) > CACHE_TTL)
    if is_stale and not _crawl_in_progress:
        asyncio.create_task(_run_crawl())

    all_articles = _cache['articles']
    loading      = len(all_articles) == 0
    enriching    = _cache.get('enriching', False)
    page_articles, current_page, total_pages = _paginate(all_articles, page)

    return templates.TemplateResponse(
        request=request,
        name='naver_news_clone.html',
        context={
            'news_list':       page_articles,
            'all_articles':    all_articles,
            'today':           _today_str(),
            'loading':         loading,
            'enriching':       enriching,
            'view_type':       view,
            'current_page':    current_page,
            'total_pages':     total_pages,
            'section_id':      None,
            'section_name':    None,
            'sub_categories':  [],
            'active_sid2':     None,
            'nav_url_base':    '/',
            'nav_extra_params':'',
            'cache_key':       'main',
        },
    )


@app.get('/refresh', response_class=RedirectResponse)
async def refresh():
    asyncio.create_task(_run_crawl())
    return RedirectResponse(url='/')


# ─────────────────────────────────────────────────────────────────────────────
# 라우트 — 카테고리 섹션
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/section/{sid1}', response_class=HTMLResponse)
async def section_page(
    request: Request,
    sid1: str,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
    sid2: str = Query(default=None),
):
    """카테고리 섹션 페이지 (정치·경제·사회·생활/문화·세계·IT/과학·오피니언)."""
    if sid1 not in SECTIONS:
        return RedirectResponse(url='/')
    if view not in VALID_VIEWS:
        view = 'title'

    section   = SECTIONS[sid1]
    cache_key = f"{sid1}:{sid2 or ''}"

    if sid2:
        # 소분류 필터: mode=LS2D 를 써야 sid2가 실제로 적용됨 (mode=LSD 는 sid1만 필터)
        base_url = (
            f'https://news.naver.com/main/list.naver'
            f'?mode=LS2D&mid=sec&sid1={sid1}&sid2={sid2}'
        )
    else:
        # 전체(소분류 없음): 대분류 URL 그대로 사용
        base_url = CATEGORY_BASE_URLS.get(sid1, CATEGORY_BASE_URLS['001'])

    cache = _get_section_cache(cache_key)
    now = time.time()
    is_stale = (not cache['articles']) or ((now - cache['fetched_at']) > CACHE_TTL)
    if is_stale and cache_key not in _section_crawling:
        asyncio.create_task(_run_section_crawl(cache_key, base_url))

    all_articles = cache['articles']
    loading      = len(all_articles) == 0
    enriching    = cache.get('enriching', False)
    page_articles, current_page, total_pages = _paginate(all_articles, page)

    nav_extra_params = f'&sid2={sid2}' if sid2 else ''

    return templates.TemplateResponse(
        request=request,
        name='naver_news_clone.html',
        context={
            'news_list':        page_articles,
            'all_articles':     all_articles,
            'today':            _today_str(),
            'loading':          loading,
            'enriching':        enriching,
            'view_type':        view,
            'current_page':     current_page,
            'total_pages':      total_pages,
            'section_id':       sid1,
            'section_name':     section['name'],
            'sub_categories':   section['sub_categories'],
            'active_sid2':      sid2,
            'nav_url_base':     f'/section/{sid1}',
            'nav_extra_params': nav_extra_params,
            'cache_key':        cache_key,
        },
    )


@app.get('/yonhap', response_class=HTMLResponse)
async def yonhap_page(
    request: Request,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
):
    """연합뉴스 속보 페이지."""
    if view not in VALID_VIEWS:
        view = 'title'

    cache_key = 'yonhap'
    cache = _get_section_cache(cache_key)
    now = time.time()
    is_stale = (not cache['articles']) or ((now - cache['fetched_at']) > CACHE_TTL)
    if is_stale and cache_key not in _section_crawling:
        asyncio.create_task(_run_section_crawl(cache_key, YONHAP_URL))

    all_articles = cache['articles']
    loading      = len(all_articles) == 0
    enriching    = cache.get('enriching', False)
    page_articles, current_page, total_pages = _paginate(all_articles, page)

    return templates.TemplateResponse(
        request=request,
        name='naver_news_clone.html',
        context={
            'news_list':        page_articles,
            'all_articles':     all_articles,
            'today':            _today_str(),
            'loading':          loading,
            'enriching':        enriching,
            'view_type':        view,
            'current_page':     current_page,
            'total_pages':      total_pages,
            'section_id':       'yonhap',
            'section_name':     '연합뉴스',
            'sub_categories':   [],
            'active_sid2':      None,
            'nav_url_base':     '/yonhap',
            'nav_extra_params': '',
            'cache_key':        'yonhap',
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/api/news', response_class=JSONResponse)
async def api_news():
    return _cache['articles']


@app.get('/api/status', response_class=JSONResponse)
async def api_status():
    total = len(_cache['articles'])
    ready = len([a for a in _cache['articles'] if a.get('body_ready')])
    return {
        'loading':    total == 0,
        'enriching':  _cache.get('enriching', False),
        'total':      total,
        'body_ready': ready,
    }


@app.get('/api/section/status/{cache_key:path}', response_class=JSONResponse)
async def api_section_status(cache_key: str):
    cache = _get_section_cache(cache_key)
    total = len(cache['articles'])
    ready = len([a for a in cache['articles'] if a.get('body_ready')])
    return {
        'loading':    total == 0,
        'enriching':  cache.get('enriching', False),
        'total':      total,
        'body_ready': ready,
    }


if __name__ == '__main__':
    print('=' * 50)
    print(' 속보 - AI 가짜뉴스 탐지 시스템')
    print(' http://localhost:8000')
    print(' API 문서: http://localhost:8000/docs')
    print('=' * 50)
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)
