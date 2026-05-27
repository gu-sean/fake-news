"""
네이버 뉴스 클론 — AI 가짜뉴스 탐지 속보 페이지
접속: http://localhost:8000
"""

import sys
import os
import time
import asyncio
import json
import ssl
import urllib.request
from datetime import datetime
from bs4 import BeautifulSoup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
import re
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.crawler import fetch_listing_pages, enrich_bodies, CATEGORY_BASE_URLS
from src.detector import get_sorted_timeline

# ── CSV 결과 데이터 로더 ──────────────────────────────────────────────────────
_RESULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'realtime_news_result.csv')

def _load_result_articles() -> list[dict]:
    try:
        df = pd.read_csv(_RESULT_CSV_PATH, encoding='utf-8-sig')
    except FileNotFoundError:
        return []

    records = []
    for _, row in df.iterrows():
        prob  = float(row.get('fake_probability(%)', 0) or 0)
        label = int(row.get('predicted_label', 0) or 0)
        img   = str(row.get('image_url', ''))
        records.append({
            'title':           str(row.get('title', '')),
            'url':             str(row.get('url', '#')),
            'press':           str(row.get('media', '')),
            'time':            str(row.get('pub_time', row.get('date', ''))),
            'image_url':       img if img else None,
            'summary':         str(row.get('summary', '')) if pd.notna(row.get('summary')) else '',
            'fake_score':      prob,         
            'body_ready':      True,
            # ── 결과 페이지 전용 ─────────────────────────────────────
            'predicted_label': label,         # 0=진짜, 1=가짜  ← 구분·정렬 기준
        })

    # predicted_label 오름차순(0→1), 동점이면 fake_score 오름차순
    records.sort(key=lambda x: (x['predicted_label'], x['fake_score']))
    return records


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
    'aside_html': '',      # 실제 네이버 aside HTML (광고 제거 후)
    'aside_at':   0,       # aside 마지막 갱신 시각
}
CACHE_TTL       = 300   # 5분
PAGE_SIZE       = 30    # 메인/섹션 페이지당 기사 수
VIEW_PAGE_SIZE  = 10    # 우측 패널(/view, /rview) 페이지당 기사 수
VIEW_PAGE_GROUP = 8     # 페이지 네비게이션 한 번에 표시할 페이지 수
CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cache')
_crawl_in_progress = False

# ── 섹션별 캐시 ───────────────────────────────────────────────────────────
_section_caches: dict[str, dict] = {}
_section_crawling: set[str] = set()
CRAWL_SEMAPHORE = asyncio.Semaphore(4)   

# ── 서버 시작 시 사전 크롤링할 섹션 목록 ────────────────────────────────────
# 1) 메인 카테고리 (정치·경제·사회·생활/문화·세계·IT/과학·오피니언)
_main_sections: list[tuple[str, str]] = [
    (f"{sid1}:", CATEGORY_BASE_URLS[sid1])
    for sid1 in SECTIONS
    if sid1 in CATEGORY_BASE_URLS
]

# 2) 세부 카테고리 (sid2 가 있는 항목만, 41개)
#    cache_key = "{sid1}:{sid2}",  base_url = mode=LS2D
_sub_sections: list[tuple[str, str]] = [
    (
        f"{sid1}:{sub['sid2']}",
        f"https://news.naver.com/main/list.naver"
        f"?mode=LS2D&mid=sec&sid1={sid1}&sid2={sub['sid2']}",
    )
    for sid1, sec in SECTIONS.items()
    for sub in sec['sub_categories']
    if sub['sid2'] is not None
]

# 3) 연합뉴스 속보
PREFETCH_SECTIONS: list[tuple[str, str]] = (
    _main_sections + _sub_sections + [('yonhap', YONHAP_URL)]
)


def _get_section_cache(cache_key: str) -> dict:
    if cache_key not in _section_caches:
        _section_caches[cache_key] = {'articles': [], 'fetched_at': 0, 'enriching': False}
    return _section_caches[cache_key]


# ── 방안 4: 디스크 캐시 영속화 ───────────────────────────────────────────────

def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _save_cache_to_disk(cache_key: str, articles: list):
    """기사 목록을 JSON 파일로 저장 — 서버 재시작 시 즉시 화면 표시 가능"""
    try:
        _ensure_cache_dir()
        safe_key = cache_key.replace(':', '__').replace('/', '__')
        path = os.path.join(CACHE_DIR, f'{safe_key}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'cache_key': cache_key, 'articles': articles},
                      f, ensure_ascii=False, default=str)
        print(f'[cache] 저장: {safe_key}.json ({len(articles)}건)')
    except Exception as e:
        print(f'[cache] 저장 실패 ({cache_key}): {e}')


def _load_all_disk_caches():
    """서버 시작 시 저장된 JSON 캐시를 메모리에 선 로드"""
    _ensure_cache_dir()
    loaded = 0
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cache_key = data.get('cache_key', '')
            articles  = data.get('articles', [])
            if not cache_key or not articles:
                continue
            mtime = os.path.getmtime(path)
            if cache_key == 'main':
                _cache['articles']   = articles
                _cache['fetched_at'] = mtime
                print(f'[cache] 메인 캐시 로드: {len(articles)}건')
            else:
                sc = _get_section_cache(cache_key)
                sc['articles']   = articles
                sc['fetched_at'] = mtime
                print(f'[cache] [{cache_key}] 캐시 로드: {len(articles)}건')
            loaded += 1
        except Exception as e:
            print(f'[cache] 로드 실패 ({fname}): {e}')
    print(f'[cache] 총 {loaded}개 캐시 파일 로드 완료')


# ── 공통 크롤링 로직  ──────────────────────────

async def _run_crawl():
    """메인 속보 비동기 크롤링 (세마포어 제한 적용)"""
    global _crawl_in_progress
    if _crawl_in_progress:
        return
    _crawl_in_progress = True
    raw = []

    # ── Phase 1: 목록 수집만 세마포어 내부에서  ─────────────────
    async with CRAWL_SEMAPHORE:
        try:
            print('[app] Phase 1 시작: 목록 페이지 수집...')
            raw = await fetch_listing_pages(max_pages=3)   
            articles = get_sorted_timeline(raw)
            # 진짜(fake_score<50) 상단, 가짜(fake_score≥50) 하단 정렬
            articles.sort(key=lambda x: (0 if (x.get('fake_score') or 0) < 50 else 1,
                                         x.get('fake_score') or 0))
            _cache['articles']   = articles
            _cache['fetched_at'] = time.time()
            _cache['enriching']  = True
            print(f'[app] Phase 1 완료: {len(articles)}건 → 즉시 화면 표시 가능')
        except Exception as e:
            print(f'[app] Phase 1 실패: {e}')
            _cache['enriching'] = False
            _crawl_in_progress = False
            return

    # ── Phase 2: 세마포어 반납 후 본문 수집  ─────────────────
    try:
        print('[app] Phase 2 시작: 기사 본문 백그라운드 수집...')
        await enrich_bodies(raw)
        _cache['fetched_at'] = time.time()
        _cache['enriching']  = False
        _save_cache_to_disk('main', _cache['articles']) 
        print('[app] Phase 2 완료')
    except Exception as e:
        print(f'[app] Phase 2 실패: {e}')
        _cache['enriching'] = False
    finally:
        _crawl_in_progress = False


async def _run_section_crawl(cache_key: str, base_url: str):
    """카테고리 섹션 비동기 크롤링 """
    if cache_key in _section_crawling:
        return
    _section_crawling.add(cache_key)
    cache = _get_section_cache(cache_key)

    # 세부 카테고리(sid2 존재)는 1페이지, 메인 카테고리는 2페이지
    # cache_key 형식: "100:" (메인) vs "100:269" (세부)
    is_sub = cache_key != 'yonhap' and cache_key.split(':')[-1] != ''
    max_pages = 1 if is_sub else 2

    # ------------------------------------------------------------------
    # Phase 1: 목록 수집만 세마포어 내부에서 빠르게 실행하고 빠져나옵니다.
    # ------------------------------------------------------------------
    async with CRAWL_SEMAPHORE:
        try:
            print(f'[app] [{cache_key}] Phase 1 시작 (세마포어 획득, max_pages={max_pages})...')
            raw = await fetch_listing_pages(base_url=base_url, max_pages=max_pages)

            articles = get_sorted_timeline(raw)
            # 진짜(fake_score<50) 상단, 가짜(fake_score≥50) 하단 정렬
            articles.sort(key=lambda x: (0 if (x.get('fake_score') or 0) < 50 else 1,
                                         x.get('fake_score') or 0))
            cache['articles']   = articles
            cache['fetched_at'] = time.time()
            cache['enriching']  = True
            print(f'[app] [{cache_key}] Phase 1 완료: {len(articles)}건 (세마포어 반납)')
            
        except Exception as e:
            print(f'[app] [{cache_key}] Phase 1 실패: {e}')
            _section_crawling.discard(cache_key)
            return

    # 세마포어 반납 후 Phase 2 진행 — 다른 섹션의 Phase 1이 곧바로 슬롯을 획득할 수 있음
    try:
        print(f'[app] [{cache_key}] Phase 2 시작 (백그라운드 본문 수집)...')
        await enrich_bodies(raw)  
        cache['fetched_at'] = time.time()
        cache['enriching']  = False
        _save_cache_to_disk(cache_key, cache['articles'])   
        print(f'[app] [{cache_key}] Phase 2 완료')
    except Exception as e:
        print(f'[app] [{cache_key}] Phase 2 실패: {e}')
        cache['enriching'] = False
    finally:
        _section_crawling.discard(cache_key)


async def _staggered_warmup():
    """서버 시작 시 모든 섹션(메인 7 + 세부 41 + 연합 1 = 49개)을
    0.2초 간격으로 순차 등록해 백그라운드 크롤링을 시작합니다."""
    for cache_key, base_url in PREFETCH_SECTIONS:
        if cache_key not in _section_crawling:
            print(f'[app] 사전 워밍 시작: [{cache_key}]')
            asyncio.create_task(_run_section_crawl(cache_key, base_url))
        await asyncio.sleep(0.2)  


async def _refresh_loop():
    """CACHE_TTL마다 모든 캐시를 자동 갱신"""
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
                await asyncio.sleep(0.5) 


# ── 서버 라이프사이클 관리 ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[app] 서버 시작 — 디스크 캐시 선 로드 중...')
    _load_all_disk_caches()                            
    print('[app] 속보(메인) + 전체 섹션 사전 크롤링 시작...')
    asyncio.create_task(_run_crawl())
    warmup_task  = asyncio.create_task(_staggered_warmup())
    refresh_task = asyncio.create_task(_refresh_loop())
    yield
    warmup_task.cancel()
    refresh_task.cancel()


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
    if sid1 not in SECTIONS:
        return RedirectResponse(url='/')
    if view not in VALID_VIEWS:
        view = 'title'

    section   = SECTIONS[sid1]
    cache_key = f"{sid1}:{sid2 or ''}"

    if sid2:
        base_url = (
            f'https://news.naver.com/main/list.naver'
            f'?mode=LS2D&mid=sec&sid1={sid1}&sid2={sid2}'
        )
    else:
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
# 라우트 — CSV 탐지 결과 페이지
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/result', response_class=HTMLResponse)
async def result_page(
    request: Request,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
):
    if view not in VALID_VIEWS:
        view = 'title'

    # predicted_label 오름차순 정렬된 전체 목록 (0=진짜 먼저, 1=가짜 나중)
    all_articles = _load_result_articles()
    real_count   = sum(1 for a in all_articles if a['predicted_label'] == 0)
    fake_count   = sum(1 for a in all_articles if a['predicted_label'] == 1)

    page_articles, current_page, total_pages = _paginate(all_articles, page)

    return templates.TemplateResponse(
        request=request,
        name='naver_news_clone.html',
        context={
            'news_list':       page_articles,
            'all_articles':    all_articles,
            'today':           _today_str(),
            'loading':         False,
            'enriching':       False,
            'view_type':       view,
            'current_page':    current_page,
            'total_pages':     total_pages,
            'section_id':      None,
            'section_name':    None,
            'sub_categories':  [],
            'active_sid2':     None,
            'nav_url_base':    '/result',
            'nav_extra_params':'',
            'cache_key':       'result',
            # 결과 페이지 전용
            'real_count':      real_count,
            'fake_count':      fake_count,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 라우트 — 우측 패널 전용 기사 목록 뷰 (스플릿 화면 재귀 방지)
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/view', response_class=HTMLResponse)
async def view_page(
    request: Request,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
):
    """우측 패널 전용: AI 탐지 결과 기사 목록 (네이버 클론 CSS 적용)."""
    if view not in VALID_VIEWS:
        view = 'title'

    all_articles = _load_result_articles()
    real_count   = sum(1 for a in all_articles if a['predicted_label'] == 0)
    fake_count   = sum(1 for a in all_articles if a['predicted_label'] == 1)

    total_pages  = max(1, (len(all_articles) + VIEW_PAGE_SIZE - 1) // VIEW_PAGE_SIZE)
    current_page = min(page, total_pages)
    start        = (current_page - 1) * VIEW_PAGE_SIZE
    page_articles = all_articles[start: start + VIEW_PAGE_SIZE]

    return templates.TemplateResponse(
        request=request,
        name='naver_news_right.html',
        context={
            'news_list':        page_articles,
            'all_articles':     all_articles,
            'show_badges':      True,
            'real_count':       real_count,
            'fake_count':       fake_count,
            'total_count':      len(all_articles),
            'today':            _today_str(),
            'section_name':     '전체',
            'section_id':       None,
            'sub_categories':   [],
            'active_sid2':      None,
            'enriching':        False,
            'cache_key':        'result',
            'view_type':        view,
            'current_page':     current_page,
            'total_pages':      total_pages,
            'nav_url_base':     '/view',
            'nav_extra_params': '',
        },
    )


@app.get('/rview/section/{sid1}', response_class=HTMLResponse)
async def rview_section(
    request: Request,
    sid1: str,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
    sid2: str = Query(default=None),
):
    """우측 패널 전용: 카테고리 섹션 기사 목록 (AI 배지 없음, 재귀 방지)."""
    if sid1 not in SECTIONS and sid1 != 'yonhap':
        return RedirectResponse(url='/view')
    if view not in VALID_VIEWS:
        view = 'title'

    cache_key = f"{sid1}:{sid2 or ''}" if sid1 != 'yonhap' else 'yonhap'
    cache     = _get_section_cache(cache_key)

    now      = time.time()
    is_stale = (not cache['articles']) or ((now - cache['fetched_at']) > CACHE_TTL)
    if is_stale and cache_key not in _section_crawling:
        base_url = (
            f'https://news.naver.com/main/list.naver?mode=LS2D'
            f'&mid=sec&sid1={sid1}&sid2={sid2}'
            if sid2 else CATEGORY_BASE_URLS.get(sid1, CATEGORY_BASE_URLS['001'])
        )
        asyncio.create_task(_run_section_crawl(cache_key, base_url))

    all_articles = cache['articles']
    section_info = SECTIONS.get(sid1, {})
    section_name = section_info.get('name', '연합뉴스') if sid1 != 'yonhap' else '연합뉴스'

    # AI 배지 없이 표시하기 위해 predicted_label 기본값 세팅
    for a in all_articles:
        if 'predicted_label' not in a:
            a['predicted_label'] = 0
        if 'fake_score' not in a:
            a['fake_score'] = 0.0

    total_pages  = max(1, (len(all_articles) + VIEW_PAGE_SIZE - 1) // VIEW_PAGE_SIZE)
    current_page = min(page, total_pages)
    start        = (current_page - 1) * VIEW_PAGE_SIZE
    page_articles = all_articles[start: start + VIEW_PAGE_SIZE]

    nav_extra_params = f'&sid2={sid2}' if sid2 else ''

    return templates.TemplateResponse(
        request=request,
        name='naver_news_right.html',
        context={
            'news_list':        page_articles,
            'all_articles':     all_articles,
            'show_badges':      False,
            'real_count':       0,
            'fake_count':       0,
            'total_count':      len(all_articles),
            'today':            _today_str(),
            'section_name':     section_name,
            'section_id':       sid1,
            'sub_categories':   section_info.get('sub_categories', []),
            'active_sid2':      sid2,
            'enriching':        cache.get('enriching', False),
            'cache_key':        cache_key,
            'loading':          len(all_articles) == 0,
            'view_type':        view,
            'current_page':     current_page,
            'total_pages':      total_pages,
            'nav_url_base':     f'/rview/section/{sid1}',
            'nav_extra_params': nav_extra_params,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 라우트 — 네이버 뉴스 서버 사이드 프록시 (iframe X-Frame-Options 우회)
# ─────────────────────────────────────────────────────────────────────────────

_NAVER_LIST_URL = 'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=001'
_PROXY_HEADERS = {
    'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36',
    'Referer':         'https://news.naver.com/',
    'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def _fetch_naver_raw(url: str) -> str:
    """urllib로 실제 네이버 뉴스 HTML 동기 다운로드 (Windows DNS 호환)."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=_PROXY_HEADERS)
    with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
        charset = 'utf-8'
        ct = resp.headers.get('Content-Type', '')
        if 'euc-kr' in ct.lower():
            charset = 'euc-kr'
        return resp.read().decode(charset, errors='replace')


_PROXY_INTERCEPT_JS = '''\
<script>
(function(){
  document.addEventListener('click', function(e){
    var a = e.target;
    while (a && a.tagName !== 'A') a = a.parentElement;
    if (!a || !a.href) return;
    var href = a.href;
    if (href.indexOf('javascript:') === 0 || href.indexOf('mailto:') === 0 || href.indexOf('#') === 0) return;
    if (a.getAttribute('target') === '_blank') return;
    if (href.indexOf('naver.com') !== -1 && href.indexOf('/proxy/page') === -1) {
      e.preventDefault();
      e.stopPropagation();
      window.location.href = '/proxy/page?url=' + encodeURIComponent(href);
    }
  }, true);
})();
</script>'''


def _fix_naver_html(html: str) -> str:
    """프록시된 HTML의 URL을 절대경로로 수정, 클릭 인터셉터 주입."""
    # 1) 프로토콜 생략 URL(//...) → https://
    html = re.sub(r'((?:src|href|action|content)=["\'])//([^"\']+)',
                  lambda m: m.group(1) + 'https://' + m.group(2), html)
    html = re.sub(r"((?:src|href|action|content)=')//([^']+)",
                  lambda m: m.group(1) + 'https://' + m.group(2), html)
    # 2) <head> 직후에 base href 삽입 → 상대경로 자동 해결
    html = re.sub(r'(<head[^>]*>)',
                  r'\1\n<base href="https://news.naver.com/" target="_self">',
                  html, count=1, flags=re.IGNORECASE)
    # 3) X-Frame-Options / CSP 메타태그 제거 (HTTP 헤더 차단 우회)
    html = re.sub(r'<meta[^>]+(?:x-frame-options|content-security-policy)[^>]*>',
                  '', html, flags=re.IGNORECASE)
    # 4) 클릭 인터셉터 삽입: Naver 링크 → /proxy/page?url= 경유 (기사 페이지 iframe 차단 방지)
    if re.search(r'</body>', html, re.IGNORECASE):
        html = re.sub(r'(</body>)', _PROXY_INTERCEPT_JS + r'\n\1', html,
                      count=1, flags=re.IGNORECASE)
    else:
        html += _PROXY_INTERCEPT_JS
    return html


# ── 네이버 실제 aside 추출 ────────────────────────────────────────────────────

# 제거할 광고 선택자 목록 (네이버 광고 패턴)
_AD_SELECTORS = [
    '.da_wrap', '.NE_B_ad', '.aside_ad',
    '[class*="da_"]', '[id*="da_"]',
    '[class*="NE_B_ad"]',
    'iframe',
    '.ad', '.advertisement',
    '._au_ad', '[class*="_ad_"]',
    '.partner_news_wrap',         
    '.promotion_area',
    '.ad_area',
    # ── 실제 네이버 사이드 광고 (사용자 제공 HTML에서 확인된 패턴) ──
    '.side_ad',                  
    '[id*="glad"]',             
    '[class*="glad"]',            
    '[id*="ad_sidebox"]',      
    '[id*="aside_ad"]',            
    '.banner_ad',                  
    '[class*="banner_ad"]',       
    'script[src*="glad"]',         
    '.naver_image_side_banner',    
]


def _extract_naver_aside_sync(raw_html: str) -> str:
    """BeautifulSoup으로 네이버 HTML에서 td.aside 내용 추출, 광고 제거.
    결과는 <td> 태그 없이 내부 HTML만 반환 (템플릿 td.aside 안에 삽입용)."""
    try:
        soup = BeautifulSoup(raw_html, 'lxml')

        aside_td = soup.select_one('td.aside')
        if not aside_td:
            aside_td = soup.select_one('#aside') or soup.find('aside')
        if not aside_td:
            return ''

        for sel in _AD_SELECTORS:
            for el in aside_td.select(sel):
                el.decompose()

        for el in aside_td.find_all('script'):
            el.decompose()


        for tag in aside_td.find_all(True):
            for attr in ('src', 'href', 'data-src'):
                val = tag.get(attr, '')
                if val.startswith('//'):
                    tag[attr] = 'https:' + val
                elif val.startswith('/') and not val.startswith('//'):
                    tag[attr] = 'https://news.naver.com' + val

        # 모든 링크는 새 탭으로 (iframe 안이므로)
        for a in aside_td.find_all('a', href=True):
            a['target'] = '_blank'

        return aside_td.decode_contents()
    except Exception as e:
        print(f'[aside] 추출 실패: {e}')
        return ''


@app.get('/proxy/naver/aside', response_class=HTMLResponse)
async def proxy_naver_aside():
    """우측 패널 aside용: 실제 네이버 aside HTML 조각 (광고 제거)."""
    # 캐시가 있고 5분 이내면 재사용
    if _cache['aside_html'] and (time.time() - _cache['aside_at']) < CACHE_TTL:
        return HTMLResponse(content=_cache['aside_html'])

    # 없거나 만료됐으면 새로 fetch
    try:
        raw  = await asyncio.to_thread(_fetch_naver_raw, _NAVER_LIST_URL)
        html = await asyncio.to_thread(_extract_naver_aside_sync, raw)
        if html:
            _cache['aside_html'] = html
            _cache['aside_at']   = time.time()
        return HTMLResponse(content=html or '')
    except Exception as e:
        print(f'[proxy/aside] 실패: {e}')
        return HTMLResponse(content='')


@app.get('/proxy/naver', response_class=HTMLResponse)
async def proxy_naver(request: Request):
    """왼쪽 패널: 실제 네이버 뉴스 페이지를 서버 프록시로 렌더링.
    동시에 aside HTML을 추출해 캐시에 저장한다."""
    try:
        raw  = await asyncio.to_thread(_fetch_naver_raw, _NAVER_LIST_URL)
        # aside 캐시가 비어 있으면 동시에 추출
        if not _cache['aside_html']:
            aside = await asyncio.to_thread(_extract_naver_aside_sync, raw)
            if aside:
                _cache['aside_html'] = aside
                _cache['aside_at']   = time.time()
        html = _fix_naver_html(raw)
        return HTMLResponse(content=html)
    except Exception as e:
        print(f'[proxy] 네이버 직접 fetch 실패 ({e}) — 크롤러 캐시 fallback')
        articles = _cache['articles']
        return templates.TemplateResponse(
            request=request,
            name='naver_news_left.html',
            context={
                'news_list': articles[:30],
                'loading':   len(articles) == 0,
                'today':     _today_str(),
            },
        )


@app.get('/proxy/page', response_class=HTMLResponse)
async def proxy_page(url: str = Query(...)):
    """왼쪽 패널: 개별 기사 페이지 프록시 — X-Frame-Options 우회."""
    import urllib.parse as _up
    parsed = _up.urlparse(url)
    # 보안: naver.com 도메인만 허용
    if 'naver.com' not in (parsed.netloc or ''):
        return HTMLResponse(
            content='<html><body style="font-family:sans-serif;padding:40px">'
                    '<p>허용되지 않는 주소입니다.</p>'
                    '<p><a href="javascript:history.back()">← 돌아가기</a></p>'
                    '</body></html>',
            status_code=403,
        )
    try:
        raw  = await asyncio.to_thread(_fetch_naver_raw, url)
        html = _fix_naver_html(raw)
        return HTMLResponse(content=html)
    except Exception as e:
        print(f'[proxy/page] 실패: {url[:100]} → {e}')
        return HTMLResponse(
            content=f'<html><body style="font-family:sans-serif;padding:40px">'
                    f'<p>페이지를 불러올 수 없습니다.</p>'
                    f'<p style="font-size:12px;color:#999">{e}</p>'
                    f'<p><a href="javascript:history.back()">← 돌아가기</a></p>'
                    f'</body></html>',
        )


# ── API 엔드포인트 ───────────────────────────────────────────────────────────

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