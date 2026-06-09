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
from src.detector import (get_sorted_timeline, predict_articles_inplace,
                          predict_logistic_inplace, predict_svm_inplace,
                          predict_ensemble_inplace,
                          _load_xgb, _load_logistic, _load_svm, _load_nb)

# ── CSV 결과 데이터 로더 ──────────────────────────────────────────────────────
_RESULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'realtime_news_result.csv')
_result_cache: dict = {'articles': [], 'mtime': 0.0}

# ── 학습 데이터 가짜뉴스 샘플 로더 ──────────────────────────────────────────────
# 팩트체크 크롤링 데이터만 사용 (AI Hub 데이터 미사용)
_FACTCHECK_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'factcheck', 'factcheck_label.csv')
_fake_sample_cache: list = []

def _load_fake_sample_articles(limit: int = 60) -> list[dict]:
    """팩트체크 크롤링 가짜뉴스 샘플 로드.
    data/factcheck/factcheck_label.csv 없으면 빈 목록 반환.
    서버 시작 시 1회만 로드 후 캐시 재사용."""
    global _fake_sample_cache
    if _fake_sample_cache:
        return _fake_sample_cache

    if not os.path.exists(_FACTCHECK_CSV_PATH):
        print('[fake_csv] 팩트체크 데이터 없음 (python src/factcheck_crawler.py 실행 필요)')
        return []

    try:
        df = pd.read_csv(_FACTCHECK_CSV_PATH, encoding='utf-8-sig')
        fake_df = df[df['label'] == 1].copy()
        if fake_df.empty:
            print('[fake_csv] 팩트체크 데이터: label=1 기사 없음')
            return []

        sample = fake_df.sample(n=min(limit, len(fake_df)), random_state=42)
        records = []
        for _, row in sample.iterrows():
            content = str(row.get('content', '') or '')
            article_url = str(row.get('url', '') or '')
            if not article_url or article_url in ('nan', 'None'):
                article_url = '#'
            verdict_raw = row.get('verdict', '')
            verdict = '' if (verdict_raw is None or str(verdict_raw).strip() in ('', 'nan', 'None')) else str(verdict_raw).strip()
            press_raw = row.get('media', '')
            press = str(press_raw) if (press_raw is not None and str(press_raw).strip() not in ('', 'nan', 'None')) else 'KBS팩트체크K'
            records.append({
                'title':                str(row.get('title', '')),
                'url':                  article_url,
                'press':                press,
                'time':                 str(row.get('date', '') or ''),
                'image_url':            None,
                'summary':              (f"[{verdict}] " if verdict else '') + content[:200],
                'text':                 content,
                'body_ready':           True,
                'source':               '팩트체크',
                'stat_distortion':      int(row.get('stat_distortion', 0) or 0),
                'causal_error':         int(row.get('causal_error', 0) or 0),
                'emotional_provocation':int(row.get('emotional_provocation', 0) or 0),
                'source_lack':          int(row.get('source_lack', 0) or 0),
                'img_mismatch':         int(row.get('img_mismatch', 0) or 0),
            })

        # XGBoost로 실제 예측 (고정값 95.0 대신)
        predict_articles_inplace(records, lang='auto')

        # 예측 실패한 기사는 기본값 설정
        for r in records:
            if 'fake_score' not in r:
                r['fake_score']      = 0.0
                r['predicted_label'] = 0

        _fake_sample_cache = records
        print(f'[fake_csv] 팩트체크 가짜뉴스 샘플 {len(records)}건 로드 (XGBoost 예측 완료)')
        return records
    except Exception as e:
        print(f'[fake_csv] 로드 실패: {e}')
        return []


def _write_result_csv(articles: list[dict]) -> None:
    """크롤링 기사를 realtime_news_result.csv 형식으로 저장.
    모델 파일이 없을 때 기본값(predicted_label=0)으로 채워 오른쪽 패널을 자동 갱신."""
    if not articles:
        return
    try:
        rows = []
        for i, a in enumerate(articles, 1):
            fake_score = round(float(a.get('fake_score') or 0), 2)
            rows.append({
                'id':                    i,
                'title':                 a.get('title', ''),
                'content':               a.get('text', a.get('summary', '')),
                'media':                 a.get('press', ''),
                'date':                  datetime.now().strftime('%Y-%m-%d'),
                'label':                 None,
                'clean_message':         '',
                'stat_distortion':       a.get('stat_distortion', 0),
                'causal_error':          a.get('causal_error', 0),
                'emotional_provocation': a.get('emotional_provocation', 0),
                'source_lack':           a.get('source_lack', 0),
                'img_mismatch':          a.get('img_mismatch', 0),
                'url':                   a.get('url', ''),
                'category':              a.get('category', '속보'),
                'pub_time':              a.get('time', ''),
                'image_url':             a.get('image_url', ''),
                'summary':               a.get('summary', ''),
                'fake_score':            fake_score,
                'body_ready':            a.get('body_ready', False),
                'predicted_label':       a.get('predicted_label', 1 if fake_score >= 50 else 0),
                'fake_probability(%)':   fake_score,
                '결과_텍스트':            '🚨 가짜뉴스' if a.get('predicted_label', fake_score >= 50) else '✅ 진짜뉴스',
            })
        pd.DataFrame(rows).to_csv(_RESULT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f'[result] realtime_news_result.csv 갱신: {len(rows)}건')
    except Exception as e:
        print(f'[result] realtime_news_result.csv 저장 실패: {e}')


def _load_result_articles() -> list[dict]:
    try:
        mtime = os.path.getmtime(_RESULT_CSV_PATH)
    except FileNotFoundError:
        return []

    if mtime == _result_cache['mtime']:
        return _result_cache['articles']

    try:
        df = pd.read_csv(_RESULT_CSV_PATH, encoding='utf-8-sig')
    except Exception:
        return _result_cache['articles']

    records = []
    for _, row in df.iterrows():
        prob  = float(row.get('fake_probability(%)', 0) or 0)
        label = int(row.get('predicted_label', 0) or 0)
        img   = str(row.get('image_url', ''))
        records.append({
            'title':                str(row.get('title', '')),
            'url':                  str(row.get('url', '#')),
            'press':                str(row.get('media', '')),
            'time':                 str(row.get('pub_time', row.get('date', ''))),
            'image_url':            img if img else None,
            'summary':              str(row.get('summary', '')) if pd.notna(row.get('summary')) else '',
            'fake_score':           prob,
            'body_ready':           True,
            'predicted_label':      label,
            'stat_distortion':      int(row.get('stat_distortion', 0) or 0),
            'causal_error':         int(row.get('causal_error', 0) or 0),
            'emotional_provocation':int(row.get('emotional_provocation', 0) or 0),
            'source_lack':          int(row.get('source_lack', 0) or 0),
            'img_mismatch':         int(row.get('img_mismatch', 0) or 0),
        })

    # 제목 기준 중복 제거 (같은 제목 다른 URL 기사 - 크롤러가 복수 수집한 경우)
    seen_titles: set = set()
    deduped: list = []
    for r in records:
        t = r['title'].strip()
        if t not in seen_titles:
            seen_titles.add(t)
            deduped.append(r)
    records = deduped

    records.sort(key=lambda x: (x['predicted_label'], x['fake_score']))
    _result_cache['articles'] = records
    _result_cache['mtime']    = mtime
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
    'proxy_html': '',      # /proxy/naver 전체 HTML 캐시
    'proxy_at':   0,       # proxy_html 마지막 갱신 시각
}
CACHE_TTL        = 300    # 5분 — 기사 데이터 캐시
_LVIEW_CACHE_TTL = 1800   # 30분 — lview/proxy HTML 캐시 (변경 빈도 낮음)
PAGE_SIZE        = 30     # 메인/섹션 페이지당 기사 수
VIEW_PAGE_SIZE   = 30     # 우측 패널(/view, /rview) 페이지당 기사 수
VIEW_PAGE_GROUP  = 8      # 페이지 네비게이션 한 번에 표시할 페이지 수
CACHE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cache')
_PROXY_HTML_PATH = os.path.join(CACHE_DIR, 'proxy_naver.html')
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

    # proxy_naver.html 디스크 캐시 로드 (왼쪽 패널 첫 로딩 즉시화)
    if os.path.exists(_PROXY_HTML_PATH):
        try:
            with open(_PROXY_HTML_PATH, 'r', encoding='utf-8') as f:
                _cache['proxy_html'] = f.read()
            _cache['proxy_at'] = os.path.getmtime(_PROXY_HTML_PATH)
            print('[cache] proxy_naver.html 디스크 캐시 로드 완료')
        except Exception as e:
            print(f'[cache] proxy_naver.html 로드 실패: {e}')


def _save_proxy_html_to_disk(html: str) -> None:
    """proxy_html을 디스크에 저장 — 서버 재시작 시 즉시 서빙 가능."""
    try:
        _ensure_cache_dir()
        with open(_PROXY_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(html)
        print('[cache] proxy_naver.html 디스크 저장 완료')
    except Exception as e:
        print(f'[cache] proxy_naver.html 저장 실패: {e}')


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
            raw = await fetch_listing_pages(max_pages=20)
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
        # 본문이 추가된 상태로 XGBoost 재예측 (정확도 향상)
        predict_articles_inplace(raw, lang='korean')
        _cache['fetched_at'] = time.time()
        _cache['enriching']  = False
        _save_cache_to_disk('main', _cache['articles'])
        _write_result_csv(_cache['articles'])
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

    # 세부 카테고리(sid2 존재)는 2페이지, 메인 카테고리는 8페이지
    # cache_key 형식: "100:" (메인) vs "100:269" (세부)
    is_sub = cache_key != 'yonhap' and cache_key.split(':')[-1] != ''
    max_pages = 2 if is_sub else 8

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
    0.2초 간격으로 순차 등록해 백그라운드 크롤링을 시작"""
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
        print('[app] 자동 갱신 주기 도래 - 스테일 캐시 갱신 시작')

        # 메인(속보) 캐시
        if (now - _cache['fetched_at']) > CACHE_TTL and not _crawl_in_progress:
            asyncio.create_task(_run_crawl())

        # 섹션 캐시 (기사 목록)
        for cache_key, base_url in PREFETCH_SECTIONS:
            sec_cache = _get_section_cache(cache_key)
            if (now - sec_cache['fetched_at']) > CACHE_TTL and cache_key not in _section_crawling:
                asyncio.create_task(_run_section_crawl(cache_key, base_url))
                await asyncio.sleep(0.5)

        # lview 프록시 HTML 캐시 — 만료된 섹션을 동시 갱신
        stale = [
            (sid, url) for sid, url in _SIDEBAR_SECTIONS
            if url and (now - _lview_proxy_cache.get(f'{sid}:', {}).get('at', 0)) > _LVIEW_CACHE_TTL
        ]
        if stale:
            asyncio.create_task(asyncio.gather(*[
                _cache_lview_one(sid, url, force=True) for sid, url in stale
            ]))


async def _warmup_proxy_naver():
    """서버 시작 시 /proxy/naver HTML을 백그라운드에서 미리 캐싱."""
    try:
        # aside 추출은 원본 URL(listType 없음)로, HTML 캐싱은 title 뷰로
        raw_base = await _fetch_naver_raw(_NAVER_LIST_URL)
        if not _cache['aside_html']:
            aside = await asyncio.to_thread(_extract_naver_aside_sync, raw_base)
            if aside:
                _cache['aside_html'] = aside
                _cache['aside_at']   = time.time()
        raw  = await _fetch_naver_raw(_NAVER_LIST_URL_TITLE)
        html = await asyncio.to_thread(_fix_naver_html, raw)   # 스레드풀
        _cache['proxy_html'] = html
        _cache['proxy_at']   = time.time()
        _save_proxy_html_to_disk(html)                          # 디스크 저장
        print('[app] /proxy/naver 워밍업 완료')
    except Exception as e:
        print(f'[app] /proxy/naver 워밍업 실패: {e}')


_SIDEBAR_SECTIONS: list[tuple[str, str]] = []  # lifespan 후 채워짐


async def _cache_lview_one(sid: str, url: str, force: bool = False,
                           sid2: str = None):
    """단일 섹션 lview 프록시 HTML 을 fetch·가공해 캐시에 저장.
    sid2 지정 시 LS2D 세부 카테고리 URL로 fetch.
    항상 listType=title 로 fetch해 5개묶음 제목형 구조를 가져옴."""
    cache_key_str = f'{sid}:{sid2}:' if sid2 else f'{sid}:'
    if not force and cache_key_str in _lview_proxy_cache:
        return
    if sid2:
        title_url = (f'https://news.naver.com/main/list.naver'
                     f'?mode=LS2D&mid=sec&sid1={sid}&sid2={sid2}&listType=title')
    else:
        title_url = url + '&listType=title' if '?' in url else url + '?listType=title'
    try:
        raw  = await _fetch_naver_raw(title_url)
        html = await asyncio.to_thread(_fix_naver_html, raw, True)   # 스레드풀
        _lview_proxy_cache[cache_key_str] = {'html': html, 'at': time.time()}
        label = f'{sid}:{sid2}' if sid2 else sid
        print(f'[lview] [{label}] 캐싱 완료')
    except Exception as e:
        label = f'{sid}:{sid2}' if sid2 else sid
        print(f'[lview] [{label}] 캐싱 실패: {e}')


async def _warmup_lview_sections():
    """서버 시작 시 사이드바 섹션 + 세부 카테고리를 순차적으로 캐싱.
    메인 9개 먼저, 이후 세부 카테고리를 백그라운드 태스크로 순차 처리."""
    global _SIDEBAR_SECTIONS
    _SIDEBAR_SECTIONS = [
        ('100',    CATEGORY_BASE_URLS.get('100', '')),
        ('101',    CATEGORY_BASE_URLS.get('101', '')),
        ('102',    CATEGORY_BASE_URLS.get('102', '')),
        ('103',    CATEGORY_BASE_URLS.get('103', '')),
        ('104',    CATEGORY_BASE_URLS.get('104', '')),
        ('105',    CATEGORY_BASE_URLS.get('105', '')),
        ('110',    CATEGORY_BASE_URLS.get('110', '')),
        ('001',    CATEGORY_BASE_URLS.get('001', '')),
        ('yonhap', YONHAP_URL),
    ]
    # 1단계: 메인 카테고리 9개 동시 캐싱
    await asyncio.gather(*[
        _cache_lview_one(sid, url)
        for sid, url in _SIDEBAR_SECTIONS if url
    ])
    print('[lview] main sections cached, starting sub-category background cache')

    # 2단계: 세부 카테고리 순차 캐싱 (과부하 방지를 위해 0.5초 간격)
    async def _cache_sub_sections():
        for sid1, section in SECTIONS.items():
            for sub in section.get('sub_categories', []):
                sid2 = sub.get('sid2')
                if not sid2:
                    continue
                await _cache_lview_one(sid1, '', sid2=sid2)
                await asyncio.sleep(0.5)
        print('[lview] 세부 카테고리 전체 캐싱 완료')

    asyncio.create_task(_cache_sub_sections())
    print('[lview] 전체 사이드바 섹션 사전 캐싱 완료')


# ── 서버 라이프사이클 관리 ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[app] 서버 시작 - 디스크 캐시 선 로드 중...')
    _load_all_disk_caches()
    # 전체 모델 선로드 (첫 요청 지연 방지)
    await asyncio.to_thread(_load_xgb,      'korean')
    await asyncio.to_thread(_load_xgb,      'english')
    await asyncio.to_thread(_load_logistic, 'korean')
    await asyncio.to_thread(_load_svm,      'korean')
    await asyncio.to_thread(_load_nb,       'korean')
    print('[app] proxy/naver 워밍업 대기 중 (첫 진입 정상 구조 보장)...')
    await _warmup_proxy_naver()          # 완료 후 서버 오픈 → 첫 요청도 올바른 Naver HTML 서빙
    print('[app] 속보(메인) + 전체 섹션 사전 크롤링 시작...')
    asyncio.create_task(_run_crawl())
    asyncio.create_task(_warmup_lview_sections())
    asyncio.create_task(asyncio.to_thread(_load_fake_sample_articles))
    warmup_task  = asyncio.create_task(_staggered_warmup())
    refresh_task = asyncio.create_task(_refresh_loop())
    yield
    warmup_task.cancel()
    refresh_task.cancel()


app = FastAPI(title="속보 - AI 가짜뉴스 탐지", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── 공통 유틸 ─────────────────────────────────────────────────────────────
VALID_VIEWS = {'title', 'summary', 'photo', 'paper'}


def _today_str() -> str:
    now = datetime.now()
    dow = ['월', '화', '수', '목', '금', '토', '일'][now.weekday()]
    return f'{now.month:02d}.{now.day:02d}({dow})'


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
    flt:  str = Query(default='all', alias='filter'),
    model: str = Query(default=None),
):
    """우측 패널 전용: AI 탐지 결과 기사 목록 (네이버 클론 CSS 적용)."""
    _DVL_FILTERS = {
        'stat_distortion':      '통계 왜곡',
        'causal_error':         '인과 오류',
        'emotional_provocation':'감정 자극',
        'source_lack':          '출처 불명',
        'img_mismatch':         '이미지 불일치',
    }
    if view not in VALID_VIEWS:
        view = 'title'
    if flt not in ('all', 'real', 'fake') and flt not in _DVL_FILTERS:
        flt = 'all'

    # 실시간 기사(네이버) + 학습 데이터 가짜뉴스 샘플 합산
    realtime_articles = _load_result_articles()
    fake_samples      = _load_fake_sample_articles(limit=60)
    all_articles      = realtime_articles + fake_samples

    # 모델 선택 시 기사 재예측
    if model == 'logistic':
        all_articles = [dict(a) for a in all_articles]
        await asyncio.to_thread(predict_logistic_inplace, all_articles)
    elif model == 'svm':
        all_articles = [dict(a) for a in all_articles]
        await asyncio.to_thread(predict_svm_inplace, all_articles)
    elif model == 'ensemble':
        all_articles = [dict(a) for a in all_articles]
        await asyncio.to_thread(predict_ensemble_inplace, all_articles)

    show_badges = model in ('xgboost', 'logistic', 'svm', 'ensemble')

    if show_badges:
        real_count = sum(1 for a in all_articles if a['predicted_label'] == 0)
        fake_count = sum(1 for a in all_articles if a['predicted_label'] == 1)
        fake_arts  = [a for a in all_articles if a['predicted_label'] == 1]
        dvl_counts = [
            ('통계 왜곡',     sum(a.get('stat_distortion', 0)       for a in fake_arts), 'stat_distortion'),
            ('인과 오류',     sum(a.get('causal_error', 0)           for a in fake_arts), 'causal_error'),
            ('감정 자극',     sum(a.get('emotional_provocation', 0)  for a in fake_arts), 'emotional_provocation'),
            ('출처 불명',     sum(a.get('source_lack', 0)            for a in fake_arts), 'source_lack'),
            ('이미지 불일치', sum(a.get('img_mismatch', 0)           for a in fake_arts), 'img_mismatch'),
        ]
        # 진짜(0) 상단, 가짜(1) 하단 정렬 + DVL 필터 적용
        if flt == 'real':
            filtered = [a for a in all_articles if a['predicted_label'] == 0]
        elif flt == 'fake':
            filtered = [a for a in all_articles if a['predicted_label'] == 1]
        elif flt in _DVL_FILTERS:
            filtered = [a for a in all_articles if a['predicted_label'] == 1 and a.get(flt, 0) == 1]
        else:
            filtered = sorted(all_articles,
                              key=lambda x: (x['predicted_label'], x.get('fake_score', 0)))
    else:
        # 모델 미선택: 분류 없이 기사만 표시
        real_count = 0
        fake_count = 0
        dvl_counts = []
        filtered   = all_articles
        flt        = 'all'

    total_pages  = max(1, (len(filtered) + VIEW_PAGE_SIZE - 1) // VIEW_PAGE_SIZE)
    current_page = min(page, total_pages)
    start        = (current_page - 1) * VIEW_PAGE_SIZE
    page_articles = filtered[start: start + VIEW_PAGE_SIZE]

    parts = []
    if flt != 'all':
        parts.append(f'filter={flt}')
    if model:
        parts.append(f'model={model}')
    filter_param    = ('&' + '&'.join(parts)) if parts else ''
    dvl_filter_name = _DVL_FILTERS.get(flt, '')

    return templates.TemplateResponse(
        request=request,
        name='naver_news_right.html',
        context={
            'news_list':        page_articles,
            'all_articles':     all_articles,
            'show_badges':      show_badges,
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
            'nav_extra_params': filter_param,
            'active_filter':    flt,
            'active_model':     model or '',
            'dvl_counts':       dvl_counts,
            'dvl_filter_name':  dvl_filter_name,
        },
    )


@app.get('/rview/section/{sid1}', response_class=HTMLResponse)
async def rview_section(
    request: Request,
    sid1: str,
    page: int = Query(default=1, ge=1),
    view: str = Query(default='title'),
    sid2: str = Query(default=None),
    flt: str = Query(default='all', alias='filter'),
    model: str = Query(default=None),
):
    """우측 패널 전용: 카테고리 섹션 기사 목록."""
    if sid1 not in SECTIONS and sid1 != 'yonhap':
        return RedirectResponse(url='/view')
    if view not in VALID_VIEWS:
        view = 'title'
    if flt not in ('all', 'real', 'fake'):
        flt = 'all'

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

    for a in all_articles:
        if 'predicted_label' not in a:
            a['predicted_label'] = 0
        if 'fake_score' not in a:
            a['fake_score'] = 0.0

    # 모델 선택 시 섹션 기사 재예측
    if model == 'logistic':
        all_articles = [dict(a) for a in all_articles]
        await asyncio.to_thread(predict_logistic_inplace, all_articles)
    elif model == 'svm':
        all_articles = [dict(a) for a in all_articles]
        await asyncio.to_thread(predict_svm_inplace, all_articles)
    elif model == 'ensemble':
        all_articles = [dict(a) for a in all_articles]
        await asyncio.to_thread(predict_ensemble_inplace, all_articles)

    show_badges = model in ('xgboost', 'logistic', 'svm', 'ensemble')

    if show_badges:
        real_count = sum(1 for a in all_articles if a.get('predicted_label', 0) == 0)
        fake_count = sum(1 for a in all_articles if a.get('predicted_label', 0) == 1)
        fake_arts  = [a for a in all_articles if a.get('predicted_label', 0) == 1]
        dvl_counts = [
            ('통계 왜곡',     sum(a.get('stat_distortion', 0)       for a in fake_arts), 'stat_distortion'),
            ('인과 오류',     sum(a.get('causal_error', 0)           for a in fake_arts), 'causal_error'),
            ('감정 자극',     sum(a.get('emotional_provocation', 0)  for a in fake_arts), 'emotional_provocation'),
            ('출처 불명',     sum(a.get('source_lack', 0)            for a in fake_arts), 'source_lack'),
            ('이미지 불일치', sum(a.get('img_mismatch', 0)           for a in fake_arts), 'img_mismatch'),
        ]
        if flt == 'real':
            filtered = [a for a in all_articles if a.get('predicted_label', 0) == 0]
        elif flt == 'fake':
            filtered = [a for a in all_articles if a.get('predicted_label', 0) == 1]
        else:
            filtered = all_articles
    else:
        real_count = 0
        fake_count = 0
        dvl_counts = []
        filtered   = all_articles
        flt        = 'all'

    total_pages  = max(1, (len(filtered) + VIEW_PAGE_SIZE - 1) // VIEW_PAGE_SIZE)
    current_page = min(page, total_pages)
    start        = (current_page - 1) * VIEW_PAGE_SIZE
    page_articles = filtered[start: start + VIEW_PAGE_SIZE]

    # 페이지네이션에서 sid2 + filter + model 상태 유지
    parts = []
    if sid2:
        parts.append(f'sid2={sid2}')
    if flt != 'all':
        parts.append(f'filter={flt}')
    if model:
        parts.append(f'model={model}')
    nav_extra_params = ('&' + '&'.join(parts)) if parts else ''

    return templates.TemplateResponse(
        request=request,
        name='naver_news_right.html',
        context={
            'news_list':        page_articles,
            'all_articles':     all_articles,
            'show_badges':      show_badges,
            'real_count':       real_count,
            'fake_count':       fake_count,
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
            'active_filter':    flt,
            'active_model':     model or '',
            'dvl_counts':       dvl_counts,
            'dvl_filter_name':  '',
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 라우트 — 네이버 뉴스 서버 사이드 프록시 (iframe X-Frame-Options 우회)
# ─────────────────────────────────────────────────────────────────────────────

_NAVER_LIST_URL       = 'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=001'
_NAVER_LIST_URL_TITLE = _NAVER_LIST_URL + '&listType=title'
_PROXY_HEADERS = {
    'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0.0.0 Safari/537.36',
    'Referer':         'https://news.naver.com/',
    'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
    'Accept':          'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


async def _fetch_naver_raw(url: str) -> str:
    """aiohttp로 실제 네이버 뉴스 HTML 비동기 다운로드 (크롤러와 동일한 방식, Windows DNS 호환)."""
    import aiohttp as _aiohttp
    resolver  = _aiohttp.ThreadedResolver()
    connector = _aiohttp.TCPConnector(resolver=resolver)
    timeout   = _aiohttp.ClientTimeout(total=15)
    async with _aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers=_PROXY_HEADERS,
    ) as session:
        async with session.get(url, allow_redirects=True) as resp:
            resp.raise_for_status()
            return await resp.text(errors='replace')


_NAVER_TITLE_INJECT = '''\
<style id="_force_title_view">
/* 실제 Naver HTML 구조: 이미지는 dt.photo, 요약은 span.lede, 제목은 dt(plain) */
ul.type06_headline > li dt.photo,
ul.type06 > li dt.photo { display:none !important; }
ul.type06_headline .lede,
ul.type06 .lede { display:none !important; }
ul.type06_headline > li,
ul.type06 > li { display:block !important; padding:7px 0 !important; border-bottom:1px solid #eaebf0; }
ul.type06_headline > li dl,
ul.type06 > li dl { display:block !important; margin:0; }
ul.type06_headline > li dt,
ul.type06 > li dt { display:block !important; }
ul.type06_headline > li dt a,
ul.type06 > li dt a { font-size:13px !important; font-weight:normal !important; color:#1a1a1a; }
/* 혹시 thumb_area 구조도 있을 경우 대비 */
.type06_headline .thumb_area,.type06 .thumb_area { display:none !important; }
</style>
<script>
(function(){
  function f(){
    var t2=document.querySelectorAll('ul.type02');
    if(t2.length>0){
      /* type02 목록이 있으면: type06 숨기고 type02만 표시 */
      ['type06_headline','type06'].forEach(function(c){
        [].forEach.call(document.querySelectorAll('ul.'+c),function(el){
          el.style.setProperty('display','none','important');
        });
      });
      [].forEach.call(t2,function(el){
        el.style.setProperty('display','block','important');
      });
    }
    /* type02 없으면 CSS가 이미지/요약만 숨김 — type06 목록은 그대로 유지 */
  }
  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded',function(){f();setTimeout(f,600);});
  } else { f(); setTimeout(f,600); }
})();
</script>'''

_PROXY_INTERCEPT_JS = '''\
<script>
(function(){
  document.addEventListener('click', function(e){
    var a = e.target;
    while (a && a.tagName !== 'A') a = a.parentElement;
    if (!a) return;
    var rawHref = a.getAttribute('href') || '';
    var href    = a.href    || '';
    if (!rawHref || rawHref === '#'
        || rawHref.indexOf('javascript:') === 0
        || rawHref.indexOf('mailto:')     === 0) return;
    if (a.getAttribute('target') === '_blank') return;

    /* 우리 서버 라우트 (/lview/, /proxy/) —
       <base href> 가 naver.com 을 가리키므로 rawHref 로 직접 이동해야 함 */
    if (rawHref.indexOf('/lview/') === 0 || rawHref.indexOf('/proxy/') === 0) {
      e.preventDefault();
      e.stopPropagation();
      window.location.href = rawHref;
      return;
    }

    /* 뷰타입 탭 (listType=) — 우리 서버 /lview/section/{sid}?listType={lt} 로 변환
       실제 Naver HTML: href="list.naver;jsessionid=...?...&listType=xxx" (상대경로)
       base href 적용 후 a.href = "https://news.naver.com/list.naver;..." 형태 */
    if (href.indexOf('news.naver.com') !== -1
        && href.indexOf('listType=') !== -1) {
      e.preventDefault();
      e.stopPropagation();
      var sidM = href.match(/[?&]sid1=(\\d+)/);
      var ltM  = href.match(/[?&]listType=([^&]+)/);
      var sid  = sidM ? sidM[1] : '001';
      var lt   = ltM  ? ltM[1]  : 'title';
      window.location.href = '/lview/section/' + sid + '?listType=' + encodeURIComponent(lt);
      return;
    }

    /* 기사 URL → 새 탭 */
    if (href.indexOf('n.news.naver.com') !== -1
        || href.indexOf('/article/') !== -1
        || href.indexOf('/mnews/')    !== -1
        || href.indexOf('/read.naver') !== -1) {
      e.preventDefault();
      e.stopPropagation();
      window.open(href, '_blank');
    }
  }, true);
})();
</script>'''

_NAVER_ASIDE_JS = '''\
<script>
/* 왼쪽 패널 aside 버튼 복원 — Naver 스크립트 제거 후 재구현 */
(function(){
  function initAside(){
    /* 랭킹 섹션: _refreshButton → 다음 5개 순환, _rankingInfoButton → 안내 토글 */
    document.querySelectorAll('[class*="_officeTopRanking"]').forEach(function(base){
      var lists   = base.querySelectorAll('._rankingList');
      var current = 0;
      var refreshBtn = base.querySelector('._refreshButton');
      if (refreshBtn && lists.length > 1) {
        refreshBtn.addEventListener('click', function(e){
          e.preventDefault();
          lists[current].style.display = 'none';
          current = (current + 1) % lists.length;
          lists[current].style.display = 'block';
        });
      }
      var infoBtn   = base.querySelector('._rankingInfoButton');
      var infoLayer = base.querySelector('._rankingInfoLayer');
      var closeBtn  = base.querySelector('._rankingInfoCloseButton');
      if (infoBtn && infoLayer) {
        infoBtn.addEventListener('click', function(e){
          e.stopPropagation();
          infoLayer.style.display = infoLayer.style.display === 'none' ? 'block' : 'none';
        });
      }
      if (closeBtn && infoLayer) {
        closeBtn.addEventListener('click', function(){
          infoLayer.style.display = 'none';
        });
      }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAside);
  } else {
    initAside();
  }
})();
</script>'''


def _fix_naver_html(html: str, force_title: bool = True) -> str:
    """프록시된 HTML의 URL을 절대경로로 수정, 클릭 인터셉터 주입."""
    _VALID_SIDS = {'001', '100', '101', '102', '103', '104', '105', '110'}

    # 1) 프로토콜 생략 URL(//...) → https://
    html = re.sub(r'((?:src|href|action|content)=["\'])//([^"\']+)',
                  lambda m: m.group(1) + 'https://' + m.group(2), html)
    html = re.sub(r"((?:src|href|action|content)=')//([^']+)",
                  lambda m: m.group(1) + 'https://' + m.group(2), html)

    # 2) X-Frame-Options / CSP / meta-refresh 메타태그 제거
    html = re.sub(r'<meta[^>]+(?:x-frame-options|content-security-policy)[^>]*>',
                  '', html, flags=re.IGNORECASE)
    html = re.sub(r'<meta[^>]+http-equiv=["\']refresh["\'][^>]*>',
                  '', html, flags=re.IGNORECASE)

    # 3) Naver 스크립트 전체 제거
    html = re.sub(r'<script\b[^>]*>.*?</script>', '', html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<script\b[^>]*/>', '', html, flags=re.IGNORECASE)

    # 4) 제목형 강제 CSS+JS 주입 (force_title 일 때만)
    if force_title:
        html = re.sub(r'</head>',
                      lambda m: _NAVER_TITLE_INJECT + '\n</head>',
                      html, count=1, flags=re.IGNORECASE)

    # 5) 우리 서버 라우트로 교체 — base href 없이 iframe URL 기준으로 해석되도록
    #    &amp; 인코딩 대응: [?&](?:amp;)? 패턴으로 &와 &amp; 모두 처리

    #    (a) /main/list.naver?...&sid1=X  →  /lview/section/X (page 파라미터 보존)
    def _sub_sid(m):
        href_val = m.group(1)
        if 'listType=' in href_val:
            return m.group(0)   # listType 링크는 아래 5-c에서 처리
        sid_m  = re.search(r'[?&](?:amp;)?sid1=(\d+)', href_val)
        sid2_m = re.search(r'[?&](?:amp;)?sid2=(\d+)', href_val)
        page_m = re.search(r'[?&](?:amp;)?page=(\d+)', href_val)
        if sid_m and sid_m.group(1) in _VALID_SIDS:
            base   = f'/lview/section/{sid_m.group(1)}'
            params = []
            if sid2_m:
                params.append(f'sid2={sid2_m.group(1)}')
            if page_m and page_m.group(1) != '1':
                params.append(f'page={page_m.group(1)}')
            qs = ('?' + '&'.join(params)) if params else ''
            return f'href="{base}{qs}"'
        if 'mode=LPOD' in href_val or 'isYeonhapFlash' in href_val:
            return 'href="/lview/section/yonhap"'
        return m.group(0)

    html = re.sub(
        r'href="((?:https?://(?:news|m)\.naver\.com)?/main/list\.naver[^"]*(?:sid1=\d+|mode=LPOD)[^"]*)"',
        _sub_sid, html,
    )

    #    (b) /section/X  →  /lview/section/X  (신형 Naver URL)
    html = re.sub(
        r'href="/section/(\d+)"',
        lambda m: f'href="/lview/section/{m.group(1)}"' if m.group(1) in _VALID_SIDS else m.group(0),
        html,
    )

    #    (c) list.naver;jsessionid=...?...&listType=Y  →  /lview/section/X?listType=Y
    #        &amp; 인코딩 대응 포함
    def _sub_listtype(m):
        href_val = m.group(1)
        sid_m = re.search(r'[?&](?:amp;)?sid1=(\d+)', href_val)
        lt_m  = re.search(r'[?&](?:amp;)?listType=([^&;"\']+)', href_val)
        if not lt_m:
            return m.group(0)
        sid = sid_m.group(1) if sid_m else '001'
        lt  = lt_m.group(1)
        return f'href="/lview/section/{sid}?listType={lt}"'

    html = re.sub(r'href="(list\.naver[^"]*listType[^"]*)"', _sub_listtype, html)

    #    (c-ext) list.naver?...&sid1=X&sid2=Y (상대경로, listType 없음) → /lview/section/X?sid2=Y
    #            세부 카테고리 링크 처리 (슬래시 없는 상대경로 대응)
    def _sub_relative_sid(m):
        href_val = m.group(1)
        if 'listType=' in href_val:
            return m.group(0)   # 이미 5-c에서 처리됨
        sid_m  = re.search(r'[?&](?:amp;)?sid1=(\d+)', href_val)
        sid2_m = re.search(r'[?&](?:amp;)?sid2=(\d+)', href_val)
        page_m = re.search(r'[?&](?:amp;)?page=(\d+)', href_val)
        if sid_m and sid_m.group(1) in _VALID_SIDS:
            base   = f'/lview/section/{sid_m.group(1)}'
            params = []
            if sid2_m:
                params.append(f'sid2={sid2_m.group(1)}')
            if page_m and page_m.group(1) != '1':
                params.append(f'page={page_m.group(1)}')
            qs = ('?' + '&'.join(params)) if params else ''
            return f'href="{base}{qs}"'
        if 'mode=LPOD' in href_val or 'isYeonhapFlash' in href_val:
            return 'href="/lview/section/yonhap"'
        return m.group(0)

    html = re.sub(
        r'href="(list\.naver[^"]*(?:sid1=\d+|mode=LPOD)[^"]*)"',
        _sub_relative_sid, html,
    )

    #    (d) href="/"  →  /proxy/naver
    html = re.sub(r'href="/"', 'href="/proxy/naver"', html)

    # 6) 나머지 Naver 상대경로(/...) → 절대 URL  — /lview/ /proxy/ 는 우리 서버이므로 제외
    #    base href 를 쓰지 않으므로 CSS·이미지 경로를 직접 절대화해야 함
    def _abs_naver(m):
        attr_q = m.group(1)   # e.g. 'href="'
        path   = m.group(2)   # e.g. '/news/img/icon.gif'
        if path.startswith('/lview/') or path.startswith('/proxy/'):
            return m.group(0)
        return attr_q + 'https://news.naver.com' + path

    html = re.sub(r'((?:href|src|action)=")(/[^"]+)', _abs_naver, html)
    html = re.sub(r"((?:href|src|action)=')(/[^']+)",
                  lambda m: m.group(1) + 'https://news.naver.com' + m.group(2)
                  if not m.group(2).startswith('/lview/') and not m.group(2).startswith('/proxy/')
                  else m.group(0), html)

    # 7) 뷰타입 탭 활성 클래스 교체 — force_title 일 때만 제목형 강제
    if force_title:
        html = re.sub(r'\btit_off\b', 'tit_on', html)
        html = re.sub(r'\bsum_on\b',  'sum_off', html)
        html = re.sub(r'\bphoto_on\b', 'photo_off', html)
        html = re.sub(r'\bnewsp_on\b', 'newsp_off', html)

    # 8) 클릭 인터셉터 + aside 버튼 JS 삽입
    inject = _PROXY_INTERCEPT_JS + '\n' + _NAVER_ASIDE_JS
    if re.search(r'</body>', html, re.IGNORECASE):
        html = re.sub(r'</body>',
                      lambda m: inject + '\n</body>',
                      html, count=1, flags=re.IGNORECASE)
    else:
        html += inject
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
        raw  = await _fetch_naver_raw(_NAVER_LIST_URL)
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
    if _cache['proxy_html'] and (time.time() - _cache['proxy_at']) < _LVIEW_CACHE_TTL:
        return HTMLResponse(content=_cache['proxy_html'])

    try:
        raw  = await _fetch_naver_raw(_NAVER_LIST_URL_TITLE)
        # aside 캐시가 비어 있으면 원본 URL로 추출
        if not _cache['aside_html']:
            raw_base = await _fetch_naver_raw(_NAVER_LIST_URL)
            aside = await asyncio.to_thread(_extract_naver_aside_sync, raw_base)
            if aside:
                _cache['aside_html'] = aside
                _cache['aside_at']   = time.time()
        html = await asyncio.to_thread(_fix_naver_html, raw)   # 스레드풀: 이벤트 루프 비블로킹
        _cache['proxy_html'] = html
        _cache['proxy_at']   = time.time()
        asyncio.create_task(asyncio.to_thread(_save_proxy_html_to_disk, html))  # 디스크 비동기 저장
        return HTMLResponse(content=html)
    except Exception as e:
        print(f'[proxy] 네이버 직접 fetch 실패 ({e}) - 크롤러 캐시 fallback')
        articles = _cache['articles']
        return templates.TemplateResponse(
            request=request,
            name='naver_news_left.html',
            context={
                'news_list':    articles[:30],
                'loading':      len(articles) == 0,
                'today':        _today_str(),
                'section_name': None,
                'section_id':   None,
            },
        )


async def _precache_lview_listtypes(sid1: str, base_naver_url: str, current_lt: str):
    """현재 섹션의 다른 뷰타입을 백그라운드 캐싱 — 탭 전환 속도 개선."""
    for lt in ('title', 'summary', 'photo', 'paper'):
        if lt == current_lt:
            continue
        ck = f'{sid1}:{lt}'
        if ck in _lview_proxy_cache and (time.time() - _lview_proxy_cache[ck].get('at', 0)) < _LVIEW_CACHE_TTL:
            continue
        try:
            url  = base_naver_url + f'&listType={lt}'
            raw  = await _fetch_naver_raw(url)
            html = await asyncio.to_thread(_fix_naver_html, raw, lt == 'title')  # 스레드풀
            _lview_proxy_cache[ck] = {'html': html, 'at': time.time()}
            print(f'[lview] [{sid1}:{lt}] 뷰타입 사전 캐싱 완료')
        except Exception as e:
            print(f'[lview] [{sid1}:{lt}] 사전 캐싱 실패: {e}')


@app.get('/lview/section/{sid1}', response_class=HTMLResponse)
async def lview_section(
    request: Request,
    sid1: str,
    listType: str = Query(default=None),
    sid2: str = Query(default=None),
    page: int = Query(default=1, ge=1),
):
    """왼쪽 패널 전용: 실제 네이버 카테고리 섹션 페이지를 프록시로 렌더링."""
    if sid1 not in SECTIONS and sid1 != 'yonhap' and sid1 != '001':
        return RedirectResponse(url='/proxy/naver')

    # listType 이 없거나 'title' 이면 제목형 강제, 그 외엔 네이버 원본 뷰 사용
    force_title = (not listType) or (listType == 'title')

    # 캐시 키: page=1 은 워밍업 캐시와 호환, page>1 은 별도 키
    if page == 1:
        cache_key_str = f'{sid1}:{sid2}:{listType or ""}' if sid2 else f'{sid1}:{listType or ""}'
    else:
        cache_key_str = f'{sid1}:{sid2}:{listType or ""}:p{page}' if sid2 else f'{sid1}:{listType or ""}:p{page}'

    cached = _lview_proxy_cache.get(cache_key_str)
    if cached and (time.time() - cached['at']) < _LVIEW_CACHE_TTL:
        return HTMLResponse(content=cached['html'])

    # 세부 카테고리(sid2): 메인 카테고리와 동일하게 Naver 프록시 HTML로 렌더링
    # LS2D URL로 실제 Naver HTML fetch → _fix_naver_html 적용 → 캐시 저장
    if sid2:
        naver_sid2_url = (
            f'https://news.naver.com/main/list.naver'
            f'?mode=LS2D&mid=sec&sid1={sid1}&sid2={sid2}&listType=title'
        )
        if page > 1:
            naver_sid2_url += f'&page={page}'
        try:
            raw  = await _fetch_naver_raw(naver_sid2_url)
            html = await asyncio.to_thread(_fix_naver_html, raw, True)
            _lview_proxy_cache[cache_key_str] = {'html': html, 'at': time.time()}
            return HTMLResponse(content=html)
        except Exception as _sid2_exc:
            print(f'[lview] sid2={sid2} fetch/parse error: {type(_sid2_exc).__name__}: {_sid2_exc}')
            # Naver fetch 실패 시 크롤러 캐시 fallback
            sec_key      = f'{sid1}:{sid2}'
            sec_cache    = _get_section_cache(sec_key)
            section_info = SECTIONS.get(sid1, {})
            section_name = section_info.get('name', '') if sid1 in SECTIONS else ''
            sub_name_map = {str(s['sid2']): s['name'] for s in section_info.get('sub_categories', []) if s.get('sid2')}
            active_sub_name = sub_name_map.get(str(sid2), '')
            if (not sec_cache['articles']) or ((time.time() - sec_cache['fetched_at']) > CACHE_TTL):
                if sec_key not in _section_crawling:
                    base_url = f'https://news.naver.com/main/list.naver?mode=LS2D&mid=sec&sid1={sid1}&sid2={sid2}'
                    asyncio.create_task(_run_section_crawl(sec_key, base_url))
            return templates.TemplateResponse(
                request=request,
                name='naver_news_left.html',
                context={
                    'news_list':      sec_cache['articles'],
                    'loading':        len(sec_cache['articles']) == 0,
                    'today':          _today_str(),
                    'section_name':   f'{section_name} · {active_sub_name}' if active_sub_name else section_name,
                    'section_id':     sid1,
                    'sub_categories': section_info.get('sub_categories', []),
                    'active_sid2':    sid2,
                },
            )

    if sid1 == 'yonhap':
        naver_url = YONHAP_URL
    else:
        naver_url = CATEGORY_BASE_URLS.get(
            sid1,
            f'https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1={sid1}',
        )
    # listType 명시 시 그대로 사용, 없으면 title 뷰(5개 묶음 구조) 기본 적용
    effective_lt = listType if listType else 'title'
    base_naver_url = naver_url
    naver_url += f'&listType={effective_lt}'
    if page > 1:
        naver_url += f'&page={page}'

    try:
        raw  = await _fetch_naver_raw(naver_url)
        html = await asyncio.to_thread(_fix_naver_html, raw, force_title)  # 스레드풀
        _lview_proxy_cache[cache_key_str] = {'html': html, 'at': time.time()}
        # 첫 페이지 방문 시 다른 뷰타입을 백그라운드에서 사전 캐싱 (탭 전환 속도 개선)
        if page == 1:
            asyncio.create_task(_precache_lview_listtypes(sid1, base_naver_url, effective_lt))
        return HTMLResponse(content=html)
    except Exception as e:
        print(f'[lview] 네이버 직접 fetch 실패 ({e}) - 크롤러 캐시 fallback')
        sec_key      = f'{sid1}:' if sid1 != 'yonhap' else 'yonhap'
        sec_cache    = _get_section_cache(sec_key)
        section_info = SECTIONS.get(sid1, {})
        section_name = section_info.get('name', '연합뉴스 속보') if sid1 in SECTIONS else '연합뉴스 속보'
        return templates.TemplateResponse(
            request=request,
            name='naver_news_left.html',
            context={
                'news_list':      sec_cache['articles'][:30],
                'loading':        len(sec_cache['articles']) == 0,
                'today':          _today_str(),
                'section_name':   section_name,
                'section_id':     sid1,
                'sub_categories': section_info.get('sub_categories', []),
                'active_sid2':    sid2,
            },
        )


_PAGE_CACHE_TTL   = 180   # 기사 페이지 캐시 3분
_page_cache: dict[str, dict] = {}  # {url: {'html': str, 'at': float}}
_lview_proxy_cache: dict[str, dict] = {}  # {sid1: {'html': str, 'at': float}}

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

    cached = _page_cache.get(url)
    if cached and (time.time() - cached['at']) < _PAGE_CACHE_TTL:
        return HTMLResponse(content=cached['html'])

    try:
        raw  = await _fetch_naver_raw(url)
        html = _fix_naver_html(raw)
        _page_cache[url] = {'html': html, 'at': time.time()}
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