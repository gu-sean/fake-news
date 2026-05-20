"""
네이버 뉴스 클론 — AI 가짜뉴스 탐지 타임라인
접속: http://localhost:8000
"""

import sys
import os
import time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from src.crawler import fetch_ranking_articles
from src.detector import get_sorted_timeline

app = FastAPI(title="네이버 뉴스 클론 - AI 가짜뉴스 탐지")
templates = Jinja2Templates(directory="templates")

# ── 메모리 캐시 (5분마다 재크롤링) ──────────────────────────────────────
_cache: dict = {'articles': [], 'fetched_at': 0}
CACHE_TTL = 300


def _get_news(force: bool = False) -> list[dict]:
    now = time.time()
    if force or not _cache['articles'] or (now - _cache['fetched_at']) > CACHE_TTL:
        print('[app] crawling naver news...')
        raw = fetch_ranking_articles(max_articles=20)
        sorted_news = get_sorted_timeline(raw)
        _cache['articles'] = sorted_news
        _cache['fetched_at'] = now
        print(f'[app] done: {len(sorted_news)} articles')
    return _cache['articles']


# ─────────────────────────────────────────────────────────────────────────────
# 라우트
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지 — 뉴스 타임라인 렌더링"""
    news_list = _get_news()
    today = datetime.now().strftime('%Y.%m.%d. ') + ['월','화','수','목','금','토','일'][datetime.now().weekday()] + '요일'
    return templates.TemplateResponse(
        request=request,
        name='naver_news_clone.html',
        context={'news_list': news_list, 'today': today},
    )


@app.get('/refresh', response_class=RedirectResponse)
async def refresh():
    """강제 재크롤링 후 메인으로 리다이렉트"""
    _get_news(force=True)
    return RedirectResponse(url='/')


@app.get('/api/news', response_class=JSONResponse)
async def api_news():
    """크롤링 결과 JSON 반환 (디버그용)"""
    return _get_news()



if __name__ == '__main__':
    print('=' * 50)
    print(' 네이버 뉴스 클론 - AI 가짜뉴스 탐지 시스템')
    print(' http://localhost:8000')
    print(' API 문서: http://localhost:8000/docs')
    print('=' * 50)
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=False)
