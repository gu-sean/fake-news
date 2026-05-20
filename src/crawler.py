"""
네이버 뉴스 크롤러
requests + BeautifulSoup 기반 정적 크롤링
"""

import time
import requests
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

RANKING_URL = 'https://news.naver.com/main/ranking/popularDay.naver'


def _get_soup(url: str, timeout: int = 10) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = 'utf-8'
        return BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f'[crawler] request failed: {url} -> {e}')
        return None


def fetch_article_body(url: str) -> str:
    """기사 본문 크롤링. 네이버 뉴스 포맷 변경에 대비해 셀렉터 우선순위 적용."""
    soup = _get_soup(url)
    if not soup:
        return ''

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


def fetch_ranking_articles(max_articles: int = 20) -> list[dict]:
    """
    네이버 뉴스 '언론사별 많이 본 뉴스' 크롤링.
    각 언론사 박스에서 1위 기사를 수집하여 max_articles개 반환.
    """
    soup = _get_soup(RANKING_URL)
    if not soup:
        print('[crawler] ranking page load failed -> using dummy data')
        return _dummy_articles()

    # 실제 구조: .rankingnews_box 가 언론사별로 82개 존재
    boxes = soup.select('.rankingnews_box')
    if not boxes:
        print('[crawler] selector mismatch -> using dummy data')
        return _dummy_articles()

    articles = []
    idx = 0

    for box in boxes:
        if len(articles) >= max_articles:
            break

        # 언론사명
        press_el = box.select_one('.rankingnews_name')
        press = press_el.get_text(strip=True) if press_el else '알 수 없음'

        # 기사 목록 (rankingnews_list > li)
        items = box.select('.rankingnews_list li')
        if not items:
            continue

        # 언론사 1위 기사만 수집
        item = items[0]
        a_tag = item.select_one('.list_title')
        time_el = item.select_one('.list_time')
        if not a_tag:
            continue

        title = a_tag.get_text(strip=True)
        url   = a_tag.get('href', '')
        pub_time = time_el.get_text(strip=True) if time_el else datetime.now().strftime('%H:%M')

        if not title or not url:
            continue

        idx += 1
        safe_press = press.encode('ascii', 'ignore').decode() or f'press_{idx}'
        print(f'[crawler] ({idx}/{max_articles}) {safe_press}')

        body = fetch_article_body(url)
        time.sleep(0.3)

        articles.append({
            'title':    title,
            'press':    press,
            'category': '뉴스',      
            'time':     pub_time,
            'url':      url,
            'text':     body,
        })

    if not articles:
        return _dummy_articles()

    return articles[:max_articles]


def _dummy_articles() -> list[dict]:
    """크롤링 실패 시 레이아웃 확인용 더미 데이터."""
    now = datetime.now().strftime('%H:%M')
    return [
        {
            'title':    '정부, 내년도 예산안 확정 발표…복지 예산 대폭 확대',
            'press':    '연합뉴스',
            'category': '정치',
            'time':     now,
            'url':      '#',
            'text':     '정부가 내년도 예산안을 확정 발표했다. 복지 예산이 대폭 확대된다.',
        },
        {
            'title':    '충격 단독 / 연예인 A씨, 비밀 스캔들 전격 폭로 — 모든 세금 면제 의혹',
            'press':    '가짜일보',
            'category': '연예',
            'time':     now,
            'url':      '#',
            'text':     '충격적인 단독 보도. 사실 확인이 되지 않은 내용을 무분별하게 배포하고 있다.',
        },
        {
            'title':    '코스피 2,600선 회복…외국인 순매수 전환',
            'press':    '한국경제',
            'category': '경제',
            'time':     now,
            'url':      '#',
            'text':     '코스피 지수가 2,600선을 회복했다. 외국인 투자자들이 순매수로 전환했다.',
        },
        {
            'title':    '속보 / AI 기술 혁신…국내 스타트업 글로벌 투자 유치',
            'press':    '디지털타임스',
            'category': 'IT/과학',
            'time':     now,
            'url':      '#',
            'text':     '국내 AI 스타트업이 글로벌 투자를 유치했다는 속보가 들어왔다.',
        },
        {
            'title':    '기후변화 대응 국제 협약 체결…130개국 서명',
            'press':    'KBS',
            'category': '세계',
            'time':     now,
            'url':      '#',
            'text':     '기후변화 대응을 위한 국제 협약이 체결됐다. 130개국이 서명에 동참했다.',
        },
    ]
