# 가짜뉴스 점수 계산 — 모델 연결 전 임시 랜덤 점수 부여

import re
import random


def _time_sort_key(article: dict) -> tuple:
    """
    기사 time 필드를 정렬 키로 변환.
    네이버 목록 time 포맷: "10:30" / "오전 10:30" / "오후 2:15" / "2024.01.01."
    최신 기사가 앞에 오도록 내림차순 키 반환.
    """
    t = article.get('time', '')

    # "오후 HH:MM" → 24시간으로 변환
    m = re.match(r'오후\s*(\d+):(\d+)', t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return (1, (12 + h if h != 12 else 12) * 60 + mn)

    # "오전 HH:MM" 또는 "HH:MM"
    m = re.match(r'(?:오전\s*)?(\d+):(\d+)', t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return (1, h * 60 + mn)

    # 날짜 포맷 "YYYY.MM.DD." → 오래된 기사
    m = re.match(r'(\d{4})\.(\d{2})\.(\d{2})', t)
    if m:
        return (0, int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3)))

    return (0, 0)


def get_sorted_timeline(raw_news_list: list) -> list:
    """
    크롤링된 기사 리스트에 임시 랜덤 fake_score를 부여하고
    최신순(time 내림차순)으로 정렬해 반환.
    - 50.0 ~ 100.0 사이 랜덤 값 (소수점 1자리)
    - 50 이상 → 가짜(빨간), 50 미만 → 진짜(초록)
    모델 연결 후 이 함수에서 실제 예측 점수로 교체 예정.
    """
    for news in raw_news_list:
        news['fake_score'] = round(random.uniform(0, 100), 1)

    # 최신 기사 순으로 정렬
    return sorted(raw_news_list, key=_time_sort_key, reverse=True)
