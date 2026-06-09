# -*- coding: utf-8 -*-
"""
Okt 형태소 분석기 공유 모듈
train_nbsvm.py와 detector.py에서 함께 사용 — joblib 역직렬화 시 이 경로로 함수를 찾음
"""
_okt = None



# vectorize.py 기준 7종 품사와 동일하게 맞춤 (Noun/Verb/Adj/Adv/Alpha/Foreign/Number)
_KEEP_POS = frozenset({'Noun', 'Verb', 'Adjective', 'Adverb', 'Alpha', 'Foreign', 'Number'})

STOPWORDS = frozenset([
    # 범용 동사 (vectorize.py KOREAN_STOPWORDS와 동일)
    '하다', '있다', '되다', '돼다', '이다', '않다', '없다',
    '받다', '늘다', '따르다', '오다', '가다', '보다', '같다',
    '아니다', '그렇다', '이렇다', '어떻다', '나오다', '보이다',
    '들다', '내다', '맞다', '시키다', '만들다', '대다',
    '이르다', '나타나다', '밝히다', '말하다',
    # 문어체 연결어 / 의존어
    '위해', '통해', '대한', '대해', '때문', '함께', '한편',
    '가운데', '지난', '이번', '현재', '최근', '특히',
    '모든', '모두', '다른', '이상', '이후', '부터',
    '다양하다', '높다', '가능하다', '필요하다', '가장',
    '이렇게', '라며', '우리', '자신',
    # 뉴스 도메인 전용
    '기자', '뉴스', '보도', '그리고', '그래서',
])


def _get_okt():
    global _okt
    if _okt is None:
        from konlpy.tag import Okt
        _okt = Okt()
    return _okt


def okt_tokenizer(text: str) -> list:
    okt = _get_okt()
    return [w for w, p in okt.pos(text, stem=True)
            if p in _KEEP_POS and w not in STOPWORDS and len(w) > 1]
