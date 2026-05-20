"""
벡터화 마스터 스크립트

단일 소스: data/processed/unified_news_refined.csv
  → media 컬럼으로 영어(Kaggle) / 한국어(AI Hub) 분리
  → 각각 TF-IDF 벡터화 후 data/vector/ 에 저장

출력 파일:
  data/vector/english_tfidf.npz        — 영어 TF-IDF 희소 행렬
  data/vector/english_labels.npy       — 영어 레이블 (0=진짜, 1=가짜)
  data/vector/english_dvl_flags.npy    — 영어 5대 분류 플래그 (n x 5)
  data/vector/english_vectorizer.pkl   — 영어 TfidfVectorizer (재사용용)
  data/vector/korean_tfidf.npz         — 한국어 TF-IDF 희소 행렬
  data/vector/korean_labels.npy        — 한국어 레이블 (0=진짜, 1=가짜)
  data/vector/korean_dvl_flags.npy     — 한국어 5대 분류 플래그 (n x 5)
  data/vector/korean_vectorizer.pkl    — 한국어 TfidfVectorizer (재사용용)

캐시:
  data/processed/unified_news_tokenized.csv — 한국어 형태소 분석 결과 (최초 1회)

"""

import os
import sys
import time
import pickle
import multiprocessing

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VEC_DIR  = os.path.join(BASE_DIR, 'data', 'vector')
os.makedirs(VEC_DIR, exist_ok=True)

SOURCE_CSV      = os.path.join(PROC_DIR, 'unified_news_refined.csv')
TOKENIZED_CACHE = os.path.join(PROC_DIR, 'unified_news_tokenized.csv')

DVL_FLAGS = ['stat_distortion', 'causal_error', 'emotional_provocation',
             'source_lack', 'img_mismatch']

ENGLISH_MEDIA = {
    'politicsNews', 'worldnews', 'News', 'Unknown',
    'politics', 'US_News', 'left-news', 'Government News',
}


# ─────────────────────────────────────────────────────────────────────────────
# 멀티프로세싱 워커 
# ─────────────────────────────────────────────────────────────────────────────

_okt = None  # 각 워커 프로세스 내 전역 Okt 인스턴스

def _init_worker():
    """워커 프로세스 시작 시 Okt를 1회 초기화."""
    global _okt
    from konlpy.tag import Okt
    _okt = Okt()

def _morph(text):
    """워커 함수: 모듈 전역 _okt 인스턴스를 재사용."""
    global _okt
    try:
        return ' '.join(_okt.morphs(str(text), stem=True))
    except Exception:
        return ''


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 분리
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    print('소스 로드: unified_news_refined.csv')
    df = pd.read_csv(SOURCE_CSV)
    print(f'  전체: {len(df):,}건')

    mask_eng = df['media'].isin(ENGLISH_MEDIA)
    eng_df = df[mask_eng].reset_index(drop=True)
    kor_df = df[~mask_eng].reset_index(drop=True)

    print(f'  영어(Kaggle):   {len(eng_df):,}건')
    print(f'  한국어(AI Hub): {len(kor_df):,}건')
    return eng_df, kor_df


# ─────────────────────────────────────────────────────────────────────────────
# 공통 저장 함수
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(prefix: str, X, y: np.ndarray, flags: np.ndarray, vec):
    sparse.save_npz(os.path.join(VEC_DIR, f'{prefix}_tfidf.npz'), X)
    np.save(os.path.join(VEC_DIR, f'{prefix}_labels.npy'), y)
    np.save(os.path.join(VEC_DIR, f'{prefix}_dvl_flags.npy'), flags)
    with open(os.path.join(VEC_DIR, f'{prefix}_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vec, f)

    for name in [f'{prefix}_tfidf.npz', f'{prefix}_labels.npy',
                 f'{prefix}_dvl_flags.npy', f'{prefix}_vectorizer.pkl']:
        size_mb = os.path.getsize(os.path.join(VEC_DIR, name)) / 1024 ** 2
        print(f'  저장: data/vector/{name} ({size_mb:.1f} MB)')


# ─────────────────────────────────────────────────────────────────────────────
# 영어 (Kaggle) 벡터화
# ─────────────────────────────────────────────────────────────────────────────

def vectorize_english(df: pd.DataFrame):
    print('\n' + '=' * 60)
    print('[영어] TF-IDF 벡터화')
    print('=' * 60)

    n_fake = int(df['label'].sum())
    print(f'  가짜: {n_fake:,} | 진짜: {len(df) - n_fake:,}')

    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
    )
    X = vec.fit_transform(df['clean_message'].fillna(''))
    y = df['label'].values.astype(np.int8)
    flags = df[DVL_FLAGS].values.astype(np.int8)

    print(f'  완료: {time.time() - t0:.1f}초 | 행렬: {X.shape}')
    save_outputs('english', X, y, flags, vec)
    print('[영어] 완료')


# ─────────────────────────────────────────────────────────────────────────────
# 한국어 (AI Hub) 벡터화
# ─────────────────────────────────────────────────────────────────────────────

def _run_morpheme_analysis(texts: pd.Series) -> pd.Series:
    """multiprocessing.Pool로 형태소 분석 병렬화.
    pandarallel 대신 Pool+initializer를 쓰는 이유:
    JPype 객체(Okt)는 dill/pickle 직렬화 불가 → pandarallel 사용 불가.
    initializer로 각 워커가 Okt를 독립적으로 초기화하면 직렬화 문제 없음.
    """
    try:
        from konlpy.tag import Okt  # noqa: F401 — 설치 확인용
    except ImportError:
        print('\n[오류] KoNLPy가 설치되지 않았습니다.')
        print('  설치: pip install konlpy  (Java 1.8+ 필요)')
        sys.exit(1)

    n_workers = multiprocessing.cpu_count()
    print(f'  멀티프로세싱 {n_workers}코어 병렬 처리')

    text_list = texts.tolist()
    t0 = time.time()

    with multiprocessing.Pool(processes=n_workers, initializer=_init_worker) as pool:
        # chunksize: 한 번에 워커에 넘길 청크 크기 (너무 작으면 오버헤드 증가)
        results = pool.map(_morph, text_list, chunksize=500)

    print(f'  형태소 분석 완료: {(time.time() - t0) / 60:.1f}분')
    return pd.Series(results, index=texts.index)


def vectorize_korean(df: pd.DataFrame):
    print('\n' + '=' * 60)
    print('[한국어] 형태소 분석 + TF-IDF 벡터화')
    print('=' * 60)

    n_fake = int(df['label'].sum())
    print(f'  가짜: {n_fake:,} | 진짜: {len(df) - n_fake:,}')

    # ── 형태소 분석 (캐시 우선) ───────────────────────────────────────
    if os.path.exists(TOKENIZED_CACHE):
        print('  캐시 발견 → 로드 (data/processed/unified_news_tokenized.csv)')
        tok_df = pd.read_csv(TOKENIZED_CACHE)
        tokens = tok_df['tokens'].fillna('')
    else:
        print('  형태소 분석 시작 (최초 1회만 — 이후 캐시 재사용)')
        tokens = _run_morpheme_analysis(df['clean_message'].fillna(''))

        cache_df = pd.DataFrame({
            'id':     df['id'].values,
            'tokens': tokens.values,
            'label':  df['label'].values,
        })
        cache_df.to_csv(TOKENIZED_CACHE, index=False, encoding='utf-8-sig')
        print('  캐시 저장 → data/processed/unified_news_tokenized.csv')

    # ── TF-IDF ────────────────────────────────────────────────────────
    print('  TF-IDF 벡터화 중...')
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vec.fit_transform(tokens)
    y = df['label'].values.astype(np.int8)
    flags = df[DVL_FLAGS].values.astype(np.int8)

    print(f'  완료: {time.time() - t0:.1f}초 | 행렬: {X.shape}')
    save_outputs('korean', X, y, flags, vec)
    print('[한국어] 완료')


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Windows spawn 방식에서 자식 프로세스가 이 블록을 재실행하지 않도록 보호
    multiprocessing.freeze_support()

    total_t0 = time.time()

    eng_df, kor_df = load_and_split()

    vectorize_english(eng_df)
    vectorize_korean(kor_df)

    total_min = (time.time() - total_t0) / 60
    print('\n' + '=' * 60)
    print(f'모든 벡터화 완료! (총 소요: {total_min:.1f}분)')
    print('=' * 60)
    print('\n[data/vector] 생성 파일:')
    for name in sorted(os.listdir(VEC_DIR)):
        size_mb = os.path.getsize(os.path.join(VEC_DIR, name)) / 1024 ** 2
        print(f'  {name:45s} {size_mb:6.1f} MB')
