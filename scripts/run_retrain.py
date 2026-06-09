# -*- coding: utf-8 -*-
"""
팩트체크 데이터 병합 후 모델 전체 재훈련 파이프라인

실행 순서:
  1. merge_factcheck.py         - 가짜뉴스 107건을 unified_news_refined.csv에 병합
  2. 형태소 캐시 업데이트       - 신규 기사 107건만 Okt 처리 후 캐시에 추가
  3. split_dataset.py           - subset_01~N.csv 재생성
  4. vectorize.py               - 한국어 TF-IDF 벡터 재생성 (XGBoost용)
  5. src/train_nbsvm.py         - NBSVM 재훈련 (subset_01.csv 기반, 20~30분)
  6. tune_xgboost.py            - XGBoost Optuna 재훈련 (40 trials)
"""

import os
import sys
import subprocess
import time
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR   = os.path.join(BASE_DIR, 'src')
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
VEC_DIR   = os.path.join(BASE_DIR, 'data', 'vector')

UNIFIED_CSV     = os.path.join(PROC_DIR, 'unified_news_refined.csv')
FACTCHECK_CSV   = os.path.join(BASE_DIR, 'data', 'factcheck', 'factcheck_label.csv')
CACHE_CSV       = os.path.join(PROC_DIR, 'unified_news_tokenized.csv')
SUBSETS_DIR     = os.path.join(PROC_DIR, 'subsets')

ENGLISH_MEDIA = {
    'politicsNews', 'worldnews', 'News', 'Unknown',
    'politics', 'US_News', 'left-news', 'Government News',
}
KEEP_POS = {'Noun', 'Verb', 'Adjective', 'Adverb', 'Alpha', 'Foreign', 'Number'}


def run(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=False,
                            text=True, encoding='utf-8', errors='replace')
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[오류] '{desc}' 실패 (exit {result.returncode}). 파이프라인 중단.")
        sys.exit(result.returncode)
    print(f"  완료: {elapsed/60:.1f}분")


# ── STEP 1: 병합 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 1/6 : 팩트체크 데이터 병합")
print("="*60)

before_df = pd.read_csv(UNIFIED_CSV, usecols=['label'])
before_cnt = len(before_df)
before_fake = (before_df['label'] == 1).sum()

run([sys.executable, os.path.join(SRC_DIR, 'merge_factcheck.py')],
    "merge_factcheck.py 실행")

after_df   = pd.read_csv(UNIFIED_CSV, usecols=['label', 'media'])
after_cnt  = len(after_df)
after_fake = (after_df['label'] == 1).sum()
n_new      = after_cnt - before_cnt
print(f"  추가된 기사: {n_new}건  |  가짜뉴스: {before_fake:,} → {after_fake:,}")


# ── STEP 2: 형태소 캐시 업데이트 ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2/6 : 형태소 캐시 업데이트 (신규 기사만 Okt 처리)")
print("="*60)

kor_df    = after_df[~after_df['media'].isin(ENGLISH_MEDIA)].reset_index(drop=True)
cache_df  = pd.read_csv(CACHE_CSV)
cache_len = len(cache_df)
kor_len   = len(kor_df)

if cache_len >= kor_len:
    print(f"  캐시({cache_len}) >= kor_df({kor_len}) — 업데이트 불필요")
else:
    n_missing = kor_len - cache_len
    print(f"  캐시 부족: {n_missing}건 추가 토크나이징 필요")

    full_kor = pd.read_csv(UNIFIED_CSV)
    full_kor = full_kor[~full_kor['media'].isin(ENGLISH_MEDIA)].reset_index(drop=True)
    new_rows  = full_kor.iloc[cache_len:]   # 새로 추가된 행들

    from konlpy.tag import Okt
    okt = Okt()

    def morph(text: str) -> str:
        try:
            tokens = [w for w, p in okt.pos(str(text), stem=True)
                      if p in KEEP_POS and len(w) > 1]
            return ' '.join(tokens)
        except Exception:
            return ''

    new_tokens = []
    for i, row in enumerate(new_rows.itertuples(), 1):
        clean = getattr(row, 'clean_message', '') or ''
        tok   = morph(clean)
        new_tokens.append(tok)
        if i % 10 == 0:
            print(f"  토크나이징 {i}/{n_missing}...")

    new_ids   = list(range(cache_len + 1, kor_len + 1))
    new_cache = pd.DataFrame({
        'id':     new_ids,
        'tokens': new_tokens,
        'label':  new_rows['label'].values,
    })
    updated = pd.concat([cache_df, new_cache], ignore_index=True)
    updated.to_csv(CACHE_CSV, index=False, encoding='utf-8-sig')
    print(f"  캐시 저장 완료: {len(updated):,}행")


# ── STEP 3: 서브셋 재분할 ─────────────────────────────────────────────────────
run([sys.executable, os.path.join(BASE_DIR, 'training', 'split_dataset.py'),
     '--input', UNIFIED_CSV, '--output_dir', SUBSETS_DIR],
    "STEP 3/6 : split_dataset.py (서브셋 재생성)")


# ── STEP 4: 재벡터화 (XGBoost용) ─────────────────────────────────────────────
run([sys.executable, os.path.join(BASE_DIR, 'training', 'vectorize.py')],
    "STEP 4/6 : vectorize.py (TF-IDF 재생성)")


# ── STEP 5: NBSVM 재훈련 ─────────────────────────────────────────────────────
run([sys.executable, os.path.join(BASE_DIR, 'training', 'train_nbsvm.py')],
    "STEP 5/6 : train_nbsvm.py (NBSVM 재훈련, 20~30분 예상)")


# ── STEP 6: XGBoost Optuna 재훈련 ────────────────────────────────────────────
run([sys.executable, os.path.join(BASE_DIR, 'scripts', 'tune_xgboost.py')],
    "STEP 6/6 : tune_xgboost.py (XGBoost Optuna 40 trials)")


print("\n" + "="*60)
print("  전체 파이프라인 완료!")
print("="*60)
print("  갱신된 모델:")
print("    models/svm/nbsvm_ko_*.pkl  (NBSVM 7개 컴포넌트)")
print("    models/xgboost/xgb_model_ko.pkl")
print("    models/xgboost/xgb_selector_ko.pkl")
print("\n  서버 재시작: python main.py")
