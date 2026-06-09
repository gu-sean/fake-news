# -*- coding: utf-8 -*-
"""
Logistic V4.3 재훈련 스크립트 (전체 데이터 사용)

피처:
  - 단어 TF-IDF (Okt 캐시 재사용, 60k) + 문자 n-gram TF-IDF (char_wb 2-4, 20k, 제목만)
  - Chi2 피처 선택 30,000개
  - 메타 피처 12개 (emo_count, quote_ratio 등) + StandardScaler
  - 문자 TF-IDF는 제목만 사용 (본문 제외 → 속도 대폭 향상)

데이터:
  - 단어 TF-IDF: unified_news_tokenized.csv (Okt 캐시, ~318k건, 재토크나이징 없음)
  - 문자 TF-IDF·메타: unified_news_refined.csv (원본 title/content, id 기준 join)

저장: models/logistic/logistic_v4_3_*.pkl (5개 컴포넌트)
소요 시간: 약 5~15분
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore', category=FutureWarning)

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

TOK_PATH  = os.path.join(_PROJECT_DIR, 'data', 'processed', 'unified_news_tokenized.csv')
REF_PATH  = os.path.join(_PROJECT_DIR, 'data', 'processed', 'unified_news_refined.csv')
SAVE_DIR  = os.path.join(_PROJECT_DIR, 'models', 'logistic')
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    title_str   = df['title'].astype(str)
    content_str = df['content'].astype(str)
    title_len   = title_str.str.len()
    content_len = content_str.str.len()
    pat = '결국|충격|경악|속보|단독|논란|분노|의혹|발각|공개|주의|확인'
    emo_count      = title_str.str.count(pat)
    quote_count    = content_str.str.count(r'["\']')
    ellipsis_count = title_str.str.count(r'\.\.\.')
    excl_count     = title_str.str.count('!') + content_str.str.count('!')
    quest_count    = title_str.str.count(r'\?') + content_str.str.count(r'\?')
    return pd.DataFrame({
        'emo_count':      emo_count,
        'emo_ratio':      emo_count      / (title_len + 1),
        'quote_count':    quote_count,
        'quote_ratio':    quote_count    / (content_len + 1),
        'ellipsis_count': ellipsis_count,
        'ellipsis_ratio': ellipsis_count / (title_len + 1),
        'excl_count':     excl_count,
        'excl_ratio':     excl_count     / (title_len + content_len + 1),
        'quest_count':    quest_count,
        'quest_ratio':    quest_count    / (title_len + content_len + 1),
        'title_len':      title_len,
        'content_len':    content_len,
    })


def main():
    # ── 데이터 로드 및 병합 ────────────────────────────────────────────────────
    print('[1/7] 데이터 로드 및 병합...', flush=True)
    tok_df = pd.read_csv(TOK_PATH, usecols=['id', 'tokens', 'label'])
    ref_df = pd.read_csv(REF_PATH, usecols=['id', 'title', 'content'])
    ref_df['id'] = ref_df['id'].astype('Int64')
    tok_df['id'] = tok_df['id'].astype('Int64')

    df = tok_df.merge(ref_df, on='id', how='inner')
    df['tokens']  = df['tokens'].fillna('')
    df['title']   = df['title'].fillna('')
    df['content'] = df['content'].fillna('')
    y = df['label'].astype(int)
    print(f'     병합 완료: {len(df):,}건 | 가짜: {(y==1).sum():,} / 진짜: {(y==0).sum():,}', flush=True)

    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)
    print(f'     Train: {len(train_df):,}건  Test: {len(test_df):,}건', flush=True)

    # ── 단어 TF-IDF (Okt 캐시 재사용, 재토크나이징 없음) ───────────────────────
    print('[2/7] 단어 TF-IDF (Okt 캐시 기반, 최대 60k)...', flush=True)
    t0 = time.time()
    tfidf_word = TfidfVectorizer(
        token_pattern=r'\S+',
        ngram_range=(1, 2), max_features=60_000, sublinear_tf=True, min_df=3,
    )
    X_tr_word = tfidf_word.fit_transform(train_df['tokens'])
    X_te_word = tfidf_word.transform(test_df['tokens'])
    print(f'     단어 피처: {X_tr_word.shape[1]:,}  ({time.time()-t0:.0f}s)', flush=True)

    # ── 문자 n-gram TF-IDF (제목만, 본문 제외 → 빠름) ─────────────────────────
    print('[3/7] 문자 n-gram TF-IDF (char_wb 2~4, 제목만, 최대 20k)...', flush=True)
    t0 = time.time()
    tr_char_text = train_df['title']
    te_char_text = test_df['title']
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=20_000, sublinear_tf=True, min_df=3,
    )
    X_tr_char = tfidf_char.fit_transform(tr_char_text)
    X_te_char = tfidf_char.transform(te_char_text)
    print(f'     문자 피처: {X_tr_char.shape[1]:,}  ({time.time()-t0:.0f}s)', flush=True)

    X_tr_comb = hstack([X_tr_word, X_tr_char]).tocsr()
    X_te_comb = hstack([X_te_word, X_te_char]).tocsr()
    print(f'     합산 피처: {X_tr_comb.shape[1]:,}', flush=True)

    # ── Chi2 → 30,000개 ────────────────────────────────────────────────────────
    print('[4/7] Chi2 피처 선택 (→ 30,000개)...', flush=True)
    t0 = time.time()
    selector = SelectKBest(chi2, k=30_000)
    X_tr_sel = selector.fit_transform(X_tr_comb, y_train)
    X_te_sel = selector.transform(X_te_comb)
    print(f'     완료 ({time.time()-t0:.0f}s)', flush=True)

    # ── 메타 피처 12개 + StandardScaler ────────────────────────────────────────
    print('[5/7] 메타 피처 12개 + StandardScaler...', flush=True)
    scaler        = StandardScaler()
    train_meta_sc = scaler.fit_transform(extract_meta_features(train_df))
    test_meta_sc  = scaler.transform(extract_meta_features(test_df))

    X_train_final = hstack([X_tr_sel, csr_matrix(train_meta_sc)]).tocsr()
    X_test_final  = hstack([X_te_sel, csr_matrix(test_meta_sc)]).tocsr()
    print(f'     최종 피처: {X_train_final.shape[1]:,} (30,000 Chi2 + 12 메타)', flush=True)

    # ── 5-fold CV C 튜닝 ───────────────────────────────────────────────────────
    print('[6/7] 5-fold StratifiedKFold CV (C 후보: 0.1, 0.5, 1.0, 2.0, 5.0)...', flush=True)
    C_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_cv_f1, best_C = 0.0, 1.0

    for C_val in C_candidates:
        cv_f1s = []
        for tr_idx, val_idx in skf.split(X_train_final, y_train):
            X_tr, X_val   = X_train_final[tr_idx], X_train_final[val_idx]
            y_tr_, y_val_ = y_train.iloc[tr_idx],  y_train.iloc[val_idx]
            m = LogisticRegression(
                C=C_val, l1_ratio=0, solver='saga',
                tol=1e-3, max_iter=3000, random_state=42,
            )
            m.fit(X_tr, y_tr_)
            _, _, f1, _ = precision_recall_fscore_support(
                y_val_, m.predict(X_val), average='macro', zero_division=0
            )
            cv_f1s.append(f1)
        mean_f1 = float(np.mean(cv_f1s))
        marker  = ' <-- best' if mean_f1 > best_cv_f1 else ''
        print(f'     C={C_val:<5} | 검증 Macro F1: {mean_f1:.4f}{marker}', flush=True)
        if mean_f1 > best_cv_f1:
            best_cv_f1, best_C = mean_f1, C_val

    # ── 최종 훈련 + 평가 ───────────────────────────────────────────────────────
    print(f'\n[7/7] 최종 훈련 (C={best_C})...', flush=True)
    t0 = time.time()
    final_model = LogisticRegression(
        C=best_C, l1_ratio=0, solver='saga',
        tol=1e-3, max_iter=3000, random_state=42,
    )
    final_model.fit(X_train_final, y_train)
    print(f'     학습 완료: {time.time()-t0:.0f}s', flush=True)

    probs = final_model.predict_proba(X_test_final)[:, 1]
    preds = (probs >= 0.5).astype(int)
    _, _, macro_f1,  _ = precision_recall_fscore_support(y_test, preds, average='macro',    zero_division=0)
    _, _, weight_f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())

    print(f'\n[결과] (테스트셋 {len(test_df):,}건)')
    print(f'  최적 C       : {best_C}')
    print(f'  Macro F1     : {macro_f1:.4f}  ({macro_f1*100:.2f}%)')
    print(f'  Weighted F1  : {weight_f1:.4f}  ({weight_f1*100:.2f}%)')
    print(f'  오탐 (FP)    : {fp}건  ({fp/((y_test==0).sum())*100:.1f}%)')
    print(f'  미탐 (FN)    : {fn}건  ({fn/((y_test==1).sum())*100:.1f}%)')
    print()
    print(classification_report(y_test, preds, target_names=['진짜(0)', '가짜(1)'], zero_division=0))

    # ── 5개 컴포넌트 저장 ──────────────────────────────────────────────────────
    pfx = os.path.join(SAVE_DIR, 'logistic_v4_3')
    joblib.dump(tfidf_word,  pfx + '_vec_word.pkl')
    joblib.dump(tfidf_char,  pfx + '_vec_char.pkl')
    joblib.dump(selector,    pfx + '_selector.pkl')
    joblib.dump(scaler,      pfx + '_scaler.pkl')
    joblib.dump(final_model, pfx + '_model.pkl')
    print(f'저장 완료: models/logistic/logistic_v4_3_*.pkl (5개 컴포넌트)')
    print(f'MACRO_F1={macro_f1:.4f}')


if __name__ == '__main__':
    main()
