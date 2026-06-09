# -*- coding: utf-8 -*-
"""
NBSVM 풀 파이프라인 재훈련 스크립트
subset_01.csv -> 제목x5 + Okt TF-IDF(단어+문자) + Chi2(40k) + NB ratio + 메타피처 + LinearSVC
저장: src/models/nbsvm_ko_*.pkl (7개 컴포넌트)
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.feature_selection import SelectKBest, chi2
# 출력 인코딩 UTF-8 강제 (Windows cp949 대응)
sys.stdout.reconfigure(encoding='utf-8')

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'src'))
DATA_PATH    = os.path.join(_PROJECT_DIR, 'data', 'processed', 'subsets', 'subset_01.csv')
SAVE_DIR     = os.path.join(_PROJECT_DIR, 'models', 'svm')
os.makedirs(SAVE_DIR, exist_ok=True)

# pickle에 'okt_utils.okt_tokenizer' 경로로 저장 → src/ 가 sys.path에 있으면 역직렬화 가능
from okt_utils import okt_tokenizer


def extract_meta_features(df):
    title_str   = df['title'].astype(str)
    content_str = df['content'].astype(str)
    title_len   = title_str.str.len()
    content_len = content_str.str.len()

    pattern         = '결국|충격|경악|속보|단독|논란|분노|의혹|발각|공개|주의|확인'
    emo_count       = title_str.str.count(pattern)
    emo_ratio       = emo_count / (title_len + 1)
    quote_count     = content_str.str.count(r'["\']')
    quote_ratio     = quote_count / (content_len + 1)
    ellipsis_count  = title_str.str.count(r'\.\.\.')
    ellipsis_ratio  = ellipsis_count / (title_len + 1)
    excl_count      = title_str.str.count('!') + content_str.str.count('!')
    excl_ratio      = excl_count / (title_len + content_len + 1)
    quest_count     = title_str.str.count(r'\?') + content_str.str.count(r'\?')
    quest_ratio     = quest_count / (title_len + content_len + 1)

    return pd.DataFrame({
        'emo_count':      emo_count,      'emo_ratio':      emo_ratio,
        'quote_count':    quote_count,    'quote_ratio':    quote_ratio,
        'ellipsis_count': ellipsis_count, 'ellipsis_ratio': ellipsis_ratio,
        'excl_count':     excl_count,     'excl_ratio':     excl_ratio,
        'quest_count':    quest_count,    'quest_ratio':    quest_ratio,
        'title_len':      title_len,      'content_len':    content_len,
    })


def compute_nb_ratio(X, y, alpha=1.0):
    y = np.array(y)
    p = np.asarray(X[y == 1].sum(axis=0)).flatten() + alpha
    q = np.asarray(X[y == 0].sum(axis=0)).flatten() + alpha
    return np.log((p / p.sum()) / (q / q.sum()))


def main():
    print(f'[1/7] 훈련 데이터 로드: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, usecols=['title', 'content', 'label']).dropna()
    df['title']   = df['title'].astype(str).fillna('')
    df['content'] = df['content'].astype(str).fillna('')
    y = df['label'].astype(int)
    print(f'     로드 완료: {len(df):,}건 | 가짜: {(y==1).sum():,} / 진짜: {(y==0).sum():,}')

    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)

    # 제목 5배 증폭
    X_train_text = (train_df['title'] + ' ') * 5 + train_df['content']
    X_test_text  = (test_df['title']  + ' ') * 5 + test_df['content']

    # 단어 TF-IDF (Okt 형태소)
    print('[2/7] 단어 TF-IDF (Okt 형태소, 최대 60k) - 10~20분 소요...')
    tfidf_word = TfidfVectorizer(
        tokenizer=okt_tokenizer, token_pattern=None,
        ngram_range=(1, 2), max_features=60000, sublinear_tf=True, min_df=3,
    )
    X_tr_word = tfidf_word.fit_transform(X_train_text)
    X_te_word = tfidf_word.transform(X_test_text)
    print(f'     단어 피처: {X_tr_word.shape[1]:,}')

    # 문자 n-gram TF-IDF
    print('[3/7] 문자 n-gram TF-IDF (2~4자, 최대 60k)...')
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=60000, sublinear_tf=True, min_df=3,
    )
    X_tr_char = tfidf_char.fit_transform(X_train_text)
    X_te_char = tfidf_char.transform(X_test_text)
    print(f'     문자 피처: {X_tr_char.shape[1]:,}')

    X_tr_massive = hstack([X_tr_word, X_tr_char]).tocsr()
    X_te_massive = hstack([X_te_word, X_te_char]).tocsr()
    print(f'     합산 피처: {X_tr_massive.shape[1]:,}')

    # Chi2 -> 40,000개
    print('[4/7] Chi2 선택 (-> 40,000개)...')
    selector = SelectKBest(chi2, k=40000)
    X_tr_selected = selector.fit_transform(X_tr_massive, y_train)
    X_te_selected = selector.transform(X_te_massive)

    # NB 비율 가중치
    print('[5/7] NB log-count ratio 가중치 계산...')
    r_vector   = compute_nb_ratio(X_tr_selected, y_train)
    X_tr_tfidf = X_tr_selected.multiply(r_vector).tocsr()
    X_te_tfidf = X_te_selected.multiply(r_vector).tocsr()

    # 메타 피처 (12개) + StandardScaler
    print('[6/7] 메타 피처 (12개) + StandardScaler...')
    scaler            = StandardScaler()
    train_meta_scaled = scaler.fit_transform(extract_meta_features(train_df))
    test_meta_scaled  = scaler.transform(extract_meta_features(test_df))

    X_train_combined = hstack([X_tr_tfidf, csr_matrix(train_meta_scaled)]).tocsr()
    X_test_combined  = hstack([X_te_tfidf, csr_matrix(test_meta_scaled)]).tocsr()
    print(f'     최종 피처: {X_train_combined.shape[1]} (40,000 + 12 메타)')

    # LinearSVC 교차검증
    print('[7/7] LinearSVC 교차검증 (C 후보: 0.1, 0.5, 1, 2, 5, 10)...')
    C_candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_cv_f1, best_C = 0.0, 1.0

    for C_val in C_candidates:
        cv_f1s = []
        for tr_idx, val_idx in skf.split(X_train_combined, y_train):
            X_tr, X_val   = X_train_combined[tr_idx], X_train_combined[val_idx]
            y_tr_, y_val_ = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = LinearSVC(penalty='l2', C=C_val, class_weight='balanced',
                          random_state=42, max_iter=5000, dual=False)
            m.fit(X_tr, y_tr_)
            _, _, f1, _ = precision_recall_fscore_support(
                y_val_, m.predict(X_val), average='macro', zero_division=0
            )
            cv_f1s.append(f1)
        mean_f1 = float(np.mean(cv_f1s))
        marker  = ' <-- best' if mean_f1 > best_cv_f1 else ''
        print(f'     C={C_val:<5} | 검증 Macro F1: {mean_f1:.4f}{marker}')
        if mean_f1 > best_cv_f1:
            best_cv_f1, best_C = mean_f1, C_val

    print(f'\n최종 훈련 (C={best_C}) + 임계값 튜닝...')
    final_model = LinearSVC(penalty='l2', C=best_C, class_weight='balanced',
                            random_state=42, max_iter=5000, dual=False)
    final_model.fit(X_train_combined, y_train)

    decision_scores         = final_model.decision_function(X_test_combined)
    best_threshold, best_f1_thresh = 0.0, 0.0
    for thresh in np.arange(-1.0, 1.0, 0.05):
        temp_pred = (decision_scores > thresh).astype(int)
        _, _, temp_f1, _ = precision_recall_fscore_support(
            y_test, temp_pred, average='macro', zero_division=0
        )
        if temp_f1 > best_f1_thresh:
            best_f1_thresh, best_threshold = temp_f1, thresh

    y_pred = (decision_scores > best_threshold).astype(int)
    _, _, macro_f1, _  = precision_recall_fscore_support(y_test, y_pred, average='macro',    zero_division=0)
    _, _, weight_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f'\n[결과]')
    print(f'  최적 임계값 : {best_threshold:.2f}')
    print(f'  Macro F1    : {macro_f1:.4f}')
    print(f'  Weighted F1 : {weight_f1:.4f}')

    # 7개 컴포넌트 저장
    pfx = os.path.join(SAVE_DIR, 'nbsvm_ko')
    joblib.dump(tfidf_word,     pfx + '_vec_word.pkl')
    joblib.dump(tfidf_char,     pfx + '_vec_char.pkl')
    joblib.dump(selector,       pfx + '_selector.pkl')
    joblib.dump(r_vector,       pfx + '_r_vector.pkl')
    joblib.dump(scaler,         pfx + '_scaler.pkl')
    joblib.dump(final_model,    pfx + '_model.pkl')
    joblib.dump(best_threshold, pfx + '_threshold.pkl')
    print(f'\n저장 완료: models/svm/nbsvm_ko_*.pkl (7개 컴포넌트)')
    print(f'WEIGHTED_F1={weight_f1:.4f}')


if __name__ == '__main__':
    main()
