"""
로지스틱 회귀 학습 스크립트


실행:
  python src/logistic.py          # 영어 + 한국어 둘 다
  python src/logistic.py --lang korean
  python src/logistic.py --lang english
"""

import os
import sys
import time
import argparse
import numpy as np
import joblib
from scipy import sparse
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_DIR    = os.path.join(BASE_DIR, 'data', 'vector')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 벡터 파일 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_vectors(lang: str):
    """
    data/vector/ 에서 TF-IDF 행렬, 레이블, DVL 플래그를 로드하고
    hstack([tfidf, dvl_flags]) 으로 결합하여 반환한다.
    """
    prefix = lang  # 'korean' or 'english'

    tfidf_path = os.path.join(VEC_DIR, f'{prefix}_tfidf.npz')
    label_path = os.path.join(VEC_DIR, f'{prefix}_labels.npy')
    dvl_path   = os.path.join(VEC_DIR, f'{prefix}_dvl_flags.npy')

    for p in [tfidf_path, label_path, dvl_path]:
        if not os.path.exists(p):
            print(f'[ERROR] 파일 없음: {p}')
            print('  먼저 python src/vectorize.py 를 실행하세요.')
            sys.exit(1)

    print(f'[{lang}] 벡터 파일 로드 중...')
    X_tfidf = sparse.load_npz(tfidf_path)
    y       = np.load(label_path)
    dvl     = np.load(dvl_path)

    # TF-IDF (n x 50000) + DVL 플래그 (n x 5) 결합
    X = hstack([X_tfidf, sparse.csr_matrix(dvl)])

    n_fake = int(y.sum())
    print(f'  행렬: {X.shape} | 가짜: {n_fake:,} | 진짜: {len(y)-n_fake:,}')
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 학습 및 저장
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(lang: str):
    print('\n' + '=' * 60)
    print(f'[{lang.upper()}] 로지스틱 회귀 학습')
    print('=' * 60)

    X, y = load_vectors(lang)

    # 8:2 분할 (레이블 비율 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  train: {X_train.shape[0]:,}건 | test: {X_test.shape[0]:,}건')

    # 학습
    print('  LogisticRegression 학습 중...')
    t0 = time.time()
    model = LogisticRegression(
        class_weight='balanced',   # 클래스 불균형 자동 보정
        max_iter=1000,
        solver='saga',             # 대용량 희소 행렬에 적합
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f'  완료: {time.time()-t0:.1f}초')

    # 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')

    print(f'\n  Accuracy : {acc:.4f}')
    print(f'  Macro-F1 : {f1:.4f}')
    print()
    print(classification_report(y_test, y_pred, target_names=['진짜(0)', '가짜(1)']))

    # 핵심 키워드 (TF-IDF 50000개 중 가짜뉴스 가중치 상위 10개)
    coef = model.coef_[0][:50000]   # DVL 플래그 5개 제외
    vec_path = os.path.join(VEC_DIR, f'{lang}_vectorizer.pkl')
    if os.path.exists(vec_path):
        import pickle
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
        feature_names = np.array(vectorizer.get_feature_names_out())
        top_idx = np.argsort(coef)[-10:][::-1]
        print('  [가짜뉴스 핵심 키워드 Top 10]')
        for i in top_idx:
            print(f'    {feature_names[i]:<20} 가중치: {coef[i]:.4f}')

    # 저장
    model_path = os.path.join(MODELS_DIR, f'{lang}_logistic.joblib')
    joblib.dump(model, model_path)
    saved_mb = os.path.getsize(model_path) / 1024**2
    print(f'\n  저장: src/models/{lang}_logistic.joblib ({saved_mb:.1f} MB)')

    return f1


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lang', choices=['korean', 'english', 'both'], default='both',
        help='학습할 언어 선택 (기본: both)'
    )
    args = parser.parse_args()

    langs = ['korean', 'english'] if args.lang == 'both' else [args.lang]

    results = {}
    for lang in langs:
        results[lang] = train_and_save(lang)

    print('\n' + '=' * 60)
    print('학습 완료 요약')
    print('=' * 60)
    for lang, f1 in results.items():
        print(f'  {lang:<10} Macro-F1: {f1:.4f}  ->  src/models/{lang}_logistic.joblib')
