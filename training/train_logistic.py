# -*- coding: utf-8 -*-
"""
Logistic V4.2 재훈련 + Pipeline 저장

- 토크나이즈 캐시(unified_news_tokenized.csv)에서 X·y를 동시 생성해 정렬 보장
- sklearn Pipeline(vectorizer + model)으로 저장 → 이후 vectorizer 교체에도 안전
- 최적 파라미터 C=1.0 (logistic_V4_2_log.txt 그리드서치 결과 기준)

사용법:
  python src/Logistic_V4_2/logistic_V4_2_train_pipeline.py
"""
import os, sys, time
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.dirname(SCRIPT_DIR)
VEC_DIR    = os.path.join(BASE, 'data', 'vector')
CACHE_PATH = os.path.join(BASE, 'data', 'processed', 'unified_news_tokenized.csv')
VEC_PATH   = os.path.join(VEC_DIR, 'korean_vectorizer.pkl')
OUT_PATH   = os.path.join(BASE, 'models', 'logistic', 'logistic_v4_2_pipeline.pkl')

print('데이터 로드 중...')
cache      = pd.read_csv(CACHE_PATH)
vectorizer = joblib.load(VEC_PATH)
X = vectorizer.transform(cache['tokens'].fillna(''))
y = cache['label'].values.astype(np.int8)
print(f'X: {X.shape}, y: {y.shape} | 가짜: {y.sum():,} / 진짜: {(y==0).sum():,}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}')

print('\nLogisticRegression 학습 시작 (C=1.0, L1/saga)...')
t0  = time.time()
clf = LogisticRegression(
    C=1.0, l1_ratio=1.0, solver='saga',
    tol=3e-4, max_iter=3000, random_state=42,
)
clf.fit(X_train, y_train)
print(f'학습 완료: {time.time()-t0:.1f}s')

preds    = clf.predict(X_test)
macro_f1 = f1_score(y_test, preds, average='macro')
acc      = accuracy_score(y_test, preds)
print(f'\nMacro F1 : {macro_f1*100:.2f}%')
print(f'Accuracy : {acc*100:.2f}%')

pipeline = Pipeline([('tfidf', vectorizer), ('clf', clf)])
joblib.dump(pipeline, OUT_PATH)
print(f'\n저장 완료: {OUT_PATH}')
