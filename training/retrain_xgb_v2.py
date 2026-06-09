# -*- coding: utf-8 -*-
"""
XGBoost V2 재훈련 (캐시 기반 + Chi2 특성선택)

unified_news_tokenized.csv + korean_vectorizer.pkl → Chi2 20k 선택 → XGBoost V2

저장:
  models/xgboost/best_model_xgboost_V2.pkl
  models/xgboost/xgb_model_ko.pkl        (eval/inference 공용)
  models/xgboost/xgb_selector_ko.pkl     (Chi2 선택기)
  models/xgboost/performance_xgboost_V2.txt
  models/xgboost/xgb_config_ko.json      (use_selector: true)
"""
import os
import sys
import time
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

CACHE_PATH  = os.path.join(PROJECT_DIR, 'data', 'processed', 'unified_news_tokenized.csv')
VEC_PATH    = os.path.join(PROJECT_DIR, 'data', 'vector', 'korean_vectorizer.pkl')
OUT_DIR     = os.path.join(PROJECT_DIR, 'models', 'xgboost')
MODEL_V2    = os.path.join(OUT_DIR, 'best_model_xgboost_V2.pkl')
MODEL_KO    = os.path.join(OUT_DIR, 'xgb_model_ko.pkl')
SELECTOR    = os.path.join(OUT_DIR, 'xgb_selector_ko.pkl')
CONFIG      = os.path.join(OUT_DIR, 'xgb_config_ko.json')
METRICS     = os.path.join(OUT_DIR, 'performance_xgboost_V2.txt')

CHI2_K = 20000  # 50k → 20k (메모리 절약)

PARAMS = dict(
    max_depth         = 9,
    learning_rate     = 0.03,
    n_estimators      = 1500,
    subsample         = 0.8,
    colsample_bytree  = 0.3,   # 20k × 0.3 = 6k features/tree
    min_child_weight  = 5,
    reg_alpha         = 1.0,
    reg_lambda        = 1.0,
    objective         = 'binary:logistic',
    eval_metric       = 'logloss',
    tree_method       = 'hist',
    random_state      = 42,
    n_jobs            = -1,
    early_stopping_rounds = 50,
    verbosity         = 0,
)

print('캐시 로드 중...', flush=True)
cache = pd.read_csv(CACHE_PATH)
print(f'  캐시: {len(cache):,}행 | 가짜: {cache["label"].sum():,} / 진짜: {(cache["label"]==0).sum():,}', flush=True)

print('vectorizer 변환 중...', flush=True)
vectorizer = joblib.load(VEC_PATH)
X = vectorizer.transform(cache['tokens'].fillna(''))
y = cache['label'].values.astype(np.int8)
print(f'  X: {X.shape}', flush=True)

# train(72%) / val(8%) / test(20%) 분할
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tr_full, y_tr_full, test_size=0.10, random_state=42, stratify=y_tr_full
)
print(f'  train: {X_train.shape[0]:,} / val: {X_val.shape[0]:,} / test: {X_test.shape[0]:,}', flush=True)

print(f'\nChi2 특성선택 (50k → {CHI2_K:,})...', flush=True)
t0 = time.time()
selector = SelectKBest(chi2, k=CHI2_K)
X_train_s = selector.fit_transform(X_train, y_train)
X_val_s   = selector.transform(X_val)
X_test_s  = selector.transform(X_test)
print(f'  완료: {time.time()-t0:.1f}s | 선택 후: {X_train_s.shape}', flush=True)

print('\nXGBoost V2 훈련 시작...', flush=True)
t0 = time.time()
model = XGBClassifier(**PARAMS)
model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False,
)
elapsed = time.time() - t0
print(f'  완료: {elapsed:.1f}s | 최적 트리: {model.best_iteration}', flush=True)

preds = model.predict(X_test_s)
mac   = f1_score(y_test, preds, average='macro')
acc   = accuracy_score(y_test, preds)
print(f'\n[결과]', flush=True)
print(f'  Macro F1 : {mac*100:.2f}%', flush=True)
print(f'  Accuracy : {acc*100:.2f}%', flush=True)
print(classification_report(y_test, preds, target_names=['진짜(0)', '가짜(1)'], zero_division=0))

joblib.dump(model,    MODEL_V2)
joblib.dump(model,    MODEL_KO)
joblib.dump(selector, SELECTOR)
with open(CONFIG, 'w', encoding='utf-8') as f:
    json.dump({'model_version': 'V2', 'use_dvl': False, 'use_selector': True, 'use_embeddings': False}, f)
with open(METRICS, 'w', encoding='utf-8') as f:
    f.write(f'Macro F1: {mac:.4f}\nAccuracy: {acc:.4f}\nBest iteration: {model.best_iteration}\n')

print(f'\n저장 완료:', flush=True)
print(f'  {MODEL_V2}', flush=True)
print(f'  {MODEL_KO}', flush=True)
print(f'  {SELECTOR}', flush=True)
print(f'  {CONFIG}', flush=True)
