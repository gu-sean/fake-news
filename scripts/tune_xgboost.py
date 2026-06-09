"""
Optuna로 한국어 XGBoost 하이퍼파라미터 튜닝
- 기존 벡터/셀렉터를 그대로 재사용 (재벡터화 없음)
- 목표: F1 macro 최대화
- 완료 후 최적 모델을 src/models/에 저장
"""

import json
import os
import sys
import time
import warnings
import joblib
import numpy as np
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    print("pip install xgboost")
    sys.exit(1)

# ── 경로 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VECTOR_DIR  = os.path.join(PROJECT_DIR, "data", "vector")
MODEL_DIR   = os.path.join(PROJECT_DIR, "models", "xgboost")

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_TRIALS     = 40   # 시도 횟수 (늘릴수록 정확하지만 느림)
K_FEATURES   = 15000  # K 스윕 결과 최적값

# ── 데이터 로드 & 분할 ────────────────────────────────────────────────────────
print("=" * 60)
print(" 데이터 로드 중...")
print("=" * 60)

X   = sparse.load_npz(os.path.join(VECTOR_DIR, "korean_tfidf.npz"))
y   = np.load(os.path.join(VECTOR_DIR, "korean_labels.npy")).astype(np.int8)
dvl = np.load(os.path.join(VECTOR_DIR, "korean_dvl_flags.npy")).astype(np.float32)
print(f"  TF-IDF: {X.shape}  진짜(0): {(y==0).sum():,}  가짜(1): {(y==1).sum():,}")

idx = np.arange(len(y))
idx_tr_full, idx_te = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
idx_tr, idx_vl = train_test_split(
    idx_tr_full, test_size=0.1, random_state=RANDOM_STATE, stratify=y[idx_tr_full])

# ── Chi2 피처 선택 (K 고정) ───────────────────────────────────────────────────
print(f"\n  Chi2 피처 선택 (K={K_FEATURES:,}) 중...")
t0 = time.time()
selector = SelectKBest(chi2, k=K_FEATURES)
Xtr_sel = selector.fit_transform(X[idx_tr], y[idx_tr])
Xvl_sel = selector.transform(X[idx_vl])
Xte_sel = selector.transform(X[idx_te])
print(f"  완료: {time.time()-t0:.1f}s")

Xtr = hstack([Xtr_sel, csr_matrix(dvl[idx_tr])], format='csr')
Xvl = hstack([Xvl_sel, csr_matrix(dvl[idx_vl])], format='csr')
Xte = hstack([Xte_sel, csr_matrix(dvl[idx_te])], format='csr')
ytr, yvl, yte = y[idx_tr], y[idx_vl], y[idx_te]

# ── 베이스라인 (기존 모델) ────────────────────────────────────────────────────
print("\n  베이스라인 모델 로드 중...")
base_f1 = 0.0
try:
    base_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model_ko.pkl"))
    base_pred  = base_model.predict(Xte)
    base_f1    = f1_score(yte, base_pred, average="macro", zero_division=0)
    print(f"  베이스라인 F1 macro: {base_f1:.4f}")
except Exception as e:
    print(f"  베이스라인 로드/예측 실패 ({e}) → base_f1=0.0 으로 설정")

# ── Optuna 목적함수 ───────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 600),
        "max_depth":          trial.suggest_int("max_depth", 3, 6),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 5.0),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 0.3, 0.9),
        "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
    }

    model = xgb.XGBClassifier(
        **params,
        eval_metric          = "logloss",
        early_stopping_rounds= 20,
        random_state         = RANDOM_STATE,
        n_jobs               = -1,
        verbosity            = 0,
        tree_method          = "hist",
        max_bin              = 64,
    )
    model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
    pred = model.predict(Xvl)
    return f1_score(yvl, pred, average="macro", zero_division=0)


# ── 튜닝 실행 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f" Optuna 튜닝 시작  ({N_TRIALS} trials)")
print(f"{'='*60}")
t_start = time.time()

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True,
               catch=(Exception,))

print(f"\n  튜닝 완료: {(time.time()-t_start)/60:.1f}분")
print(f"  최적 Val F1: {study.best_value:.4f}")
print(f"\n  최적 하이퍼파라미터:")
for k, v in study.best_params.items():
    print(f"    {k:25s}: {v}")

# ── 최적 파라미터로 테스트셋 최종 평가 ───────────────────────────────────────
print(f"\n{'='*60}")
print(" 최적 파라미터로 최종 평가 (테스트셋)")
print(f"{'='*60}")

best_params = study.best_params
best_model  = xgb.XGBClassifier(
    **best_params,
    eval_metric          = "logloss",
    early_stopping_rounds= 20,
    random_state         = RANDOM_STATE,
    n_jobs               = -1,
    verbosity            = 0,
    tree_method          = "hist",
    max_bin              = 64,
)
best_model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
ypred = best_model.predict(Xte)
tuned_f1 = f1_score(yte, ypred, average="macro", zero_division=0)

print(f"\n  베이스라인  F1 macro: {base_f1:.4f}")
print(f"  튜닝 후     F1 macro: {tuned_f1:.4f}  ({tuned_f1 - base_f1:+.4f})")
print()
print(classification_report(yte, ypred, target_names=["진짜(0)", "가짜(1)"], zero_division=0))

cm = confusion_matrix(yte, ypred)
print("  [혼동행렬]")
print(f"                  예측-진짜  예측-가짜")
print(f"  실제-진짜(0):  {cm[0,0]:8,}  {cm[0,1]:8,}   오분류율 {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}%")
print(f"  실제-가짜(1):  {cm[1,0]:8,}  {cm[1,1]:8,}   오분류율 {cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}%")

# ── 모델 저장 ─────────────────────────────────────────────────────────────────
if tuned_f1 > base_f1:
    save_path_model    = os.path.join(MODEL_DIR, "xgb_model_ko.pkl")
    save_path_selector = os.path.join(MODEL_DIR, "xgb_selector_ko.pkl")
    save_path_config   = os.path.join(MODEL_DIR, "xgb_config_ko.json")
    joblib.dump(best_model, save_path_model)
    joblib.dump(selector,   save_path_selector)
    with open(save_path_config, 'w') as f:
        json.dump({'use_dvl': True, 'k': K_FEATURES, 'use_embeddings': False}, f)
    print(f"\n  모델 저장 완료 (F1 {base_f1:.4f} → {tuned_f1:.4f})")
    print(f"    {save_path_model}")
    print(f"    {save_path_selector}")
    print(f"    {save_path_config}")
else:
    print(f"\n  기존 모델이 더 좋아서 저장하지 않습니다. (기존 {base_f1:.4f} >= 튜닝 {tuned_f1:.4f})")
