# -*- coding: utf-8 -*-
"""
Logistic / SVM / Ensemble 임계값 최적화
미탐(가짜→진짜) 최소화 + F1 최대화 기준
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL  = os.path.join(BASE, 'models')
VECTOR = os.path.join(BASE, 'data', 'vector')
sys.path.insert(0, os.path.join(BASE, 'src'))

DVL_COLS = ['stat_distortion','causal_error','emotional_provocation','source_lack','img_mismatch']

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print('데이터 로드...')
df = pd.read_csv(
    os.path.join(BASE, 'data', 'processed', 'subsets', 'subset_01.csv'),
    usecols=['title', 'content', 'label'] + DVL_COLS,
).dropna(subset=['title','content','label'])
df['title']   = df['title'].astype(str).fillna('')
df['content'] = df['content'].astype(str).fillna('')
y = df['label'].astype(int)

_, test_df, _, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)
test_df = test_df.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
print(f'테스트셋: {len(test_df):,}건 | 가짜: {(y_test==1).sum()} / 진짜: {(y_test==0).sum()}')

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def search(name, probs, y_true):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    print(f'  {"임계값":>6}  {"F1 macro":>9}  {"오탐율":>7}  {"미탐율":>7}  {"오탐건":>6}  {"미탐건":>6}')
    print(f'  {"-"*56}')
    best = None
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        mac   = f1_score(y_true, preds, average='macro')
        fp    = ((preds==1) & (y_true==0)).sum()   # 진짜→가짜 (오탐)
        fn    = ((preds==0) & (y_true==1)).sum()   # 가짜→진짜 (미탐)
        n_real = (y_true==0).sum()
        n_fake = (y_true==1).sum()
        mark  = ' ★' if best is None or mac > best[0] else ''
        print(f'  {t:>6.2f}  {mac*100:>8.2f}%  {fp/n_real*100:>6.1f}%  {fn/n_fake*100:>6.1f}%  {fp:>6}  {fn:>6}{mark}')
        if best is None or mac > best[0]:
            best = (mac, t, fp, fn)
    print(f'\n  → 최적 임계값: {best[1]}  F1: {best[0]*100:.2f}%  미탐: {best[3]}건')
    return best[1], probs


# ── 1. Logistic ────────────────────────────────────────────────────────────────
print('\n[1/3] Logistic 확률 계산...')
from okt_utils import okt_tokenizer
lr_pipeline = joblib.load(os.path.join(BASE, 'models', 'logistic', 'logistic_v4_2_pipeline.pkl'))
raw_texts   = (test_df['title'] + ' ' + test_df['content']).tolist()
print(f'  Okt 토크나이징 {len(raw_texts):,}건...')
texts_lr  = [' '.join(okt_tokenizer(t)) for t in raw_texts]
probs_lr  = lr_pipeline.predict_proba(texts_lr)[:, 1]
best_t_lr, _ = search('Logistic Regression (Pipeline+Okt)', probs_lr, y_test)

# ── 2. SVM ─────────────────────────────────────────────────────────────────────
print('\n[2/3] SVM 확률 계산...')
import __main__
__main__.okt_tokenizer = okt_tokenizer

pfx       = os.path.join(MODEL, 'svm', 'nbsvm_ko')
vec_word  = joblib.load(pfx + '_vec_word.pkl')
vec_char  = joblib.load(pfx + '_vec_char.pkl')
selector  = joblib.load(pfx + '_selector.pkl')
r_vector  = joblib.load(pfx + '_r_vector.pkl')
scaler    = joblib.load(pfx + '_scaler.pkl')
svm_model = joblib.load(pfx + '_model.pkl')
threshold_svm = joblib.load(pfx + '_threshold.pkl')

texts_svm = (test_df['title'] + ' ') * 5 + test_df['content']
X_w  = vec_word.transform(texts_svm)
X_c  = vec_char.transform(texts_svm)
X_m  = hstack([X_w, X_c]).tocsr()
X_s  = selector.transform(X_m)
X_nb = X_s.multiply(r_vector).tocsr()
t, c = test_df['title'].astype(str), test_df['content'].astype(str)
tl, cl = t.str.len(), c.str.len()
pat = '결국|충격|경악|속보|단독|논란|분노|의혹|발각|공개|주의|확인'
ec  = t.str.count(pat); qc = c.str.count(r'["\']')
lc  = t.str.count(r'\.\.\.'); exc = t.str.count('!')+c.str.count('!')
qsc = t.str.count(r'\?')+c.str.count(r'\?')
meta = pd.DataFrame({
    'emo_count': ec, 'emo_ratio': ec/(tl+1),
    'quote_count': qc, 'quote_ratio': qc/(cl+1),
    'ellipsis_count': lc, 'ellipsis_ratio': lc/(tl+1),
    'excl_count': exc, 'excl_ratio': exc/(tl+cl+1),
    'quest_count': qsc, 'quest_ratio': qsc/(tl+cl+1),
    'title_len': tl, 'content_len': cl,
})
meta_sc   = scaler.transform(meta)
X_comb    = hstack([X_nb, csr_matrix(meta_sc)]).tocsr()
scores    = svm_model.decision_function(X_comb)
probs_svm = 1.0 / (1.0 + np.exp(-scores))
best_t_svm, _ = search('SVM (NBSVM)', probs_svm, y_test)

# ── 3. Ensemble ────────────────────────────────────────────────────────────────
print('\n[3/3] 앙상블 확률 계산...')
nb_model = joblib.load(os.path.join(BASE, 'models', 'naive_bayes', 'best_model_naive_V1.pkl'))
nb_vec   = joblib.load(os.path.join(VECTOR, 'korean_vectorizer.pkl'))
X_nb2    = nb_vec.transform(raw_texts)
probs_nb = nb_model.predict_proba(X_nb2)[:, 1]

W = {'logistic': 0.7105, 'nb': 0.6273, 'svm': 0.7301}
total_w   = sum(W.values())
probs_ens = (probs_lr * W['logistic'] + probs_nb * W['nb'] + probs_svm * W['svm']) / total_w
best_t_ens, _ = search('Ensemble (LR+NB+SVM)', probs_ens, y_test)

# ── 최종 요약 ──────────────────────────────────────────────────────────────────
print(f'\n\n{"="*60}')
print('  최적 임계값 요약')
print(f'{"="*60}')
for name, t, probs in [('Logistic', best_t_lr, probs_lr),
                        ('SVM',      best_t_svm, probs_svm),
                        ('Ensemble', best_t_ens, probs_ens)]:
    preds_new = (probs >= t).astype(int)
    preds_old = (probs >= 0.5).astype(int)
    f1_new = f1_score(y_test, preds_new, average='macro')
    f1_old = f1_score(y_test, preds_old, average='macro')
    fn_new = ((preds_new==0) & (y_test==1)).sum()
    fn_old = ((preds_old==0) & (y_test==1)).sum()
    print(f'  {name:<10} 임계값 0.50→{t:.2f}  F1: {f1_old*100:.2f}%→{f1_new*100:.2f}% (+{(f1_new-f1_old)*100:.2f}%)  미탐: {fn_old}→{fn_new}건 ({fn_old-fn_new:+})')
print(f'{"="*60}')
