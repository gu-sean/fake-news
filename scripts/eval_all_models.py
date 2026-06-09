# -*- coding: utf-8 -*-
"""
모델 성능 종합 평가
- [1~5] 4개 모델 + 앙상블 공정 비교  (동일 테스트셋 6,000건)
- [6]   DVL 피처 포함/제외 XGBoost 성능 비교
- [7]   XGBoost 임계값(threshold) 최적화
"""
import sys, os, warnings, time
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL  = os.path.join(BASE, 'models')
VECTOR = os.path.join(BASE, 'data', 'vector')
sys.path.insert(0, os.path.join(BASE, 'src'))

DVL_COLS = ['stat_distortion', 'causal_error', 'emotional_provocation',
            'source_lack', 'img_mismatch']

# ── 데이터 로드 & 분할 ──────────────────────────────────────────────────────────
print('데이터 로드...')
df = pd.read_csv(
    os.path.join(BASE, 'data', 'processed', 'subsets', 'subset_01.csv'),
    usecols=['id', 'title', 'content', 'label'] + DVL_COLS,
).dropna(subset=['title', 'content', 'label'])
df['title']   = df['title'].astype(str).fillna('')
df['content'] = df['content'].astype(str).fillna('')
y = df['label'].astype(int)

_, test_df, _, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)
test_df = test_df.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
print(f'테스트셋: {len(test_df):,}건 | 가짜: {(y_test==1).sum()} / 진짜: {(y_test==0).sum()}')


def report(name, y_true, y_pred):
    mac = f1_score(y_true, y_pred, average='macro')
    wgt = f1_score(y_true, y_pred, average='weighted')
    print(f'\n{"="*55}')
    print(f'  {name}')
    print(f'{"="*55}')
    print(f'  Macro F1    : {mac:.4f}  ({mac*100:.2f}%)')
    print(f'  Weighted F1 : {wgt:.4f}  ({wgt*100:.2f}%)')
    print(classification_report(y_true, y_pred,
          target_names=['진짜(0)', '가짜(1)'], zero_division=0))
    return mac, wgt


results = {}
probs_lr = probs_nb = probs_svm = None
xgb_model = xgb_vec = xgb_selector = None

# ── 1. XGBoost ────────────────────────────────────────────────────────────────
print('\n[1/5] XGBoost 평가...')
try:
    import json as _json
    xgb_cfg_path = os.path.join(MODEL, 'xgboost', 'xgb_config_ko.json')
    _xgb_cfg = _json.load(open(xgb_cfg_path)) if os.path.exists(xgb_cfg_path) else {}
    _use_sel = _xgb_cfg.get('use_selector', True)
    _use_dvl = _xgb_cfg.get('use_dvl', False)

    xgb_model = joblib.load(os.path.join(MODEL,  'xgboost', 'xgb_model_ko.pkl'))
    xgb_vec   = joblib.load(os.path.join(VECTOR, 'korean_vectorizer.pkl'))

    # korean_vectorizer는 Okt 형태소 토큰으로 피팅 → 토크나이징 필요
    from okt_utils import okt_tokenizer as _okt_fn
    tok_cache_xgb = pd.read_csv(
        os.path.join(BASE, 'data', 'processed', 'unified_news_tokenized.csv'),
        usecols=['id', 'tokens'],
    )
    test_xgb = test_df.merge(tok_cache_xgb, on='id', how='left')
    test_xgb['tokens'] = test_xgb['tokens'].fillna('')
    _matched_xgb = (test_xgb['tokens'] != '').sum()
    print(f'  tokens 캐시 매칭: {_matched_xgb}/{len(test_xgb)}건')
    if _matched_xgb < len(test_xgb) * 0.5:
        print(f'  캐시 매칭 부족 → Okt 토크나이징...')
        raw_xgb = (test_xgb['title'] + ' ' + test_xgb['content']).tolist()
        test_xgb['tokens'] = [' '.join(_okt_fn(t)) for t in raw_xgb]

    X_xgb = xgb_vec.transform(test_xgb['tokens'])

    if _use_sel:
        xgb_selector = joblib.load(os.path.join(MODEL, 'xgboost', 'xgb_selector_ko.pkl'))
        X_xgb = xgb_selector.transform(X_xgb)

    if _use_dvl:
        dvl   = test_df[DVL_COLS].fillna(0).values.astype('float32')
        X_xgb = hstack([X_xgb, csr_matrix(dvl)], format='csr')

    _ver = _xgb_cfg.get('model_version', '')
    _lbl = f'XGBoost V{_ver}  (TF-IDF 50k)' if _ver else 'XGBoost  (TF-IDF 50k)'
    xgb_proba = xgb_model.predict_proba(X_xgb)[:, 1]
    preds  = (xgb_proba >= 0.5).astype(int)
    results['XGBoost'] = report(_lbl, y_test, preds)
except Exception as e:
    print(f'  XGBoost 실패: {e}')
    xgb_proba = None

# ── 2. NBSVM (SVM) ────────────────────────────────────────────────────────────
print('\n[2/5] NBSVM (SVM) 평가...')
try:
    import __main__
    from okt_utils import okt_tokenizer as _okt_fn
    __main__.okt_tokenizer = _okt_fn

    pfx       = os.path.join(MODEL, 'svm', 'nbsvm_ko')
    vec_word  = joblib.load(pfx + '_vec_word.pkl')
    vec_char  = joblib.load(pfx + '_vec_char.pkl')
    selector  = joblib.load(pfx + '_selector.pkl')
    r_vector  = joblib.load(pfx + '_r_vector.pkl')
    scaler    = joblib.load(pfx + '_scaler.pkl')
    svm_model = joblib.load(pfx + '_model.pkl')
    threshold = joblib.load(pfx + '_threshold.pkl')

    texts_svm = (test_df['title'] + ' ') * 5 + test_df['content']
    X_w  = vec_word.transform(texts_svm)
    X_c  = vec_char.transform(texts_svm)
    X_m  = hstack([X_w, X_c]).tocsr()
    X_s  = selector.transform(X_m)
    X_nb = X_s.multiply(r_vector).tocsr()

    t, c = test_df['title'].astype(str), test_df['content'].astype(str)
    tl, cl = t.str.len(), c.str.len()
    pat = '결국|충격|경악|속보|단독|논란|분노|의혹|발각|공개|주의|확인'
    ec = t.str.count(pat);  qc = c.str.count(r'["\']')
    lc = t.str.count(r'\.\.\.');  exc = t.str.count('!')+c.str.count('!')
    qsc = t.str.count(r'\?')+c.str.count(r'\?')
    meta = pd.DataFrame({
        'emo_count': ec, 'emo_ratio': ec/(tl+1),
        'quote_count': qc, 'quote_ratio': qc/(cl+1),
        'ellipsis_count': lc, 'ellipsis_ratio': lc/(tl+1),
        'excl_count': exc, 'excl_ratio': exc/(tl+cl+1),
        'quest_count': qsc, 'quest_ratio': qsc/(tl+cl+1),
        'title_len': tl, 'content_len': cl,
    })
    meta_sc = scaler.transform(meta)
    X_comb  = hstack([X_nb, csr_matrix(meta_sc)]).tocsr()

    scores    = svm_model.decision_function(X_comb)
    probs_svm = 1.0 / (1.0 + np.exp(-scores))
    preds     = (scores > float(threshold)).astype(int)
    results['NBSVM'] = report('NBSVM  (Okt+Chi2 40k+NB ratio+meta)', y_test, preds)
except Exception as e:
    print(f'  NBSVM 실패: {e}')

# ── 3. Logistic V4.3 (우선) / V4.2 (fallback) ───────────────────────────────
print('\n[3/5] Logistic Regression 평가...')
_LR_V43_PFX = os.path.join(BASE, 'models', 'logistic', 'logistic_v4_3')
_v43_exists  = os.path.exists(_LR_V43_PFX + '_model.pkl')

if _v43_exists:
    print('  [V4.3] 컴포넌트 로드 (단어/문자 TF-IDF + Chi2 30k + 메타 12개)...')
    try:
        # Okt 캐시에서 test 행 tokens 조회
        tok_cache = pd.read_csv(
            os.path.join(BASE, 'data', 'processed', 'unified_news_tokenized.csv'),
            usecols=['id', 'tokens'],
        )
        test_with_tok = test_df.merge(tok_cache, on='id', how='left')
        test_with_tok['tokens'] = test_with_tok['tokens'].fillna('')

        lr43_vw  = joblib.load(_LR_V43_PFX + '_vec_word.pkl')
        lr43_vc  = joblib.load(_LR_V43_PFX + '_vec_char.pkl')
        lr43_sel = joblib.load(_LR_V43_PFX + '_selector.pkl')
        lr43_sc  = joblib.load(_LR_V43_PFX + '_scaler.pkl')
        lr43_m   = joblib.load(_LR_V43_PFX + '_model.pkl')

        matched = (test_with_tok['tokens'] != '').sum()
        print(f'  tokens 캐시 매칭: {matched}/{len(test_with_tok)}건')
        if matched < len(test_with_tok) * 0.5:
            # 캐시 매칭률 50% 미만 → Okt 재토크나이징
            print(f'  캐시 매칭 부족 → Okt 토크나이징 ({len(test_with_tok):,}건)...')
            from okt_utils import okt_tokenizer as _okt_fn
            raw = (test_with_tok['title'] + ' ' + test_with_tok['content']).tolist()
            test_with_tok['tokens'] = [' '.join(_okt_fn(t)) for t in raw]
        X_w43  = lr43_vw.transform(test_with_tok['tokens'])
        char_t = test_with_tok['title']   # 모델 훈련과 동일: 제목만
        X_c43  = lr43_vc.transform(char_t)
        X_s43  = lr43_sel.transform(hstack([X_w43, X_c43]).tocsr())

        t43, c43 = test_with_tok['title'].astype(str), test_with_tok['content'].astype(str)
        tl43, cl43 = t43.str.len(), c43.str.len()
        pat = '결국|충격|경악|속보|단독|논란|분노|의혹|발각|공개|주의|확인'
        ec43 = t43.str.count(pat); qc43 = c43.str.count(r'["\']')
        lc43 = t43.str.count(r'\.\.\.'); ex43 = t43.str.count('!')+c43.str.count('!')
        qs43 = t43.str.count(r'\?')+c43.str.count(r'\?')
        meta43 = pd.DataFrame({
            'emo_count': ec43, 'emo_ratio': ec43/(tl43+1),
            'quote_count': qc43, 'quote_ratio': qc43/(cl43+1),
            'ellipsis_count': lc43, 'ellipsis_ratio': lc43/(tl43+1),
            'excl_count': ex43, 'excl_ratio': ex43/(tl43+cl43+1),
            'quest_count': qs43, 'quest_ratio': qs43/(tl43+cl43+1),
            'title_len': tl43, 'content_len': cl43,
        })
        meta43_sc = lr43_sc.transform(meta43)
        X_f43 = hstack([X_s43, csr_matrix(meta43_sc)]).tocsr()

        probs_lr = lr43_m.predict_proba(X_f43)[:, 1]
        preds    = (probs_lr >= 0.5).astype(int)
        results['Logistic V4.3'] = report(
            'Logistic V4.3  (Okt캐시 단어+문자 TF-IDF+Chi2 30k+메타12)', y_test, preds)
    except Exception as e:
        print(f'  Logistic V4.3 실패: {e}')
        _v43_exists = False

if not _v43_exists:
    print('  [V4.2] Pipeline 로드 (TF-IDF 50k)...')
    try:
        from okt_utils import okt_tokenizer as _okt_fn
        lr_pipeline = joblib.load(os.path.join(BASE, 'models', 'logistic', 'logistic_v4_2_pipeline.pkl'))
        raw_texts   = (test_df['title'] + ' ' + test_df['content']).tolist()
        print(f'  Okt 토크나이징 {len(raw_texts):,}건...')
        texts_lr = [' '.join(_okt_fn(t)) for t in raw_texts]
        probs_lr = lr_pipeline.predict_proba(texts_lr)[:, 1]
        preds    = (probs_lr >= 0.5).astype(int)
        results['Logistic V4.2'] = report(
            'Logistic V4.2  (Pipeline+Okt, TF-IDF 50k)', y_test, preds)
    except Exception as e:
        print(f'  Logistic V4.2 실패: {e}')

# ── 4. NaiveBayes ─────────────────────────────────────────────────────────────
print('\n[4/5] NaiveBayes 평가...')
try:
    nb_model = joblib.load(os.path.join(BASE, 'models', 'naive_bayes', 'best_model_naive_V1.pkl'))
    nb_vec   = joblib.load(os.path.join(VECTOR, 'korean_vectorizer.pkl'))
    texts    = (test_df['title'] + ' ' + test_df['content']).tolist()
    X_nb2    = nb_vec.transform(texts)
    probs_nb = nb_model.predict_proba(X_nb2)[:, 1]
    preds    = (probs_nb >= 0.5).astype(int)
    results['NaiveBayes'] = report('NaiveBayes  (TF-IDF 50k)', y_test, preds)
except Exception as e:
    print(f'  NaiveBayes 실패: {e}')

# ── 5. Ensemble ───────────────────────────────────────────────────────────────
print('\n[5/5] 앙상블 평가 (LR + NB + SVM, F1 가중 소프트 보팅)...')
try:
    W = {'logistic': 0.7105, 'nb': 0.6273, 'svm': 0.7301}
    available = []
    if probs_lr  is not None: available.append((probs_lr,  W['logistic']))
    if probs_nb  is not None: available.append((probs_nb,  W['nb']))
    if probs_svm is not None: available.append((probs_svm, W['svm']))

    if not available:
        print('  사용 가능한 모델 없음')
    else:
        total_w  = sum(w for _, w in available)
        ens_prob = sum(p * w for p, w in available) / total_w
        preds    = (ens_prob >= 0.5).astype(int)
        results['Ensemble'] = report('Ensemble  (LR + NB + SVM, 소프트 보팅)', y_test, preds)
except Exception as e:
    print(f'  앙상블 실패: {e}')

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print('  최종 요약 (동일 테스트셋 6,000건 기준)')
print(f'{"="*55}')
print(f'  {"모델":<35} {"Macro F1":>10} {"Weighted F1":>12}')
print(f'  {"-"*57}')
for name, (mac, wgt) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f'  {name:<35} {mac*100:>9.2f}% {wgt*100:>11.2f}%')
print(f'{"="*55}')


# ══════════════════════════════════════════════════════════════════════════════
# 6. DVL 피처 기여도 분석
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n\n{"="*55}')
print('  [6] DVL 피처 포함/제외 XGBoost 성능 비교')
print(f'{"="*55}')

try:
    import xgboost as xgb_lib
    from sklearn.feature_selection import SelectKBest, chi2

    K = 15000
    XGB_PARAMS = dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=0,
        tree_method='hist', max_bin=64,
    )

    vec_dvl  = joblib.load(os.path.join(VECTOR, 'korean_vectorizer.pkl'))
    texts_all = (df['title'] + ' ' + df['content']).tolist()
    X_all     = vec_dvl.transform(texts_all)
    y_all     = df['label'].astype(int).values
    dvl_all   = df[DVL_COLS].fillna(0).values.astype('float32')

    idx = np.arange(len(y_all))
    idx_tr_full, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_all)
    idx_tr, idx_vl      = train_test_split(idx_tr_full, test_size=0.1, random_state=42,
                                            stratify=y_all[idx_tr_full])

    print(f'Chi2 피처 선택 (K={K:,}) 중...')
    t0 = time.time()
    sel = SelectKBest(chi2, k=K)
    Xtr_sel = sel.fit_transform(X_all[idx_tr], y_all[idx_tr])
    Xvl_sel = sel.transform(X_all[idx_vl])
    Xte_sel = sel.transform(X_all[idx_te])
    print(f'  완료: {time.time()-t0:.1f}s')

    dvl_results = {}
    for use_dvl in [True, False]:
        tag = 'DVL 포함' if use_dvl else 'DVL 제외'
        if use_dvl:
            Xtr = hstack([Xtr_sel, csr_matrix(dvl_all[idx_tr])], format='csr')
            Xvl = hstack([Xvl_sel, csr_matrix(dvl_all[idx_vl])], format='csr')
            Xte = hstack([Xte_sel, csr_matrix(dvl_all[idx_te])], format='csr')
        else:
            Xtr, Xvl, Xte = Xtr_sel, Xvl_sel, Xte_sel

        print(f'\n── {tag} 학습 중 (피처 {Xtr.shape[1]:,}개) ──')
        m = xgb_lib.XGBClassifier(**XGB_PARAMS)
        t0 = time.time()
        m.fit(Xtr, y_all[idx_tr], eval_set=[(Xvl, y_all[idx_vl])], verbose=False)
        print(f'  학습 완료: {time.time()-t0:.1f}s  최적 트리: {m.best_iteration+1}')

        ypred  = m.predict(Xte)
        f1     = f1_score(y_all[idx_te], ypred, average='macro', zero_division=0)
        f1_r   = f1_score(y_all[idx_te], ypred, pos_label=0, average='binary', zero_division=0)
        f1_f   = f1_score(y_all[idx_te], ypred, pos_label=1, average='binary', zero_division=0)
        dvl_results[tag] = (f1, f1_r, f1_f)
        print(f'  F1 macro={f1:.4f}  진짜F1={f1_r:.4f}  가짜F1={f1_f:.4f}')

    print(f'\n{"="*55}')
    print(f'  {"":15s}  {"F1 macro":^10} {"진짜 F1":^10} {"가짜 F1":^10}')
    print(f'{"="*55}')
    for tag, (f1, r, f) in dvl_results.items():
        print(f'  {tag:13s}  {f1:.4f}      {r:.4f}      {f:.4f}')
    diff = dvl_results.get('DVL 포함', (0,))[0] - dvl_results.get('DVL 제외', (0,))[0]
    print(f'\n  DVL 기여도: {diff:+.4f}  (양수=도움, 음수=노이즈)')
    print(f'{"="*55}')
except Exception as e:
    print(f'  DVL 분석 실패: {e}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. XGBoost 임계값 최적화
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n\n{"="*55}')
print('  [7] XGBoost 임계값(threshold) 최적화')
print(f'{"="*55}')

try:
    if xgb_proba is None:
        raise RuntimeError('XGBoost 예측 확률 없음 (1번 평가 실패)')

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print(f'\n  {"임계값":^8} {"F1 macro":^10} {"Accuracy":^10} {"진짜 F1":^10} {"가짜 F1":^10}')
    print('  ' + '-'*52)

    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        ypred    = (xgb_proba >= thr).astype(int)
        f1_macro = f1_score(y_test, ypred, average='macro',  zero_division=0)
        f1_real  = f1_score(y_test, ypred, pos_label=0, average='binary', zero_division=0)
        f1_fake  = f1_score(y_test, ypred, pos_label=1, average='binary', zero_division=0)
        acc      = accuracy_score(y_test, ypred)
        mark = '  ★' if f1_macro > best_f1 else ''
        if f1_macro > best_f1:
            best_f1, best_thr = f1_macro, thr
        print(f'  {thr:.2f}     {f1_macro:.4f}     {acc:.4f}     {f1_real:.4f}     {f1_fake:.4f}{mark}')

    base_f1 = f1_score(y_test, (xgb_proba >= 0.50).astype(int), average='macro', zero_division=0)
    print(f'\n  최적 임계값: {best_thr}  →  F1 macro: {best_f1:.4f}')
    print(f'  기본값(0.50) 대비: {best_f1 - base_f1:+.4f}')

    print(f'\n  [최적 임계값 {best_thr} 상세]')
    ypred_best = (xgb_proba >= best_thr).astype(int)
    print(classification_report(y_test, ypred_best,
          target_names=['진짜(0)', '가짜(1)'], zero_division=0))
    cm = confusion_matrix(y_test, ypred_best)
    print(f'  오분류율 — 진짜→가짜: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}%  '
          f'가짜→진짜: {cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}%')
    print(f'{"="*55}')
except Exception as e:
    print(f'  임계값 분석 실패: {e}')
