"""
가짜뉴스 점수 계산
  - XGBoost: TF-IDF Chi2 [+ DVL] → XGBoost  (한국어/영어)
  - Logistic: TF-IDF 50k → LogisticRegression  (한국어)
  - 모델 파일 없으면 랜덤 점수 fallback
"""

import json
import os
import re
import random
import sys
import numpy as np
import pandas as pd

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_MODELS_DIR  = os.path.join(_PROJECT_DIR, 'models')
_MODEL_DIR   = os.path.join(_MODELS_DIR, 'xgboost')   # XGBoost 전용 (하위 호환)
_VECTOR_DIR  = os.path.join(_PROJECT_DIR, 'data', 'vector')

_KOREAN_RATIO_THRESHOLD = 0.15
_EMB_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
_EMB_TEXT_MAX   = 400

_xgb_cache: dict = {}
_lr_cache:  dict = {}
_svm_cache: dict = {}
_nb_cache:  dict = {}
_emb_model = None

_LR_PIPELINE_PATH = os.path.join(_MODELS_DIR, 'logistic', 'logistic_v4_2_pipeline.pkl')
_LR_V43_PREFIX    = os.path.join(_MODELS_DIR, 'logistic', 'logistic_v4_3')
_NB_MODEL_PATH    = os.path.join(_MODELS_DIR, 'naive_bayes', 'best_model_naive_V1.pkl')

# 앙상블 가중치 (테스트셋 macro F1 기준)
_ENSEMBLE_WEIGHTS = {
    'logistic': 0.7105,
    'svm':      0.7301,  # NBSVM 재훈련 결과 (Okt+Chi2 40k+NB ratio+meta)
    'nb':       0.6273,
}


def _load_xgb(lang: str = 'korean') -> dict | None:
    if lang in _xgb_cache:
        return _xgb_cache[lang]
    prefix = lang[:2]
    try:
        import joblib
        selector_path = os.path.join(_MODEL_DIR, f'xgb_selector_{prefix}.pkl')
        config_path   = os.path.join(_MODEL_DIR, f'xgb_config_{prefix}.json')

        use_emb = False
        use_dvl = False
        use_sel = True  # 기본값: selector 사용 (구버전 호환)
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                use_emb = cfg.get('use_embeddings', False)
                use_dvl = cfg.get('use_dvl', False)
                use_sel = cfg.get('use_selector', True)

        result = dict(
            model          = joblib.load(os.path.join(_MODEL_DIR,  f'xgb_model_{prefix}.pkl')),
            vectorizer     = joblib.load(os.path.join(_VECTOR_DIR, f'{lang}_vectorizer.pkl')),
            selector       = joblib.load(selector_path) if (use_sel and os.path.exists(selector_path)) else None,
            use_embeddings = use_emb,
            use_dvl        = use_dvl,
        )
        _xgb_cache[lang] = result
        dvl_info = ' + DVL' if use_dvl else ''
        emb_info = ' + 임베딩' if use_emb else ''
        print(f'[detector] XGBoost {lang} 로드 완료 (Chi2{dvl_info}{emb_info})')
        return result
    except Exception as e:
        print(f'[detector] XGBoost {lang} 로드 실패: {e}')
        _xgb_cache[lang] = None
        return None


def _load_emb_model():
    global _emb_model
    if _emb_model is not None:
        return _emb_model
    try:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(_EMB_MODEL_NAME)
        print(f'[detector] 임베딩 모델 로드: {_EMB_MODEL_NAME}')
    except Exception as e:
        print(f'[detector] 임베딩 모델 로드 실패: {e}')
        _emb_model = False  # 재시도 방지
    return _emb_model if _emb_model else None


def _detect_lang(text: str) -> str:
    if not text or len(text) < 10:
        return 'korean'
    korean_chars = len(re.findall(r'[가-힣]', text))
    ratio = korean_chars / len(text)
    return 'korean' if ratio >= _KOREAN_RATIO_THRESHOLD else 'english'


def _article_text(a: dict) -> str:
    parts = [
        str(a.get('title',   '') or ''),
        str(a.get('summary', '') or ''),
        str(a.get('text',    '') or ''),
    ]
    return ' '.join(p for p in parts if p)


def _predict_batch(articles: list, lang: str) -> bool:
    if not articles:
        return True
    m = _load_xgb(lang)
    if not m:
        return False
    try:
        # korean_vectorizer는 Okt 형태소 토큰으로 피팅 → 반드시 토크나이징 후 입력
        try:
            sys.path.insert(0, _SCRIPT_DIR)
            from okt_utils import okt_tokenizer as _okt_xgb
        except ImportError:
            from src.okt_utils import okt_tokenizer as _okt_xgb
        raw_texts = [_article_text(a) for a in articles]
        texts = [' '.join(_okt_xgb(t)) for t in raw_texts]

        # TF-IDF → (선택적) Chi2 선택
        X = m['vectorizer'].transform(texts)
        if m.get('selector') is not None:
            X = m['selector'].transform(X)

        # DVL 5개 플래그 결합 (tune_xgboost.py 로 학습된 모델)
        if m.get('use_dvl'):
            _DVL_COLS = ['stat_distortion', 'causal_error', 'emotional_provocation',
                         'source_lack', 'img_mismatch']
            from scipy.sparse import hstack, csr_matrix
            dvl = np.array(
                [[float(a.get(c, 0) or 0) for c in _DVL_COLS] for a in articles],
                dtype='float32',
            )
            X = hstack([X, csr_matrix(dvl)], format='csr')

        # 임베딩 결합 (모델이 임베딩 포함으로 학습된 경우)
        if m.get('use_embeddings'):
            emb_model = _load_emb_model()
            if emb_model is not None:
                from scipy.sparse import hstack, csr_matrix
                emb_texts = [t[:_EMB_TEXT_MAX] for t in texts]
                embeddings = emb_model.encode(
                    emb_texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                X = hstack([X, csr_matrix(embeddings.astype('float32'))], format='csr')

        probs = m['model'].predict_proba(X)[:, 1]

        for a, prob in zip(articles, probs):
            a['fake_score']      = round(float(prob) * 100, 1)
            a['predicted_label'] = 1 if prob >= 0.5 else 0
            a['lang']            = lang
        return True

    except Exception as e:
        print(f'[detector] XGBoost {lang} 예측 실패: {e}')
        return False


def predict_articles_inplace(articles: list, lang: str = 'auto') -> bool:
    """XGBoost로 fake_score(0~100)·predicted_label을 article 딕셔너리에 인플레이스 기록."""
    if not articles:
        return False

    if lang != 'auto':
        return _predict_batch(articles, lang)

    ko_articles: list = []
    en_articles: list = []
    for a in articles:
        if _detect_lang(_article_text(a)) == 'english':
            en_articles.append(a)
        else:
            ko_articles.append(a)

    if en_articles:
        print(f'[detector] 언어 감지: 한국어 {len(ko_articles)}건 / 영어 {len(en_articles)}건')

    success = False

    if ko_articles:
        success |= _predict_batch(ko_articles, 'korean')

    if en_articles:
        if not _predict_batch(en_articles, 'english'):
            print(f'[detector] 영어 모델 없음 → 한국어 모델로 {len(en_articles)}건 fallback')
            success |= _predict_batch(en_articles, 'korean')
        else:
            success = True

    return success


def _load_svm(lang: str = 'korean') -> dict | None:
    """NBSVM 풀 파이프라인 로드 (7개 컴포넌트)."""
    if lang in _svm_cache:
        return _svm_cache[lang]
    pfx = os.path.join(_MODELS_DIR, 'svm', 'nbsvm_ko')
    try:
        import joblib
        import __main__

        # joblib이 __main__.okt_tokenizer를 찾으므로, 로드 중에만 임시 주입
        try:
            sys.path.insert(0, _SCRIPT_DIR)
            from okt_utils import okt_tokenizer as _okt_fn
        except ImportError:
            from src.okt_utils import okt_tokenizer as _okt_fn

        _had = hasattr(__main__, 'okt_tokenizer')
        __main__.okt_tokenizer = _okt_fn
        try:
            vec_word = joblib.load(pfx + '_vec_word.pkl')
        finally:
            if not _had:
                try:
                    delattr(__main__, 'okt_tokenizer')
                except AttributeError:
                    pass

        result = dict(
            vec_word  = vec_word,
            vec_char  = joblib.load(pfx + '_vec_char.pkl'),
            selector  = joblib.load(pfx + '_selector.pkl'),
            r_vector  = joblib.load(pfx + '_r_vector.pkl'),
            scaler    = joblib.load(pfx + '_scaler.pkl'),
            model     = joblib.load(pfx + '_model.pkl'),
            threshold = joblib.load(pfx + '_threshold.pkl'),
        )
        _svm_cache[lang] = result
        print('[detector] NBSVM korean 로드 완료 (Okt+Chi2+NB ratio+meta, 40,012 피처)')
        return result
    except Exception as e:
        print(f'[detector] NBSVM {lang} 로드 실패: {e}')
        _svm_cache[lang] = None
        return None


def _svm_meta_features(articles: list) -> 'np.ndarray':
    """NBSVM용 12개 메타 피처 추출."""
    rows = [{'title': str(a.get('title', '') or ''), 'content': str(a.get('text', '') or a.get('summary', '') or '')} for a in articles]
    df = pd.DataFrame(rows)
    title_str   = df['title']
    content_str = df['content']
    title_len   = title_str.str.len()
    content_len = content_str.str.len()
    pat = '결국|충격|경악|속보|단독|논란|분노|의혹|발각|공개|주의|확인'
    emo_count      = title_str.str.count(pat)
    quote_count    = content_str.str.count(r'["\']')
    ellipsis_count = title_str.str.count(r'\.\.\.')
    excl_count     = title_str.str.count('!') + content_str.str.count('!')
    quest_count    = title_str.str.count(r'\?') + content_str.str.count(r'\?')
    meta = pd.DataFrame({
        'emo_count':      emo_count,
        'emo_ratio':      emo_count / (title_len + 1),
        'quote_count':    quote_count,
        'quote_ratio':    quote_count / (content_len + 1),
        'ellipsis_count': ellipsis_count,
        'ellipsis_ratio': ellipsis_count / (title_len + 1),
        'excl_count':     excl_count,
        'excl_ratio':     excl_count / (title_len + content_len + 1),
        'quest_count':    quest_count,
        'quest_ratio':    quest_count / (title_len + content_len + 1),
        'title_len':      title_len,
        'content_len':    content_len,
    })
    return meta


def predict_svm_inplace(articles: list) -> bool:
    """NBSVM 풀 파이프라인으로 fake_score·predicted_label을 인플레이스 기록 (한국어 전용)."""
    if not articles:
        return False
    m = _load_svm('korean')
    if not m:
        return False
    try:
        from scipy.sparse import hstack, csr_matrix
        # 제목 5배 증폭
        texts = [(str(a.get('title', '') or '') + ' ') * 5 + str(a.get('text', '') or a.get('summary', '') or '') for a in articles]

        X_word    = m['vec_word'].transform(texts)
        X_char    = m['vec_char'].transform(texts)
        X_massive = hstack([X_word, X_char]).tocsr()
        X_sel     = m['selector'].transform(X_massive)
        X_tfidf   = X_sel.multiply(m['r_vector']).tocsr()

        meta_scaled = m['scaler'].transform(_svm_meta_features(articles))
        X_combined  = hstack([X_tfidf, csr_matrix(meta_scaled)]).tocsr()

        scores = m['model'].decision_function(X_combined)
        probs  = 1.0 / (1.0 + np.exp(-scores))  # sigmoid

        for a, score, prob in zip(articles, scores, probs):
            a['fake_score']      = round(float(prob) * 100, 1)
            a['predicted_label'] = 1 if float(score) > float(m['threshold']) else 0
        return True
    except Exception as e:
        print(f'[detector] NBSVM 예측 실패: {e}')
        return False


def _load_nb(lang: str = 'korean') -> dict | None:
    if lang in _nb_cache:
        return _nb_cache[lang]
    try:
        import joblib
        result = dict(
            model      = joblib.load(_NB_MODEL_PATH),
            vectorizer = joblib.load(os.path.join(_VECTOR_DIR, 'korean_vectorizer.pkl')),
        )
        _nb_cache[lang] = result
        print(f'[detector] NaiveBayes 로드 완료 (TF-IDF 50k)')
        return result
    except Exception as e:
        print(f'[detector] NaiveBayes 로드 실패: {e}')
        _nb_cache[lang] = None
        return None


def predict_ensemble_inplace(articles: list) -> bool:
    """NaiveBayes + SVM + Logistic F1-가중 소프트 보팅 앙상블."""
    if not articles:
        return False

    # 공통 TF-IDF 벡터 (50k) — Logistic 벡터라이저 재사용
    m_lr  = _load_logistic('korean')
    m_svm = _load_svm('korean')
    m_nb  = _load_nb('korean')

    if not any([m_lr, m_svm, m_nb]):
        return False

    texts       = [_article_text(a) for a in articles]

    # NB용 공통 벡터 (50k)
    X_full = None
    if m_nb:
        try:
            X_full = m_nb['vectorizer'].transform(texts)
        except Exception:
            pass

    probs_list: list = []
    weights:    list = []

    if m_lr:
        try:
            from src.okt_utils import okt_tokenizer
            if m_lr['version'] == 'v4.3':
                from scipy.sparse import hstack as _hstack43, csr_matrix as _csr43
                titles_lr   = [str(a.get('title',   '') or '') for a in articles]
                contents_lr = [str(a.get('text', '') or a.get('summary', '') or '') for a in articles]
                full_texts_43 = [t + ' ' + c for t, c in zip(titles_lr, contents_lr)]
                texts_tok     = [' '.join(okt_tokenizer(t)) for t in full_texts_43]
                X_word = m_lr['vec_word'].transform(texts_tok)
                X_char = m_lr['vec_char'].transform(titles_lr)  # 제목만
                X_sel  = m_lr['selector'].transform(_hstack43([X_word, X_char]).tocsr())
                meta_sc = m_lr['scaler'].transform(_svm_meta_features(articles))
                X_final = _hstack43([X_sel, _csr43(meta_sc)]).tocsr()
                probs_list.append(m_lr['model'].predict_proba(X_final)[:, 1])
            else:
                texts_lr = [' '.join(okt_tokenizer(t)) for t in texts]
                probs_list.append(m_lr['pipeline'].predict_proba(texts_lr)[:, 1])
            weights.append(_ENSEMBLE_WEIGHTS['logistic'])
        except Exception as e:
            print(f'[detector] 앙상블 Logistic 실패: {e}')

    if m_nb and X_full is not None:
        try:
            probs_list.append(m_nb['model'].predict_proba(X_full)[:, 1])
            weights.append(_ENSEMBLE_WEIGHTS['nb'])
        except Exception as e:
            print(f'[detector] 앙상블 NaiveBayes 실패: {e}')

    if m_svm:
        try:
            from scipy.sparse import hstack as _hstack, csr_matrix as _csr
            texts_svm = [(str(a.get('title', '') or '') + ' ') * 5 + str(a.get('text', '') or a.get('summary', '') or '') for a in articles]
            X_word    = m_svm['vec_word'].transform(texts_svm)
            X_char    = m_svm['vec_char'].transform(texts_svm)
            X_massive = _hstack([X_word, X_char]).tocsr()
            X_sel     = m_svm['selector'].transform(X_massive)
            X_tfidf   = X_sel.multiply(m_svm['r_vector']).tocsr()
            meta_sc   = m_svm['scaler'].transform(_svm_meta_features(articles))
            X_comb    = _hstack([X_tfidf, _csr(meta_sc)]).tocsr()
            scores    = m_svm['model'].decision_function(X_comb)
            probs_list.append(1.0 / (1.0 + np.exp(-scores)))
            weights.append(_ENSEMBLE_WEIGHTS['svm'])
        except Exception as e:
            print(f'[detector] 앙상블 SVM 실패: {e}')

    if not probs_list:
        return False

    total_w = sum(weights)
    ensemble_prob = sum(p * w for p, w in zip(probs_list, weights)) / total_w

    for a, prob in zip(articles, ensemble_prob):
        a['fake_score']      = round(float(prob) * 100, 1)
        a['predicted_label'] = 1 if prob >= 0.5 else 0
    return True


def _load_logistic(lang: str = 'korean') -> dict | None:
    if lang in _lr_cache:
        return _lr_cache[lang]
    import joblib
    # V4.3 우선 시도
    if os.path.exists(_LR_V43_PREFIX + '_model.pkl'):
        try:
            try:
                sys.path.insert(0, _SCRIPT_DIR)
                from okt_utils import okt_tokenizer as _okt_fn
            except ImportError:
                from src.okt_utils import okt_tokenizer as _okt_fn
            import __main__
            _had = hasattr(__main__, 'okt_tokenizer')
            __main__.okt_tokenizer = _okt_fn
            try:
                vec_word = joblib.load(_LR_V43_PREFIX + '_vec_word.pkl')
            finally:
                if not _had:
                    try:
                        delattr(__main__, 'okt_tokenizer')
                    except AttributeError:
                        pass
            result = dict(
                version  = 'v4.3',
                vec_word = vec_word,
                vec_char = joblib.load(_LR_V43_PREFIX + '_vec_char.pkl'),
                selector = joblib.load(_LR_V43_PREFIX + '_selector.pkl'),
                scaler   = joblib.load(_LR_V43_PREFIX + '_scaler.pkl'),
                model    = joblib.load(_LR_V43_PREFIX + '_model.pkl'),
            )
            _lr_cache[lang] = result
            print('[detector] Logistic V4.3 로드 완료 (제목×5 + 단어/문자 TF-IDF + Chi2 30k + 메타12)')
            return result
        except Exception as e:
            print(f'[detector] Logistic V4.3 로드 실패 → V4.2 fallback: {e}')
    # V4.2 fallback
    try:
        pipeline = joblib.load(_LR_PIPELINE_PATH)
        result = dict(version='v4.2', pipeline=pipeline)
        _lr_cache[lang] = result
        print('[detector] Logistic V4.2 Pipeline 로드 완료 (TF-IDF 50k)')
        return result
    except Exception as e:
        print(f'[detector] Logistic 로드 실패: {e}')
        _lr_cache[lang] = None
        return None


def predict_logistic_inplace(articles: list) -> bool:
    """LogisticRegression으로 fake_score·predicted_label을 인플레이스 기록 (한국어 전용)."""
    if not articles:
        return False
    m = _load_logistic('korean')
    if not m:
        return False
    try:
        from src.okt_utils import okt_tokenizer
        if m['version'] == 'v4.3':
            from scipy.sparse import hstack, csr_matrix
            titles   = [str(a.get('title',   '') or '') for a in articles]
            contents = [str(a.get('text', '') or a.get('summary', '') or '') for a in articles]
            # word TF-IDF: 전체 텍스트 토크나이징 (훈련 tokens 컬럼과 일치)
            full_texts = [t + ' ' + c for t, c in zip(titles, contents)]
            texts_tok  = [' '.join(okt_tokenizer(t)) for t in full_texts]
            X_word = m['vec_word'].transform(texts_tok)
            # char TF-IDF: 제목만 (훈련 시 title 컬럼 사용과 일치)
            X_char = m['vec_char'].transform(titles)
            X_sel  = m['selector'].transform(hstack([X_word, X_char]).tocsr())
            meta_sc = m['scaler'].transform(_svm_meta_features(articles))
            X_final = hstack([X_sel, csr_matrix(meta_sc)]).tocsr()
            probs = m['model'].predict_proba(X_final)[:, 1]
        else:
            texts = [' '.join(okt_tokenizer(_article_text(a))) for a in articles]
            probs = m['pipeline'].predict_proba(texts)[:, 1]
        for a, prob in zip(articles, probs):
            a['fake_score']      = round(float(prob) * 100, 1)
            a['predicted_label'] = 1 if prob >= 0.5 else 0
        return True
    except Exception as e:
        print(f'[detector] Logistic 예측 실패: {e}')
        return False


def _time_sort_key(article: dict) -> tuple:
    t = article.get('time', '')

    m = re.match(r'오후\s*(\d+):(\d+)', t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return (1, (12 + h if h != 12 else 12) * 60 + mn)

    m = re.match(r'(?:오전\s*)?(\d+):(\d+)', t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return (1, h * 60 + mn)

    m = re.match(r'(\d{4})\.(\d{2})\.(\d{2})', t)
    if m:
        return (0, int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3)))

    return (0, 0)


def get_sorted_timeline(raw_news_list: list) -> list:
    if not predict_articles_inplace(raw_news_list, lang='auto'):
        for news in raw_news_list:
            if 'fake_score' not in news:
                news['fake_score']      = round(random.uniform(0, 100), 1)
                news['predicted_label'] = 1 if news['fake_score'] >= 50 else 0

    return sorted(raw_news_list, key=_time_sort_key, reverse=True)
