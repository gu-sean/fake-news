# Fake-News

네이버 뉴스 실시간 크롤링 + 머신러닝 기반 가짜뉴스 탐지 시스템


## 1. 프로젝트 개요

네이버 뉴스를 실시간으로 크롤링하고, 4개 머신러닝 모델(XGBoost · Logistic Regression · NBSVM · NaiveBayes)과 소프트 보팅 앙상블로 각 기사의 진위 여부를 분석합니다.  
분석 결과는 네이버 뉴스 UI 형태로 렌더링되며, 기사마다 신뢰도 점수와 진짜/가짜 배지가 표시됩니다.


## 2. 데이터 표준 스키마

| 컬럼명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `id` | int | 행 고유 식별자 (1부터 순번) |
| `title` | str | 기사 제목 (원문) |
| `content` | str | 기사 본문 (원문, 최대 10,000자 제한) |
| `media` | str | 출처 매체명 (불명 시 `Unknown` 처리) |
| `date` | str | 발행일 (`YYYY-MM-DD` 표준화) |
| `label` | int | 정답 레이블 (`0` = 진짜 뉴스, `1` = 가짜 뉴스) |
| `clean_message` | str | 학습용 정제 텍스트 (HTML 태그·특수문자 제거) |
| `stat_distortion` | int | [DVL 1] 통계 왜곡 패턴 포함 여부 (`0` 또는 `1`) |
| `causal_error` | int | [DVL 2] 인과 오류 패턴 포함 여부 (`0` 또는 `1`) |
| `emotional_provocation` | int | [DVL 3] 감정 자극 패턴 포함 여부 (`0` 또는 `1`) |
| `source_lack` | int | [DVL 4] 출처 불명 패턴 포함 여부 (`0` 또는 `1`) |
| `img_mismatch` | int | [DVL 5] 이미지 불일치 패턴 포함 여부 (`0` 또는 `1`) |


## 3. 주요 특징

### 멀티 모델 추론 구조
- UI에서 모델을 선택하면 해당 모델로 실시간 추론
- `XGBoost` · `Logistic` · `SVM` · `NaiveBayes` · `Ensemble` 5개 선택 가능
- `src/detector.py`에서 모델별 추론 함수를 분리 관리, 결과는 인메모리 캐싱

### TF-IDF + DVL 하이브리드 아키텍처 (XGBoost)
- TF-IDF 벡터(Chi2 선택 15,000차원)와 DVL 5대 플래그를 `hstack`으로 결합
- XGBoost는 텍스트 어휘 패턴과 통계적 왜곡 패턴을 동시에 학습

### Okt 형태소 분석 기반 벡터화
- **Logistic / NBSVM**: KoNLPy(Okt) 형태소 분석 후 TF-IDF 벡터화
- `src/okt_utils.py`에 토크나이저를 공유 모듈로 분리 → joblib 역직렬화 시 참조 일관성 보장
- 형태소 분석 결과는 `data/processed/unified_news_tokenized.csv`에 캐싱

### 이중 언어 분리 처리
- **영어(Kaggle) 24,973건**: `clean_message` 기준 TF-IDF, `stop_words='english'`
- **한국어(AI Hub) 318,235건**: Okt 형태소 분석 후 TF-IDF
- `unified_news_refined.csv` 단일 소스에서 `media` 컬럼으로 분리

### 네이버 뉴스 클론 UI
- **aiohttp 비동기 크롤링**: 20개 동시 요청, 카테고리별 기사 실시간 수집
- **FastAPI + Jinja2**: AI 탐지 점수를 부여해 네이버 뉴스 형태로 렌더링
- **AI 배지**: 점수 50% 이상 → `가짜 X%` (빨간색), 미만 → `진짜` (초록색)
- **전체 섹션 사전 워밍**: 서버 시작 시 모든 탭을 백그라운드 사전 크롤링
- **5분 자동 갱신**: 전체 캐시 주기적 새로고침, `/refresh`로 강제 재크롤링 가능
- **제목 중복 제거**: 동일 제목 기사(다른 URL)는 크롤링 단계와 결과 렌더링 단계에서 모두 제거


## 4. DVL (Dynamic Verification Layer) 5대 패턴

가짜뉴스 특유의 조작 패턴을 정량 플래그로 추출해 분류 특징(Feature)으로 활용합니다.

| 플래그 | 정의 | 탐지 키워드 예시 |
| :--- | :--- | :--- |
| `stat_distortion` | 수치·빈도를 과장·절대화 | `100%`, `모든`, `항상`, `절대`, `always` |
| `causal_error` | 검증되지 않은 인과관계 서술 | `때문에`, `증명`, `causes`, `proves` |
| `emotional_provocation` | 분노·공포·혐오 감정 유발 | `충격`, `경악`, `shocking`, `outrage` |
| `source_lack` | 익명 소식통에 의존 | `관계자`, `익명`, `sources say`, `reportedly` |
| `img_mismatch` | 본문과 무관한 이미지 묘사 | `사진 속`, `이 사진은`, `photo shows` |


## 5. 서버 실행

```bash
python main.py --step server          # 웹 서버만 실행 (기본 포트 8000)
python main.py --step server --port 8080
```

접속: http://localhost:8000



## 6. 프로젝트 디렉토리 구조

```text
fake_news/
├── app.py                              # FastAPI 서버 — 라우트·캐시·사전 워밍·자동 갱신
├── main.py                             # CLI 진입점 (--step server/train/vectorize/all)
├── requirements.txt                    # 패키지 의존성
├── README.md
│
├── src/                                # 런타임 추론 코드
│   ├── detector.py                     # 모델별 추론 함수 + fake_score 부여
│   ├── crawler.py                      # 네이버 뉴스 비동기 크롤러 (aiohttp)
│   ├── okt_utils.py                    # Okt 토크나이저 공유 모듈
│   ├── factcheck_crawler.py            # 팩트체크 기사 크롤러
│   └── merge_factcheck.py              # 팩트체크 데이터 → unified_news_refined.csv 병합
│
├── models/                             # 학습된 모델 파일
│   ├── xgboost/
│   │   ├── xgb_model_ko.pkl            # XGBoost 한국어 모델 (최신)
│   │   ├── xgb_model_en.pkl            # XGBoost 영어 모델
│   │   ├── xgb_selector_ko.pkl         # 한국어 Chi2 피처 선택기 (K=15,000)
│   │   ├── xgb_selector_en.pkl         # 영어 Chi2 피처 선택기
│   │   ├── xgb_scaler_ko.pkl           # 한국어 피처 스케일러
│   │   ├── xgb_scaler_en.pkl           # 영어 피처 스케일러
│   │   ├── xgb_svd_ko.pkl              # 한국어 SVD 차원 축소기
│   │   ├── xgb_svd_en.pkl              # 영어 SVD 차원 축소기
│   │   ├── xgb_config_ko.json          # 설정 (use_dvl, k)
│   │   ├── xgb_results.txt             # K값별 성능 비교 로그
│   │   ├── xgboost_V1_log.txt          # V1 그리드서치 전체 로그
│   │   ├── best_config_xgboost_V1.txt  # V1 최적 하이퍼파라미터 기록
│   │   ├── best_model_xgboost_V1.pkl   # XGBoost V1 모델
│   │   ├── best_model_xgboost_V2.pkl   # XGBoost V2 모델
│   │   └── performance_xgboost_V2.txt  # V2 성능 평가 기록
│   ├── logistic/
│   │   ├── logistic_v4_2_pipeline.pkl  # TF-IDF + LogisticRegression Pipeline (추론용)
│   │   ├── best_model_V4_2.pkl         # 최적 모델 단독 저장본
│   │   ├── best_config_V4_2.txt        # 최적 하이퍼파라미터 기록
│   │   └── logistic_V4_2_log.txt       # 그리드서치 전체 로그
│   ├── svm/
│   │   ├── nbsvm.py                    # NBSVM 클래스 정의
│   │   ├── nbsvm_ko_vec_word.pkl       # Okt 단어 TF-IDF 벡터라이저
│   │   ├── nbsvm_ko_vec_char.pkl       # 문자 n-gram TF-IDF 벡터라이저
│   │   ├── nbsvm_ko_selector.pkl       # Chi2 선택기 (40,000)
│   │   ├── nbsvm_ko_r_vector.pkl       # NB log-count ratio 가중치
│   │   ├── nbsvm_ko_scaler.pkl         # 메타피처 StandardScaler
│   │   ├── nbsvm_ko_model.pkl          # LinearSVC 모델
│   │   ├── nbsvm_ko_threshold.pkl      # 최적 결정 임계값
│   │   ├── svm_model_ko.pkl            # SVM 단독 모델
│   │   ├── svm_best_tuning_values.pkl  # 최적 튜닝 파라미터
│   │   ├── svm_config_ko.json          # SVM 설정
│   │   └── README.md                   # SVM 모델 사용법
│   └── naive_bayes/
│       ├── best_model_naive_V1.pkl     # MultinomialNB 최적 모델
│       ├── best_config_naive_V1.txt    # 최적 하이퍼파라미터 기록
│       └── naive_V1_log.txt            # 전체 튜닝 로그
│
├── training/                           # 모델 재훈련 스크립트
│   ├── train_logistic.py               # Logistic V4.2 Pipeline 훈련
│   ├── train_logistic_v43.py           # Logistic V4.3 실험 훈련
│   ├── train_nbsvm.py                  # NBSVM 전체 파이프라인 훈련 (20~30분)
│   ├── train_naive.py                  # NaiveBayes 그리드서치 훈련
│   ├── preprocess.py                   # 데이터 전처리 (정제·DVL 추출)
│   ├── split_dataset.py                # 대용량 데이터 균등 분할 (subset_01~N.csv)
│   ├── vectorize.py                    # TF-IDF 벡터화 (영어·한국어 분리)
│   ├── retrain_xgb_v2.py              # XGBoost V2 재훈련 스크립트
│   ├── xgboost_V1_train_pipeline.py    # XGBoost V1 훈련 파이프라인
│   ├── xgboost_V2_train_pipeline.py    # XGBoost V2 훈련 파이프라인
│   ├── logistic_V4_2_train_pipeline.py # Logistic V4.2 훈련 파이프라인 (레거시)
│   └── naive_V1_train_pipeline.py      # NaiveBayes V1 훈련 파이프라인 (레거시)
│
├── scripts/                            # 평가·튜닝 스크립트
│   ├── eval_all_models.py              # 4개 모델 + 앙상블 종합 평가
│   ├── tune_threshold.py               # Logistic·SVM·Ensemble 임계값 최적화
│   ├── tune_xgboost.py                 # XGBoost Optuna 하이퍼파라미터 튜닝
│   └── run_retrain.py                  # 전체 재훈련 파이프라인 (6단계)
│
├── static/                             # CSS 정적 파일
│   ├── newshome.css                    # 메인 레이아웃 스타일시트
│   └── news.css                        # 기사 상세 페이지 스타일시트
│
├── templates/                          # Jinja2 HTML 템플릿
│   ├── naver_news_clone.html           # 메인 클론 UI (배지·카테고리·페이지네이션)
│   ├── naver_news_left.html            # 좌측 기사 레이아웃
│   └── naver_news_right.html           # 우측 기사 레이아웃
│
├── data/
│   ├── raw/                            # 원천 데이터
│   │   ├── Fake.csv                    # 영어 가짜 뉴스 (Kaggle)
│   │   ├── True.csv                    # 영어 진짜 뉴스 (Kaggle)
│   │   ├── Fake_Real_News_Data.csv     # 영어 통합 데이터셋
│   │   ├── fake_real_news.arff         # ARFF 포맷 영어 데이터셋
│   │   ├── Training/                   # AI Hub 한국어 학습 데이터
│   │   │   └── 02.라벨링데이터/        # 카테고리별 zip (21개)
│   │   └── Validation/                 # AI Hub 한국어 검증 데이터
│   │       └── 02.라벨링데이터/        # 카테고리별 zip (21개)
│   ├── processed/                      # 전처리 완료 산출물
│   │   ├── unified_news_refined.csv    # 최종 전처리 완료 데이터 (영어+한국어)
│   │   ├── unified_news_refined_backup.csv  # 백업본
│   │   ├── unified_news_tokenized.csv  # 한국어 Okt 형태소 분석 캐시
│   │   └── subsets/                    # 균등 분할 서브셋 (subset_01~11.csv, 각 30,000건)
│   ├── vector/                         # TF-IDF 벡터화 결과물
│   │   ├── korean_tfidf.npz            # 한국어 TF-IDF 희소 행렬
│   │   ├── korean_labels.npy           # 한국어 레이블
│   │   ├── korean_dvl_flags.npy        # 한국어 DVL 5대 플래그
│   │   ├── korean_vectorizer.pkl       # 한국어 TF-IDF 벡터라이저
│   │   ├── english_tfidf.npz           # 영어 TF-IDF 희소 행렬
│   │   ├── english_labels.npy          # 영어 레이블
│   │   ├── english_dvl_flags.npy       # 영어 DVL 5대 플래그
│   │   └── english_vectorizer.pkl      # 영어 TF-IDF 벡터라이저
│   ├── factcheck/                      # 팩트체크 데이터
│   │   ├── factcheck_raw.csv           # 팩트체크 원문 크롤링 결과
│   │   └── factcheck_label.csv         # 레이블 정제 완료 데이터
│   ├── realtime/                       # 실시간 크롤링 수집 결과
│   │   └── realtime_news.csv           # 네이버 뉴스 실시간 수집 CSV
│   ├── realtime_news_result.csv        # 추론 결과 포함 실시간 뉴스 CSV
│   └── cache/                          # 서버 캐시 (카테고리별 JSON)
│
├── visualization/                      # 데이터 시각화
│   ├── src/
│   │   ├── data_visual1.py             # 클래스 분포 차트 생성
│   │   ├── data_visual2.py             # 가짜뉴스 단어 빈도 차트
│   │   ├── data_visual3.py             # 가짜뉴스 TF-IDF 차트
│   │   ├── data_visual4.py             # 진짜뉴스 단어 빈도 차트
│   │   └── data_visual5.py             # 진짜뉴스 TF-IDF 차트
│   └── img/
│       ├── data_visual1_class_distribution.png
│       ├── data_visual2_word_frequency.png
│       ├── data_visual3_tfidf.png
│       ├── data_visual4_real_word_frequency.png
│       └── data_visual5_real_tfidf.png
│
└── docs/                               # 연구 및 설계 산출물
    ├── 전처리 및 전체 설계 정리.pdf
    ├── 가짜뉴스_로지스틱(임시).pdf
    └── logistic.py 코드 분석(임시).pdf
```