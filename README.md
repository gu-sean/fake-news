# Fake-News

네이버 뉴스 실시간 크롤링 + 머신러닝 기반 가짜뉴스 탐지 시스템


## 1. 프로젝트 개요

TF-IDF 벡터화와 로지스틱 회귀(Logistic Regression) 기반 머신러닝 모델을 활용해 가짜 뉴스 탐지 시스템을 구축합니다.  
네이버 뉴스를 실시간으로 크롤링하고, 각 기사의 진위 여부를 AI가 분석해 사용자에게 신뢰도 점수와 함께 제공합니다.



## 2. 데이터 표준 스키마 (Schema)

| 컬럼명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `id` | int | 행 고유 식별자 (1부터 순번) |
| `title` | str | 기사 제목 (원문) |
| `content` | str | 기사 본문 (원문, 최대 10,000자 제한) |
| `media` | str | 출처 매체명 (불명 시 `Unknown` 처리) |
| `date` | str | 발행일 (`YYYY-MM-DD` 표준화, 불명 시 `YYYY-01-01` 또는 `""`) |
| `label` | int | 정답 레이블 (`0` = 진짜 뉴스, `1` = 가짜 뉴스) |
| `clean_message` | str | 학습용 정제 텍스트 (소문자화, HTML 태그 및 특수문자 전처리 제거) |
| `stat_distortion` | int | [DVL 1] 통계 왜곡 패턴 포함 여부 (`0` 또는 `1`) |
| `causal_error` | int | [DVL 2] 인과 오류 패턴 포함 여부 (`0` 또는 `1`) |
| `emotional_provocation` | int | [DVL 3] 감정 자극 패턴 포함 여부 (`0` 또는 `1`) |
| `source_lack` | int | [DVL 4] 출처 불명 패턴 포함 여부 (`0` 또는 `1`) |
| `img_mismatch` | int | [DVL 5] 이미지 불일치 패턴 포함 여부 (`0` 또는 `1`) |


## 3. 주요 특징

### TF-IDF + DVL Hybrid Architecture
* **텍스트 벡터 + 정량 지표의 결합**: TF-IDF로 추출한 고빈도 어휘 벡터(50,000차원)와, 텍스트에서 룰 기반으로 추출한 정량적 DVL 플래그(5차원)를 `hstack`으로 결합(총 50,005차원)하여 분류기에 입력하는 하이브리드 구조입니다. 어휘 패턴과 통계적 왜곡을 동시에 탐지하여 정밀도를 극대화합니다.

### 이중 언어 분리 벡터화
* **영어(Kaggle) 24,973건**: `clean_message` 기준 TF-IDF 벡터화, `stop_words='english'` 적용
* **한국어(AI Hub) 318,235건**: KoNLPy(Okt) 형태소 분석 후 TF-IDF 벡터화, 형태소 분석 결과는 `unified_news_tokenized.csv`에 캐싱하여 재실행 시 재사용
* 두 언어 모두 `unified_news_refined.csv` 단일 소스에서 `media` 컬럼으로 분리

### 네이버 뉴스 클론 사이트
* **aiohttp 비동기 크롤링**: 20개 동시 요청으로 네이버 뉴스 카테고리별 기사 실시간 수집 (Phase 1: 목록, Phase 2: 본문)
* **FastAPI + Jinja2**: 크롤링 결과에 AI 탐지 점수를 부여해 네이버 뉴스 UI 형태로 렌더링
* **AI 배지 표시**: 점수 50% 이상 → `가짜 X%` (빨간색), 50% 미만 → `진짜` (초록색)
* **전체 섹션 사전 워밍**: 서버 시작 시 모든 탭(정치·경제·사회 등)을 백그라운드에서 사전 크롤링 → 탭 전환 즉시 기사 표시
* **5분 자동 갱신**: 주기적으로 전체 캐시를 자동 새로고침, `/refresh` 엔드포인트로 강제 재크롤링 가능

### 모델 교체 가능 구조
* `src/models/` 폴더에 `.joblib` 파일만 교체하면 즉시 다른 모델로 전환
* 현재: 로지스틱 회귀(Low-level) → 추후 SVM(Mid-level) 추가 예정

### DVL (Dynamic Verification Layer) 5대 패턴 분류
가짜뉴스 특유의 통계적 왜곡과 자극적 조작 패턴을 잡아내기 위해, 본문 텍스트에서 5대 정량 플래그를 추출하여 분류 특징(Feature)으로 활용합니다.

#### 1. `stat_distortion` (통계 왜곡)
* **정의**: 수치나 빈도를 과장·절대화하는 표현을 사용해 독자에게 확정적 사실처럼 보이게 만드는 패턴입니다.
* **탐지 키워드**: `100%`, `모든`, `항상`, `절대`, `never`, `always`
* **표현 예시**: *"모든 전문가가 동의한다"*, *"항상 이래왔다"*

#### 2. `causal_error` (인과 오류)
* **정의**: 인과관계가 미약하거나 검증되지 않은 두 사건을 무리하게 원인-결과로 연결하는 패턴입니다.
* **탐지 키워드**: `때문에`, `원인`, `증명`, `결과적으로`, `causes`, `proves`
* **표현 예시**: *"A 때문에 B가 발생했다는 것이 증명됐다"*

#### 3. `emotional_provocation` (감정 자극)
* **정의**: 독자의 분노·공포·혐오 감정을 의도적으로 유발해 이성적이고 비판적인 사고를 차단하는 패턴입니다.
* **탐지 키워드**: `충격`, `경악`, `분노`, `shocking`, `outrage`, `unbelievable`
* **표현 예시**: *"충격적인 진실이 밝혀졌다"*, *"unbelievable scandal"*

#### 4. `source_lack` (출처 불명)
* **정의**: 익명의 관계자나 불특정 소식통에만 의존하여 검증 불가능한 일방적 주장을 사실처럼 서술하는 패턴입니다.
* **탐지 키워드**: `관계자`, `소식통`, `익명`, `sources say`, `reportedly`
* **표현 예시**: *"익명의 관계자에 따르면"*, *"sources say the president..."*

#### 5. `img_mismatch` (이미지 불일치)
* **정의**: 본문 내용과 유기적 연관성이 없는 과거 사진이나 왜곡된 영상을 현재 사실인 것처럼 묘사하는 패턴입니다.
* **탐지 키워드**: `사진 속`, `이 사진은`, `photo shows`, `pictured here`
* **표현 예시**: *"이 사진은 현장을 포착한 것이다"*, *"photo shows the incident"*


## 4. 프로젝트 디렉토리 구조 (Directory Structure)

```text
fake_news/
 ├── app.py                              # FastAPI 서버 — 라우트·캐시·전체 섹션 사전 워밍·5분 자동 갱신
 ├── main.py                             # 학습 실행 진입점 (CLI 래퍼)
 ├── requirements.txt                    # 패키지 의존성 목록
 ├── README.md                           # 프로젝트 설명 문서
 │
 ├── src/                                # 핵심 소스 코드
 │   ├── crawler.py                      # 네이버 뉴스 크롤러 — aiohttp 비동기 20개 동시 수집
 │   │                                   #   Phase 1: 목록 페이지 파싱 (즉시 반환)
 │   │                                   #   Phase 2: 기사 본문 백그라운드 수집 + CSV 저장
 │   ├── detector.py                     # fake_score 부여 및 최신순 정렬
 │   │                                   #   (모델 미연결 시 랜덤 점수, 연결 후 실제 예측으로 교체)
 │   ├── vectorize.py                    # TF-IDF 벡터화 마스터 스크립트 (영어·한국어 분리 처리)
 │   ├── logistic_V1.py                  # 로지스틱 회귀 V1 학습 스크립트 (→ data/vector/ 저장)
 │   ├── preprocess.py                   # 데이터 전처리 (로딩·정제·DVL 플래그 추출)
 │   └── split_dataset.py                # 메모리 방어용 대용량 데이터 균등 분할 스크립트
 │
 ├── data/                               # 데이터셋 저장 디렉토리
 │   ├── raw/                            # 원천 데이터
 │   │   ├── Fake.csv                    # 영어 가짜뉴스 (Kaggle)
 │   │   ├── True.csv                    # 영어 진짜뉴스 (Kaggle)
 │   │   ├── Fake_Real_News_Data.csv     # 영어 혼합 데이터셋 (Kaggle)
 │   │   ├── fake_real_news.arff         # ARFF 포맷 영어 혼합 데이터셋
 │   │   ├── Training/
 │   │   │   └── 02.라벨링데이터/        # AI Hub 한국어 학습 데이터 zip (42개)
 │   │   │       └── TL_Part1~2_*.zip
 │   │   └── Validation/
 │   │       └── 02.라벨링데이터/        # AI Hub 한국어 검증 데이터 zip (42개)
 │   │           └── VL_Part1~2_*.zip
 │   │
 │   ├── processed/                      # 전처리 완료 산출물
 │   │   ├── unified_news_refined.csv    # 최종 전처리 완료 데이터
 │   │   ├── unified_news_tokenized.csv  # 한국어 Okt 형태소 분석 캐시
 │   │   └── subsets/                    # OOM 방어용 균등 분할 서브셋
 │   │       └── subset_01~11.csv        # 각 30,000건 (진짜·가짜 균등 믹스)
 │   │
 │   ├── realtime/                       # 실시간 크롤링 수집 결과
 │   │   └── realtime_news.csv           # 네이버 뉴스 실시간 수집 CSV (표준 스키마 + 추가 컬럼)
 │   │
 │   └── vector/                         # TF-IDF 벡터화 결과물 (vectorize.py 생성)
 │       ├── english_tfidf.npz           # 영어 TF-IDF 희소 행렬
 │       ├── english_labels.npy          # 영어 레이블 (0=진짜, 1=가짜)
 │       ├── english_dvl_flags.npy       # 영어 DVL 5대 플래그
 │       ├── english_vectorizer.pkl      # 영어 TF-IDF 벡터라이저
 │       ├── korean_tfidf.npz            # 한국어 TF-IDF 희소 행렬
 │       ├── korean_labels.npy           # 한국어 레이블
 │       ├── korean_dvl_flags.npy        # 한국어 DVL 5대 플래그
 │       └── korean_vectorizer.pkl       # 한국어 TF-IDF 벡터라이저
 │
 ├── templates/                          # Jinja2 HTML 템플릿 + 정적 파일
 │   ├── naver_news_clone.html           # 네이버 뉴스 클론 UI (AI 탐지 배지·카테고리·페이지네이션)
 │   ├── newshome.css                    # 메인 레이아웃 스타일시트
 │   └── news.css                        # 기사 상세 페이지 스타일시트
 │
 ├── visualization/                      # 데이터 시각화
 │   ├── src/
 │   │   ├── data_visual1.py             # 클래스 분포 차트 생성 스크립트
 │   │   ├── data_visual2.py             # 단어 빈도 차트 생성 스크립트
 │   │   └── data_visual3.py             # TF-IDF 시각화 스크립트
 │   └── img/
 │       ├── data_visual1_class_distribution.png   # 클래스 분포 차트
 │       ├── data_visual2_word_frequency.png        # 단어 빈도 차트
 │       └── data_visual3_tfidf.png                 # TF-IDF 시각화 이미지
 │
 └── docs/                               # 연구 및 설계 산출물
     ├── logistic.py 코드 분석(임시).pdf
     ├── 가짜뉴스_로지스틱(임시).pdf
     └── 전처리 및 전체 설계 정리.pdf
```