# Fake-News

뉴스 크롤링 및 모델 기반 가짜뉴스 판별 시스템


## 1. 프로젝트 개요

딥러닝 기반 자연어 처리(NLP)를 활용해 가짜 뉴스를 실시간으로 탐지하고, 정보의 신뢰도를 확보하여 사회적 혼란을 방지합니다.


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

### Hybrid Architecture 
* **문맥 의미 분석 + 정량 검증의 결합**: `klue/roberta-base` 모델이 추출한 고차원 문맥 벡터(768차원)와, 텍스트에서 룰 기반으로 추출한 정량적 DVL 플래그(5차원)를 결합(`torch.cat`)하여 최종 분류기(MLP)를 통과시키는 하이브리드 신경망 구조입니다. 의미론적 흐름과 통계적 왜곡을 동시에 탐지하여 정밀도를 극대화합니다.

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
 ├── main.py                        # 학습 실행 진입점 (CLI 래퍼)
 ├── requirements.txt               # 패키지 의존성 목록
 ├── README.md                      # 프로젝트 설명 문서
 ├── .gitignore                     # Git 관리 제외 파일 설정
 │
 ├── src/                           # 프로젝트 핵심 소스 코드 패키지
 │   ├── preprocess.py              # 데이터 전처리 (로딩·7단계 정제·DVL 필터링 통합)
 │   ├── split_dataset.py           # 메모리(RAM) 방어용 대용량 데이터 분할 스크립트
 │   ├── dataset.py                 # Lazy Tokenization & Dynamic Padding 구현 PyTorch 데이터셋
 │   ├── model.py                   # KLUE-RoBERTa + DVL 결합 하이브리드 MLP 신경망 구조
 │   └── train.py                   # 차등 학습률(Differential LR) 최적화 기반 학습 코어 스크립트
 │
 ├── data/                          # 데이터셋 저장 디렉토리
 │   ├── raw/                       # 원천 데이터 백업 디렉토리 
 │   │   ├── Fake.csv               # 영어 가짜뉴스 원본 데이터셋
 │   │   ├── True.csv               # 영어 진짜뉴스 원본 데이터셋
 │   │   ├── Fake_Real_News_Data.csv # 믹스드 가짜뉴스 데이터셋
 │   │   ├── Training/
 │   │   │   └── 02.라벨링데이터/   # AI HUB 한국어 데이터 zip 파트 (42개)
 │   │   └── Validation/
 │   │       └── 02.라벨링데이터/   # AI HUB 한국어 데이터 zip 파트 (42개)
 │   │
 │   └── processed/                 # 전처리 완료 산출물 디렉토리
 │       ├── unified_news_refined.csv # 7단계 파이프라인 완결 전체 데이터 (343,208건)
 │       ├── training_results.txt   # 에포크별 Loss, Accuracy, F1-Score 학습 결과 요약본
 │       └── subsets/               # split_dataset.py 실행 결과물 (OOM 방어 목적)
 │           ├── subset_01.csv      # 진짜/가짜 각각 15,000건 균등 분할 믹스 (총 30,000건)
 │           ├── subset_02.csv      # 각 subset별 독립적 인덱스(id) 재부여
 │           └── ... (subset_11.csv 까지 균등 분할 배치 완료)
 │
 ├── checkpoints/                   # 학습 완료 모델 가중치 저장소
 │   └── best_model.pt              # Validation 세트 기준 macro-F1 최고 성능 가중치 파일
 │
 └── docs/                          # 프로젝트 연구 및 설계 산출물 아카이빙 폴더
     └── 전처리_및_전체_설계_정리.pdf # 연구 방법론 및 정밀 설계서 전문 (PDF)

