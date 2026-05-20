# 모델 예측, 정렬
import os
import pickle
import joblib
import numpy as np
from scipy.sparse import hstack

# 경로 설정 (프로젝트 루트 기준 파일 위치 지정)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")
VEC_DIR   = os.path.join(BASE_DIR, "data", "vector")

# 1. 서버 시작 시 디스크에서 벡터라이저와 학습된 모델을 원타임 로드
try:
    with open(os.path.join(VEC_DIR, "english_vectorizer.pkl"), "rb") as f:
        VECTORIZER = pickle.load(f)
    MODEL = joblib.load(os.path.join(MODEL_DIR, "english_logistic.joblib"))
    print("[SUCCESS] 가짜뉴스 탐지 AI 모델 및 벡터 사전 로드 완료")
except Exception as e:
    VECTORIZER = None
    MODEL = None
    print(f"[WARNING] 모델 파일을 찾을 수 없어 더미 로직으로 가동합니다. 에러: {e}")


def predict_fake_probability(title: str, text: str, dvl_flags: list) -> float:
    # 모델 파일이 정상적으로 로드된 경우 (실제 가동)
    if VECTORIZER and MODEL:
        full_text = title + " " + text
        
        # 1. 입력받은 실시간 텍스트를 5,000차원 희소 행렬로 변환
        live_tfidf = VECTORIZER.transform([full_text])
        
        # 2. 5대 왜곡 지표(DVL 플래그)를 2차원 리스트 형태로 생성
        live_dvl = [dvl_flags]  # 예: [[0, 1, 0, 0, 1]]
        
        # 3. 학습할 때와 동일하게 텍스트 행렬 옆에 가로로 결합
        live_final = hstack([live_tfidf, live_dvl])
        
        # 4. 모델 추론 (SVM 계열일 경우 decision_function 후 시그모이드 변환)
        score = MODEL.decision_function(live_final)[0]
        prob = 1 / (1 + np.exp(-score))  # 0.0 ~ 1.0 사이 확률 맵핑
        return round(float(prob) * 100, 1)  # 백분율 화면 표시용 (예: 84.5)
        
    # 모델 파일이 아직 없을 때 시연 및 레이아웃 확인용 더미 로직
    else:
        title_lower = title.lower()
        if "충격" in title_lower or "단독" in title_lower or "모든 세금 면제" in title_lower:
            return 94.2
        elif "발견" in title_lower or "속보" in title_lower:
            return 78.5
        return 4.2


def get_sorted_timeline(raw_news_list: list) -> list:
    processed_news = []
    for news in raw_news_list:
        # 임시로 모든 기사에 DVL 플래그 더미값 [0, 0, 1, 0, 0] 주입
        # 나중에 크롤링 결과 데이터나 유저 인풋에 맞춰 연동하시면 됩니다.
        dummy_flags = [0, 0, 1, 0, 0] 
        
        score = predict_fake_probability(news["title"], news.get("text", ""), dummy_flags)
        news["fake_score"] = score
        processed_news.append(news)
        
    # 오름차순 정렬 (가짜 확률이 가장 낮은 진짜 뉴스가 0번 인덱스, 즉 최상단으로 배치)
    return sorted(processed_news, key=lambda x: x["fake_score"])