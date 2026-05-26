import pandas as pd
import numpy as np
import joblib
import os
import sys

# ==========================================
# 1. 파일 경로 설정 및 크롤링 데이터 로드
# ==========================================
path = os.path.dirname(os.path.abspath(__file__))
crawl_filename = os.path.join(path, "realtime_news.csv") # 질문자님의 크롤링 파일명

if not os.path.exists(crawl_filename):
    print(f"🚨 크롤링 데이터({crawl_filename})를 찾을 수 없습니다!")
    sys.exit()

print("📂 실전용 크롤링 데이터를 로드 중...")
try:
    crawl_df = pd.read_csv(crawl_filename)
    print(f"✅ 데이터 로드 성공 (총 {len(crawl_df)}행의 기사 발견)")
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    sys.exit()

# ==========================================
# 2. 1단계에서 저장한 AI 모델과 단어장 불러오기
# ==========================================
vectorizer_path = os.path.join(path, 'best_vectorizer.pkl')
model_path = os.path.join(path, 'best_model.pkl')

if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    print("🚨 학습된 모델 파일이 없습니다! 1단계(훈련) 코드를 먼저 실행해 '.pkl' 파일을 만들어주세요.")
    sys.exit()

print("\n🧠 학습된 AI 모델과 단어장을 불러오는 중...")
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)
print("✅ 모델 장착 완료!")

# ==========================================
# 3. 데이터 전처리 (학습 때와 동일한 형식으로 맞추기)
# ==========================================
# 질문자님의 파일 구조에 맞게 제목(title)과 본문(clean_message 또는 text)을 합칩니다.
text_col = 'clean_message' if 'clean_message' in crawl_df.columns else 'text'

# 결측치(빈칸)가 있으면 에러가 나므로 빈 문자열로 채워줍니다.
crawl_df['title'] = crawl_df['title'].fillna("")
crawl_df[text_col] = crawl_df[text_col].fillna("")

# 텍스트 결합
X_real = crawl_df['title'].astype(str) + " " + crawl_df[text_col].astype(str)

# ==========================================
# 4. 실전 예측 진행 (가장 중요한 부분 ⭐️)
# ==========================================
print("\n🔄 실전 기사를 모델이 이해할 수 있는 숫자로 변환 중...")
# 절대 fit_transform이 아닙니다! 학습된 단어장을 활용해 transform(번역)만 수행합니다.
X_real_tfidf = vectorizer.transform(X_real)

print("🎯 AI가 가짜뉴스를 판별 중입니다...")
# 0(진짜) 또는 1(가짜) 라벨 예측
crawl_df['predicted_label'] = model.predict(X_real_tfidf) 

# 가짜뉴스(1)일 확률을 0~100% 사이의 수치로 계산
crawl_df['fake_probability(%)'] = np.round(model.predict_proba(X_real_tfidf)[:, 1] * 100, 2)

# 가독성을 위해 한글 결과 컬럼 추가
crawl_df['결과_텍스트'] = crawl_df['predicted_label'].map({0: '✅ 진짜뉴스', 1: '🚨 가짜뉴스'})

# ==========================================
# 5. 최종 결과물 CSV 파일로 저장
# ==========================================
output_filename = os.path.join(path, "realtime_news_result.csv")
# 한글 깨짐 방지를 위해 utf-8-sig 인코딩 사용
crawl_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print(f"🎉 판별 완료! 최종 결과가 '{os.path.basename(output_filename)}' 파일로 저장되었습니다.")
print("="*60)

# 결과 미리보기 (상위 5개만 화면에 출력)
print("\n👀 [판별 결과 미리보기 (상위 5개)]")
preview_cols = ['title', '결과_텍스트', 'fake_probability(%)']
print(crawl_df[preview_cols].head().to_string(index=False))