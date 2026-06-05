import pandas as pd
import os
import sys
import numpy as np
import joblib # 💾 모델 저장을 위해 추가된 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 1. 훈련용 데이터 로드 (DACON 데이터셋 맞춤)
# ==========================================
path = os.path.dirname(os.path.abspath(__file__))
train_filename = os.path.join(path, "subset_01.csv") # 훈련용 파일 고정

if not os.path.exists(train_filename):
    print(f"🚨 훈련용 데이터({train_filename})를 찾을 수 없습니다!")
    sys.exit()

# 데이콘 데이터셋 컬럼에 맞게 수정
use_cols = ['title', 'content', 'label'] 

print("📂 훈련용 데이터를 로드 중...")
try:
    total_df = pd.read_csv(train_filename, usecols=use_cols)
    print(f"✅ {os.path.basename(train_filename)} 로드 성공")
except Exception as e:
    print(f"❌ 데이터 로드 실패: {e}")
    sys.exit()

total_df = total_df.dropna(subset=['title', 'content', 'label'])

# 두 컬럼(제목+본문) 결합
X = total_df['title'].astype(str) + " " + total_df['content'].astype(str)
y = total_df['label'].astype(int) # label 대신 info
print(f"✅ 학습 준비 완료 (총 데이터: {len(total_df)}행)")

# 학습/테스트 데이터 분할 (내부 모의고사용)
# 데이터가 너무 적을 경우를 대비해 예외 처리 추가
if len(total_df) < 10:
    print("🚨 데이터가 너무 적어 분할할 수 없습니다. 더 많은 데이터를 확보해주세요.")
    sys.exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. max_features 사용자 입력받기
# ==========================================
print("\n" + "="*50)
print("⚙️ TF-IDF max_features 범위를 설정합니다.")
try:
    START_FEATURES = int(input("▶️ 초기값을 입력하세요 (예: 1000): "))
    END_FEATURES = int(input("⏹️ 종료값을 입력하세요 (예: 5000): "))
    STEP_FEATURES = int(input("🔄 증가값을 입력하세요 (예: 1000): "))
except ValueError:
    print("🚨 오류: 숫자만 입력해야 합니다! 프로그램을 종료합니다.")
    sys.exit()
print("="*50)

# ==========================================
# 3. 자동화 테스트 및 TXT 기록, 최고 모델 저장
# ==========================================
output_txt = "model_results_log.txt"

with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("🚀 TF-IDF max_features 자동화 테스트 결과 🚀\n")
    f.write(f"설정 범위: {START_FEATURES} ~ {END_FEATURES} (증가값: {STEP_FEATURES})\n")
    f.write("="*60 + "\n")

print(f"\n🚀 max_features {START_FEATURES}부터 {END_FEATURES}까지 {STEP_FEATURES} 간격으로 테스트를 시작합니다...")

# 최고 성능 모델 저장을 위한 변수 초기화
best_f1 = 0
best_max_feat = 0

for max_feat in range(START_FEATURES, END_FEATURES + 1, STEP_FEATURES):
    print(f"\n==========================================")
    print(f"🔄 TF-IDF 변환 중 (max_features={max_feat})...")
    
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer(max_features=max_feat) 
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("🧠 로지스틱 회귀 학습 중...")
    # n_jobs=-1 삭제하여 경고(Warning) 제거
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    # 예측 및 평가
    y_pred = lr_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"🎯 정확도: {acc:.4f} / F1-Score: {f1:.4f}")

    # 최고 성능 갱신 시 모델 덮어쓰기 저장 💾
    if f1 > best_f1:
        best_f1 = f1
        best_max_feat = max_feat
        joblib.dump(tfidf, 'best_vectorizer.pkl')
        joblib.dump(lr_model, 'best_model.pkl')
        print(f"🌟 최고 성능 갱신! 모델 저장됨 (F1: {best_f1:.4f})")

    # 핵심 단어 추출 (상위 5개)
    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = lr_model.coef_[0]
    indices = np.argsort(coefficients)
    
    # 단어장이 너무 작을 경우 예외처리
    top_n = min(5, len(feature_names))
    top_keywords = [f"{feature_names[i]}({coefficients[i]:.2f})" for i in reversed(indices[-top_n:])]
    keywords_str = ", ".join(top_keywords)

    # 결과를 TXT 파일에 추가 기록
    with open(output_txt, 'a', encoding='utf-8') as f:
        f.write(f"\n🔹 [max_features: {max_feat}]\n")
        f.write(f"  - 정확도(Accuracy)  : {acc:.4f}\n")
        f.write(f"  - 정밀도(Precision) : {precision:.4f}\n")
        f.write(f"  - 재현율(Recall)    : {recall:.4f}\n")
        f.write(f"  - F1-Score          : {f1:.4f}\n")
        f.write(f"  - 🚩 가짜뉴스 핵심 단어 Top {top_n}: {keywords_str}\n")
        f.write("-" * 60 + "\n")

print("\n" + "="*50)
print(f"✅ 모든 테스트 완료! 결과가 '{output_txt}' 파일에 기록되었습니다.")
print(f"🏆 가장 성능이 좋았던 모델(max_features={best_max_feat}, F1={best_f1:.4f})이 '.pkl' 파일로 저장되었습니다.")
print("="*50)