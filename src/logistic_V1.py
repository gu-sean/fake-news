import pandas as pd
import glob
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 1. 데이터 로드 (Chunking 방식으로 메모리 방어)
# ==========================================
path = os.path.dirname(os.path.abspath(__file__))
all_files = glob.glob(os.path.join(path, "*.csv"))

if not all_files:
    print("🚨 CSV 파일을 찾을 수 없습니다!")
    sys.exit()

use_cols = ['text', 'clean_message', 'label'] 
df_list = []

# 파일당 최대 읽어올 행 수
MAX_ROWS_PER_FILE = 30

print("📂 메모리 절약 모드로 데이터를 로드 중...")
for filename in all_files:
    try:
        df = pd.read_csv(filename, usecols=use_cols, nrows=MAX_ROWS_PER_FILE, engine='c')
        df_list.append(df)
        print(f"✅ {os.path.basename(filename)} 로드 성공")
    except Exception as e:
        print(f"❌ {os.path.basename(filename)} 로드 실패: {e}")

if not df_list:
    print("🚨 결합할 데이터가 없습니다.")
    sys.exit()

total_df = pd.concat(df_list, axis=0, ignore_index=True)
total_df = total_df.dropna(subset=['text', 'clean_message', 'label'])

# 두 컬럼 결합
X = total_df['text'].astype(str) + " " + total_df['clean_message'].astype(str)
y = total_df['label'].astype(int)
print(f"✅ 학습 준비 완료 (샘플링된 데이터: {len(total_df)}행)")

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. max_features 사용자 입력받기
# ==========================================
print("\n" + "="*50)
print("⚙️ TF-IDF max_features 범위를 설정합니다.")
try:
    START_FEATURES = int(input("▶️ 초기값을 입력하세요 (예: 50000): "))
    END_FEATURES = int(input("⏹️ 종료값을 입력하세요 (예: 250000): "))
    STEP_FEATURES = int(input("🔄 증가값을 입력하세요 (예: 50000): "))
except ValueError:
    print("🚨 오류: 숫자만 입력해야 합니다! 프로그램을 종료합니다.")
    sys.exit()
print("="*50)

# ==========================================
# 3. 자동화 테스트 및 TXT 저장
# ==========================================
output_txt = "model_results_log.txt" # 저장될 메모장(txt) 파일명

# 텍스트 파일 초기화 및 헤더 작성 ('w' 모드)
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("🚀 TF-IDF max_features 자동화 테스트 결과 🚀\n")
    f.write(f"설정 범위: {START_FEATURES} ~ {END_FEATURES} (증가값: {STEP_FEATURES})\n")
    f.write("="*60 + "\n")

print(f"\n🚀 max_features {START_FEATURES}부터 {END_FEATURES}까지 {STEP_FEATURES} 간격으로 테스트를 시작합니다...")

for max_feat in range(START_FEATURES, END_FEATURES + 1, STEP_FEATURES):
    print(f"\n==========================================")
    print(f"🔄 TF-IDF 변환 중 (max_features={max_feat})...")
    
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer(max_features=max_feat) 
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("🧠 로지스틱 회귀 학습 중...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)
    lr_model.fit(X_train_tfidf, y_train)

    # 예측 및 평가
    y_pred = lr_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"🎯 정확도: {acc:.4f} / F1-Score: {f1:.4f}")

    # 핵심 단어 추출 (상위 5개)
    feature_names = np.array(tfidf.get_feature_names_out())
    coefficients = lr_model.coef_[0]
    indices = np.argsort(coefficients)
    
    top_keywords = [f"{feature_names[i]}({coefficients[i]:.2f})" for i in reversed(indices[-5:])]
    keywords_str = ", ".join(top_keywords)

    # 🔹 결과를 TXT 파일에 추가 기록 ('a' 모드)
    with open(output_txt, 'a', encoding='utf-8') as f:
        f.write(f"\n🔹 [max_features: {max_feat}]\n")
        f.write(f"  - 정확도(Accuracy)  : {acc:.4f}\n")
        f.write(f"  - 정밀도(Precision) : {precision:.4f}\n")
        f.write(f"  - 재현율(Recall)    : {recall:.4f}\n")
        f.write(f"  - F1-Score          : {f1:.4f}\n")
        f.write(f"  - 🚩 가짜뉴스 핵심 단어 Top 5: {keywords_str}\n")
        f.write("-" * 60 + "\n")

print("\n" + "="*50)
print(f"✅ 모든 테스트 완료! 결과가 성공적으로 '{output_txt}' 파일에 기록되었습니다.")
print("="*50)