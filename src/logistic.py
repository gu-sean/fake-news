import pandas as pd
import glob
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. 데이터 로드 (Chunking 방식으로 메모리 방어)
# ==========================================
path = os.path.dirname(os.path.abspath(__file__))
all_files = glob.glob(os.path.join(path, "*.csv"))

if not all_files:
    print("🚨 CSV 파일을 찾을 수 없습니다!")
    sys.exit()

use_cols = ['content', 'clean_message', 'label'] 
df_list = []

# 파일당 최대 읽어올 행 수 (메모리 상황에 따라 20000~50000 조절)
MAX_ROWS_PER_FILE = 30000 

print("📂 메모리 절약 모드로 데이터를 로드 중...")
for filename in all_files:
    try:
        # nrows를 사용하여 파일의 일부분만 읽어와 메모리 폭발 방지
        df = pd.read_csv(filename, usecols=use_cols, nrows=MAX_ROWS_PER_FILE, engine='c')
        df_list.append(df)
        print(f"✅ {os.path.basename(filename)} 로드 성공")
    except Exception as e:
        print(f"❌ {os.path.basename(filename)} 로드 실패: {e}")

if not df_list:
    print("🚨 결합할 데이터가 없습니다.")
    sys.exit()

total_df = pd.concat(df_list, axis=0, ignore_index=True)
total_df = total_df.dropna(subset=['content', 'clean_message', 'label'])

# 두 컬럼 결합
X = total_df['content'].astype(str) + " " + total_df['clean_message'].astype(str)
y = total_df['label'].astype(int)
print(f"✅ 학습 준비 완료 (샘플링된 데이터: {len(total_df)}행)")

# ==========================================
# 2. 데이터 분할 및 벡터화
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🔄 TF-IDF 변환 중 (메모리 보호 모드)...")
# max_features를 10000으로 낮춰 메모리 점유율 추가 감소
tfidf = TfidfVectorizer(max_features=30000) 
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==========================================
# 3. 모델 학습 및 가중치 분석
# ==========================================
print("🧠 로지스틱 회귀 학습 중...")
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)
lr_model.fit(X_train_tfidf, y_train)

# 결과 출력
y_pred = lr_model.predict(X_test_tfidf)
print("\n" + "="*50)
print(f"🎯 샘플링 모델 정확도: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)
print(classification_report(y_test, y_pred))

# 핵심 단어 추출
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = lr_model.coef_[0]
indices = np.argsort(coefficients)

print("\n🚩 [가짜뉴스(1) 핵심 키워드 Top 10]")
for i in reversed(indices[-10:]):
    print(f"{feature_names[i]:<15} | 가중치: {coefficients[i]:.4f}")