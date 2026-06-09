"""
팩트체크 수집 데이터를 기존 학습 데이터에 합치는 스크립트

사용법:
  python src/merge_factcheck.py              # 합치기만
  python src/merge_factcheck.py --revectorize  # 합치기 + 재벡터화
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_CSV = os.path.join(BASE_DIR, "data", "processed", "unified_news_refined.csv")
FACTCHECK_CSV = os.path.join(BASE_DIR, "data", "factcheck", "factcheck_label.csv")
BACKUP_CSV   = os.path.join(BASE_DIR, "data", "processed", "unified_news_refined_backup.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revectorize", action="store_true",
                        help="합친 후 자동으로 재벡터화 실행")
    args = parser.parse_args()

    # ── 파일 확인 ───────────────────────────────────────────────────────────────
    if not os.path.exists(FACTCHECK_CSV):
        print("[오류] 팩트체크 데이터가 없습니다.")
        print("먼저 실행: python src/factcheck_crawler.py")
        sys.exit(1)

    df_new = pd.read_csv(FACTCHECK_CSV, encoding="utf-8-sig")
    print(f"팩트체크 데이터: {len(df_new)}건 (가짜: {(df_new['label']==1).sum()}, 진짜: {(df_new['label']==0).sum()})")

    df_old = pd.read_csv(ORIGINAL_CSV, encoding="utf-8-sig")
    print(f"기존 데이터: {len(df_old)}건")

    # ── 백업 ────────────────────────────────────────────────────────────────────
    df_old.to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")
    print(f"기존 데이터 백업: {BACKUP_CSV}")

    # ── 컬럼 정렬 후 합치기 ─────────────────────────────────────────────────────
    common_cols = [c for c in df_old.columns if c in df_new.columns]
    df_merged = pd.concat([df_old, df_new[common_cols]], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["title"], keep="first")

    df_merged.to_csv(ORIGINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n합친 데이터 저장: {ORIGINAL_CSV}")
    print(f"  전체: {len(df_merged)}건")
    print(f"  진짜(0): {(df_merged['label']==0).sum():,}건")
    print(f"  가짜(1): {(df_merged['label']==1).sum():,}건")

    # ── 재벡터화 ────────────────────────────────────────────────────────────────
    if args.revectorize:
        print("\n재벡터화 시작...")
        subprocess.run([sys.executable, "src/vectorize.py"], cwd=BASE_DIR, check=True)
        print("재벡터화 완료. 이제 재학습을 실행하세요:")
        print("  python src/xgboost_train_pipeline.py --lang korean")
    else:
        print("\n다음 단계:")
        print("  python src/vectorize.py           # 재벡터화")
        print("  python src/xgboost_train_pipeline.py --lang korean  # 재학습")


if __name__ == "__main__":
    main()
