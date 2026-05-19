# 학습 실행 진입점

import argparse
import sys
import subprocess
from pathlib import Path

root_dir = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="DVL Fake News Detector - 학습 실행")
    parser.add_argument("--data_path",      default="data/processed/unified_news_refined.csv",
                        help="전처리 완료된 CSV 파일 경로")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="모델 체크포인트 저장 경로")
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--freeze_layers",  type=int,   default=0,
                        help="하위 N개 인코더 레이어 고정 (메모리 절약용)")
    parser.add_argument("--result_path",   default="data/processed/training_results.txt",
                        help="학습 결과를 저장할 txt 파일 경로")
    args = parser.parse_args()

    data_path = root_dir / args.data_path
    if not data_path.exists():
        print(f"오류: 전처리된 데이터 파일이 없습니다: {data_path}")
        print("먼저 전처리를 실행하세요: python src/preprocess.py")
        sys.exit(1)

    train_script = root_dir / "src" / "train.py"
    if not train_script.exists():
        print(f"오류: {train_script} 파일이 없습니다.")
        sys.exit(1)

    print("=== 학습 시작 ===")
    try:
        subprocess.run(
            [sys.executable, "-m", "src.train",
             "--data_path",      args.data_path,
             "--checkpoint_dir", args.checkpoint_dir,
             "--epochs",         str(args.epochs),
             "--batch_size",     str(args.batch_size),
             "--freeze_layers",  str(args.freeze_layers),
             "--result_path",    args.result_path],
            check=True,
            cwd=str(root_dir),
        )
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: 학습 실패 (exit code {e.returncode})")
        sys.exit(1)

    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()
