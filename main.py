import argparse
import sys
import subprocess
from pathlib import Path

root_dir = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="DVL Fake News Detector")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "refine", "all"],
        default="preprocess",
        help="실행 단계 선택 (preprocess: 원본→통합, refine: 추가 정제, all: 전체)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for p in ["data/raw", "data/processed"]:
        (root_dir / p).mkdir(parents=True, exist_ok=True)

    print(f"=== 실행 단계: {args.stage} ===")

    processed_file = str(root_dir / "data" / "processed" / "unified_news_processed.csv")
    refined_file   = str(root_dir / "data" / "processed" / "unified_news_refined.csv")

    if args.stage in ("preprocess", "all"):
        print("전처리를 시작합니다 (src/preprocess.py 호출)...")

        preprocess_script = root_dir / "src" / "preprocess.py"
        raw_dir = str(root_dir / "data" / "raw")

        if not preprocess_script.exists():
            print(f"오류: {preprocess_script} 파일이 없습니다.")
            sys.exit(1)

        try:
            subprocess.run(
                [sys.executable, str(preprocess_script),
                "--raw_dir", raw_dir,
                "--output", processed_file],
                check=True,
                cwd=str(root_dir),
            )
        except subprocess.CalledProcessError as e:
            print(f"오류 발생: 전처리 스크립트 실행 실패 (exit code {e.returncode})")
            sys.exit(1)

    if args.stage in ("refine", "all"):
        print("추가 정제를 시작합니다 (src/refine.py 호출)...")

        refine_script = root_dir / "src" / "refine.py"

        if not refine_script.exists():
            print(f"오류: {refine_script} 파일이 없습니다.")
            sys.exit(1)

        if not Path(processed_file).exists():
            print(f"오류: 입력 파일이 없습니다 → {processed_file}")
            print("먼저 'python main.py --stage preprocess' 를 실행하세요.")
            sys.exit(1)

        try:
            subprocess.run(
                [sys.executable, str(refine_script),
                "--input",  processed_file,
                "--output", refined_file],
                check=True,
                cwd=str(root_dir),
            )
        except subprocess.CalledProcessError as e:
            print(f"오류 발생: 정제 스크립트 실행 실패 (exit code {e.returncode})")
            sys.exit(1)

    print("모든 공정이 완료되었습니다.")


if __name__ == "__main__":
    main()