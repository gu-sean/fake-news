"""
통합 실행 진입점

사용법:
  python main.py --step all        # 벡터화 → 학습 → 서버 순서대로
  python main.py --step vectorize  # TF-IDF 벡터화만
  python main.py --step train      # 모델 학습만
  python main.py --step server     # 웹 서버만

옵션:
  --lang   korean / english / both  (train 단계에서 사용, 기본: both)
  --port   웹 서버 포트 (기본: 8000)
"""

import argparse
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], label: str):
    print(f'\n{"="*50}')
    print(f' {label}')
    print(f'{"="*50}')
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] {label} 실패 (exit code {e.returncode})')
        sys.exit(1)


def step_vectorize():
    vec_files = list((ROOT / 'data' / 'vector').glob('*.npz'))
    if vec_files:
        print('[vectorize] 벡터 파일이 이미 존재합니다. 건너뜁니다.')
        print('  재생성하려면 python src/vectorize.py 를 직접 실행하세요.')
        return
    run([sys.executable, 'src/vectorize.py'], 'TF-IDF 벡터화')


def step_train(lang: str):
    run(
        [sys.executable, 'src/logistic.py', '--lang', lang],
        f'로지스틱 회귀 학습 ({lang})',
    )


def step_server(port: int):
    print(f'\n{"="*50}')
    print(f' FastAPI 웹 서버 시작')
    print(f' http://localhost:{port}')
    print(f' API 문서: http://localhost:{port}/docs')
    print(f'{"="*50}')
    subprocess.run(
        [sys.executable, '-m', 'uvicorn', 'app:app',
         '--host', '0.0.0.0', '--port', str(port)],
        cwd=str(ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description='DVL Fake News Detector')
    parser.add_argument(
        '--step',
        choices=['all', 'vectorize', 'train', 'server'],
        default='server',
        help='실행할 단계 (기본: server)',
    )
    parser.add_argument(
        '--lang',
        choices=['korean', 'english', 'both'],
        default='both',
        help='학습 언어 선택 (기본: both)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='웹 서버 포트 (기본: 8000)',
    )
    args = parser.parse_args()

    # 필수 파일 존재 확인
    if not (ROOT / 'data' / 'processed' / 'unified_news_refined.csv').exists():
        print('[ERROR] data/processed/unified_news_refined.csv 가 없습니다.')
        print('  먼저 전처리를 실행하세요: python src/preprocess.py')
        sys.exit(1)

    if args.step == 'all':
        step_vectorize()
        step_train(args.lang)
        step_server(args.port)

    elif args.step == 'vectorize':
        step_vectorize()

    elif args.step == 'train':
        # 벡터 파일 없으면 먼저 벡터화 실행
        vec_files = list((ROOT / 'data' / 'vector').glob('*.npz'))
        if not vec_files:
            print('[train] 벡터 파일이 없어 벡터화를 먼저 실행합니다.')
            step_vectorize()
        step_train(args.lang)

    elif args.step == 'server':
        step_server(args.port)


if __name__ == '__main__':
    main()
