import itertools
import os
import sys
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DIR = os.path.join(SCRIPT_DIR, "vector")

TFIDF_PATH = os.path.join(VECTOR_DIR, "korean_tfidf.npz")

LABEL_PATH = os.path.join(VECTOR_DIR, "korean_labels.npy")

DVL_FLAG_PATH = os.path.join(VECTOR_DIR, "korean_dvl_flags.npy")

# dvl_flags의 각 열에 사람이 읽을 수 있는 이름을 붙이기 위한 목록입니다.
PROBLEM_TYPE_COLUMNS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

LR_GRID = {
    "C": [0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0],
    "penalty": ["l1"],
    "solver": ["liblinear"],
    "class_weight": [None],
    "tol": [3e-4],
    "l1_ratio": [None],
}

LOG_PATH = os.path.join(SCRIPT_DIR, "logistic_V4_2_log.txt")

BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model_V4_2.pkl")

BEST_CONFIG_PATH = os.path.join(SCRIPT_DIR, "best_config_V4_2.txt")

RANDOM_STATE = 42

# 전체 데이터 중 20%를 test set으로 사용합니다.
TEST_SIZE = 0.2

# 로지스틱 회귀 최적화의 최대 반복 횟수입니다.
MAX_ITER = 3000

# 기본 실행 횟수입니다.
DEFAULT_MAX_TUNING_RUNS = 12

# LR_GRID의 모든 후보 조합 중 실제 사용 가능한 조합만 하나씩 반환합니다.
def iter_param_grid(grid):
    
    keys = list(grid.keys())

    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        if is_valid_lr_config(params):
            yield params

# LogisticRegression에서 허용되는 solver/penalty/l1_ratio 조합인지 확인합니다.
def is_valid_lr_config(params):
    
    penalty = params["penalty"]
    solver = params["solver"]
    l1_ratio = params["l1_ratio"]

    # liblinear는 l1, l2를 지원하지만 elasticnet은 지원하지 않습니다.
    # l1_ratio는 elasticnet 전용이므로 liblinear에서는 None이어야 합니다.
    if solver == "liblinear":
        return penalty in {"l1", "l2"} and l1_ratio is None

    # saga는 l1, l2, elasticnet을 모두 지원합니다.
    if solver == "saga":
        if penalty in {"l1", "l2"}:
            return l1_ratio is None
        if penalty == "elasticnet":
            return l1_ratio is not None

    # 위 조건에 해당하지 않으면 잘못된 조합입니다.
    return False

# 환경변수에서 양의 정수를 읽어옵니다. 없거나 잘못되면 기본값을 사용합니다.
def get_env_int(name, default=None):
    
    # 예: MAX_TUNING_RUNS=5로 실행하면 5개 조합만 빠르게 테스트할 수 있습니다.
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    return parsed if parsed > 0 else default

# 환경변수에서 0과 1 사이의 실수를 읽어옵니다.
def get_env_float(name, default):
    
    # 예: TEST_SIZE=0.3으로 실행하면 test set 비율을 30%로 바꿀 수 있습니다.
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = float(value)
    except ValueError:
        return default

    # train_test_split의 test_size는 0과 1 사이여야 합니다.
    return parsed if 0 < parsed < 1 else default

# 콘솔 출력과 로그 파일 기록을 동시에 수행합니다.
def append_log(message):
    
    # flush=True를 사용해 장시간 학습 중에도 메시지가 바로 보이게 합니다.
    print(message, flush=True)

    # 로그 파일에 같은 메시지를 한 줄씩 추가합니다.
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
        log_file.flush()

# 파라미터 dict를 로그에 넣기 좋은 한 줄 문자열로 바꿉니다.
def format_params(params):
    
    formatted = []

    for key, value in params.items():
        # None은 사람이 읽기 좋게 default로 표시합니다.
        formatted.append(f"{key}={value if value is not None else 'default'}")

    return ", ".join(formatted)

# 학습에 필요한 세 개의 vector 파일이 존재하는지 확인합니다.
def validate_input_files():
    
    missing = [
        path for path in [TFIDF_PATH, LABEL_PATH, DVL_FLAG_PATH]
        if not os.path.exists(path)
    ]

    # 파일이 없으면 뒤에서 복잡한 오류가 나기 전에 바로 종료합니다.
    if missing:
        print("Required vector files were not found:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

# TF-IDF 행렬, 라벨, 문제 유형 플래그를 읽고 기본 검사를 수행합니다.
def load_vector_data():
    
    # 먼저 파일 존재 여부부터 확인합니다.
    validate_input_files()

    # TF-IDF는 대부분 0인 고차원 데이터이므로 sparse matrix로 저장되어 있습니다.
    X = sparse.load_npz(TFIDF_PATH)

    # 라벨과 문제 유형 플래그는 numpy 배열로 불러옵니다.
    y = np.load(LABEL_PATH)
    dvl_flags = np.load(DVL_FLAG_PATH)

    # scikit-learn 선형 모델에 안정적으로 넣기 위해 CSR 형식으로 맞춥니다.
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    # 라벨은 1차원 정수 배열로 맞춥니다.
    y = np.asarray(y).astype(int).ravel()

    # 문제 유형 플래그도 numpy 배열로 보장합니다.
    dvl_flags = np.asarray(dvl_flags)

    # 플래그가 1차원으로 저장된 경우, 열이 1개인 2차원 배열로 변환합니다.
    if dvl_flags.ndim == 1:
        dvl_flags = dvl_flags.reshape(-1, 1)

    # 세 데이터의 행 개수가 같아야 같은 기사에 대한 정보라고 볼 수 있습니다.
    row_counts = {
        "TF-IDF": X.shape[0],
        "labels": y.shape[0],
        "dvl_flags": dvl_flags.shape[0],
    }

    # 행 개수가 다르면 데이터가 서로 어긋난 것이므로 학습을 중단합니다.
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Input row counts do not match: {row_counts}")

    # 빠른 테스트용 옵션입니다.
    # 예: MAX_VECTOR_ROWS=2000으로 실행하면 전체 데이터 대신 2,000행만 샘플링합니다.
    max_rows = get_env_int("MAX_VECTOR_ROWS")
    if max_rows and max_rows < X.shape[0]:
        X, _, y, _, dvl_flags, _ = train_test_split(
            X,
            y,
            dvl_flags,
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    return X, y, dvl_flags

# 현재 하이퍼파라미터 조합으로 LogisticRegression 모델을 생성합니다.
def make_model(params):
    
    return LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver=params["solver"],
        class_weight=params["class_weight"],
        tol=params["tol"],
        # l1_ratio는 elasticnet에서만 의미가 있으므로 현재 l1 탐색에서는 None입니다.
        l1_ratio=params["l1_ratio"] if params["penalty"] == "elasticnet" else None,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        # liblinear에서는 병렬 처리가 크게 의미 없으므로 1로 고정합니다.
        n_jobs=1,
    )


def get_problem_type_names(dvl_flags):
    """dvl_flags 열 개수에 맞는 문제 유형 이름 목록을 만듭니다."""
    # 준비된 이름 중 실제 열 개수만큼만 먼저 사용합니다.
    names = PROBLEM_TYPE_COLUMNS[:dvl_flags.shape[1]]

    # 데이터의 열이 더 많으면 problem_type_6 같은 자동 이름을 붙입니다.
    if len(names) < dvl_flags.shape[1]:
        names.extend(
            f"problem_type_{index + 1}"
            for index in range(len(names), dvl_flags.shape[1])
        )

    return names

# 문제 유형별로 모델이 fake라고 예측한 비율을 계산합니다.
def score_problem_types(y_true, y_pred, dvl_flags):
    
    scores = {}
    problem_type_names = get_problem_type_names(dvl_flags)

    for index, problem_type in enumerate(problem_type_names):
        mask = dvl_flags[:, index].astype(int) == 1

        sample_count = int(mask.sum())

        if sample_count == 0:
            scores[problem_type] = {
                "sample_count": 0,
                "fake_recall": None,
            }
            continue

        type_y_pred = y_pred[mask]

        fake_recall = float(np.mean(type_y_pred == 1))

        scores[problem_type] = {
            "sample_count": sample_count,
            "fake_recall": fake_recall,
        }

    return scores

# 모델 하나를 학습하고 성능 지표를 계산합니다.
def fit_and_score_model(params, X_train, X_test, y_train, y_test, dvl_test):
    
    start_time = time.time()

    model = make_model(params)

    # 수렴 경고를 잡아서 로그에 남길 수 있게 합니다.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)

    # 발생한 warning 중 ConvergenceWarning만 따로 모읍니다.
    convergence_warnings = [
        warning for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    roc_auc = None
    if hasattr(model, "decision_function"):
        try:
            roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
        except ValueError:
            roc_auc = None

    # main 루프에서 로그 출력, best model 저장, best config 저장에 사용할 결과 묶음입니다.
    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "problem_type_scores": score_problem_types(y_test, y_pred, dvl_test),
        "convergence_warning_count": len(convergence_warnings),
        "fit_seconds": time.time() - start_time,
    }

# 현재까지 가장 좋은 모델의 설정과 성능을 txt 파일로 저장합니다.
def write_best_config(best_result):
    
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        config_file.write("Best model configuration - Logistic V4.2 pre-vectorized\n")
        config_file.write("=" * 60 + "\n")

        # 핵심 성능 지표를 저장합니다.
        config_file.write(f"Macro F1: {best_result['f1']:.4f}\n")
        config_file.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        config_file.write(f"Precision: {best_result['precision']:.4f}\n")
        config_file.write(f"Recall: {best_result['recall']:.4f}\n")

        # ROC-AUC는 계산 가능한 경우에만 저장합니다.
        if best_result["roc_auc"] is not None:
            config_file.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")

        # 오탐/미탐 상황을 볼 수 있도록 confusion matrix도 저장합니다.
        config_file.write(
            f"Confusion Matrix [tn, fp, fn, tp]: "
            f"[{best_result['tn']}, {best_result['fp']}, "
            f"{best_result['fn']}, {best_result['tp']}]\n"
        )

        # 가장 좋은 로지스틱 회귀 하이퍼파라미터 조합입니다.
        config_file.write(f"LogisticRegression: {format_params(best_result['lr_params'])}\n")

        # 문제 유형별 재현률(fake recall)도 함께 저장합니다.
        config_file.write("\nProblem type fake recall\n")
        for problem_type, score in best_result["problem_type_scores"].items():
            if score["fake_recall"] is None:
                config_file.write(f"- {problem_type}: no test samples\n")
            else:
                config_file.write(
                    f"- {problem_type}: {score['fake_recall']:.4f} "
                    f"({score['sample_count']} samples)\n"
                )

# 전체 학습 파이프라인을 실행하는 진입점입니다.
def main():
    
    # vector 데이터를 읽습니다. 실패하면 원인을 출력하고 종료합니다.
    try:
        X, y, dvl_flags = load_vector_data()
    except Exception as exc:
        print(f"Failed to load vector data: {exc}")
        sys.exit(1)

    # train/test split을 하려면 최소한의 행 수가 필요합니다.
    if X.shape[0] < 10:
        print("Not enough rows to split train/test data.")
        sys.exit(1)

    # 환경변수 TEST_SIZE가 있으면 그 값을 사용하고, 없으면 기본값 0.2를 사용합니다.
    test_size = get_env_float("TEST_SIZE", TEST_SIZE)

    # 학습 데이터와 테스트 데이터를 분리합니다.
    # stratify=y를 넣어 양쪽의 클래스 비율이 비슷하게 유지되도록 합니다.
    split_data = train_test_split(
        X,
        y,
        dvl_flags,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test, y_train, y_test, dvl_train, dvl_test = split_data

    # train 쪽 문제 유형 플래그는 현재 평가에 쓰지 않으므로 삭제합니다.
    del dvl_train

    # grid에서 유효한 로지스틱 회귀 조합을 모두 만듭니다.
    lr_configs = list(iter_param_grid(LR_GRID))
    total_runs = len(lr_configs)

    # 환경변수 MAX_TUNING_RUNS가 있으면 실행 횟수를 제한할 수 있습니다.
    max_runs = get_env_int("MAX_TUNING_RUNS", DEFAULT_MAX_TUNING_RUNS)
    max_runs = min(max_runs, total_runs)

    # 이번 실행의 기본 정보를 로그 파일 맨 위에 기록합니다.
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("Pre-vectorized TF-IDF + LogisticRegression tuning results - V4.2\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"TF-IDF file: {TFIDF_PATH}\n")
        log_file.write(f"Label file: {LABEL_PATH}\n")
        log_file.write(f"DVL flag file: {DVL_FLAG_PATH}\n")
        log_file.write(f"Matrix shape: {X.shape}\n")
        log_file.write(f"Train rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}\n")
        log_file.write(f"Test size: {test_size}\n")
        log_file.write(f"LR configs: {len(lr_configs)}\n")
        log_file.write(f"Planned runs: {max_runs} / {total_runs}\n")
        log_file.write(f"MAX_VECTOR_ROWS: {get_env_int('MAX_VECTOR_ROWS') or 'all'}\n")
        log_file.write("=" * 80 + "\n")

    # 콘솔과 로그 파일에 실행 시작 정보를 남깁니다.
    append_log(f"Loaded TF-IDF matrix: {X.shape}, nnz={X.nnz}")
    append_log(f"Training rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")
    append_log(f"Running {max_runs} / {total_runs} LogisticRegression combinations")
    append_log(f"Log file: {LOG_PATH}")

    # 현재까지 가장 좋은 결과를 저장할 변수입니다.
    best_result = None

    # 전체 튜닝 소요 시간을 재기 시작합니다.
    start_time = time.time()

    # 지정된 개수만큼 하이퍼파라미터 조합을 반복 실행합니다.
    for run_count, lr_params in enumerate(lr_configs[:max_runs], start=1):
        append_log("")
        append_log(f"[Run {run_count}/{max_runs}] LR: {format_params(lr_params)}")

        # 특정 조합이 실패해도 전체 튜닝이 중단되지 않도록 try/except로 감쌉니다.
        try:
            result = fit_and_score_model(
                lr_params,
                X_train,
                X_test,
                y_train,
                y_test,
                dvl_test,
            )
        except Exception as exc:
            append_log(f"  model failed: {exc}")
            continue

        # 이번 run의 주요 성능 지표를 로그에 남깁니다.
        append_log(
            f"  Accuracy={result['accuracy']:.4f} | Precision={result['precision']:.4f} "
            f"| Recall={result['recall']:.4f} | Macro-F1={result['f1']:.4f}"
        )

        # ROC-AUC가 계산되었으면 함께 남깁니다.
        if result["roc_auc"] is not None:
            append_log(f"  ROC-AUC={result['roc_auc']:.4f}")

        # confusion matrix를 로그에 남깁니다.
        append_log(
            f"  Confusion Matrix: tn={result['tn']}, fp={result['fp']}, "
            f"fn={result['fn']}, tp={result['tp']}"
        )

        # 이 조합의 학습 시간을 남깁니다.
        append_log(f"  Fit time: {result['fit_seconds']:.1f}s")

        # 수렴 경고가 있으면 개수를 표시합니다.
        if result["convergence_warning_count"]:
            append_log(
                f"  warning: convergence warning "
                f"({result['convergence_warning_count']} time(s))"
            )

        # 문제 유형별 fake recall을 로그에 남깁니다.
        if result["problem_type_scores"]:
            append_log("  Problem type fake recall:")
            for problem_type, score in result["problem_type_scores"].items():
                if score["fake_recall"] is None:
                    append_log(f"    - {problem_type}: no test samples")
                else:
                    append_log(
                        f"    - {problem_type}: {score['fake_recall']:.4f} "
                        f"({score['sample_count']} samples)"
                    )

        # Macro-F1이 가장 높은 모델을 best model로 판단합니다.
        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = {
                **result,
                "lr_params": lr_params.copy(),
            }

            # 가장 좋은 모델 객체를 pkl로 저장합니다.
            joblib.dump(result["model"], BEST_MODEL_PATH)

            # 가장 좋은 설정과 점수도 txt로 저장합니다.
            write_best_config(best_result)

            append_log("  New best model saved.")

    # 전체 튜닝 소요 시간을 계산합니다.
    elapsed = time.time() - start_time
    append_log("")
    append_log("=" * 80)
    append_log(f"Finished in {elapsed / 60:.1f} minutes")

    # 성공한 모델이 하나도 없으면 실패로 종료합니다.
    if best_result is None:
        append_log("No successful model run.")
        sys.exit(1)

    # 최종 best model 요약을 로그에 남깁니다.
    append_log("Best combination")
    append_log(f"  Macro-F1: {best_result['f1']:.4f}")
    append_log(f"  Accuracy: {best_result['accuracy']:.4f}")
    append_log(f"  Precision: {best_result['precision']:.4f}")
    append_log(f"  Recall: {best_result['recall']:.4f}")
    if best_result["roc_auc"] is not None:
        append_log(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
    append_log(f"  LogisticRegression: {format_params(best_result['lr_params'])}")
    append_log(f"Saved model: {BEST_MODEL_PATH}")
    append_log(f"Saved best config: {BEST_CONFIG_PATH}")
    append_log("=" * 80)


# 이 파일을 직접 실행했을 때만 학습이 시작됩니다.
# 다른 파일에서 import할 때 자동으로 main()이 돌지 않게 하는 표준 패턴입니다.
if __name__ == "__main__":
    main()
