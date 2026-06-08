import itertools
import os
import sys
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(SCRIPT_DIR, "vector")
TFIDF_PATH = os.path.join(VECTOR_DIR, "korean_tfidf.npz")

# 각 기사의 정답 라벨입니다.
LABEL_PATH = os.path.join(VECTOR_DIR, "korean_labels.npy")

# 가짜 뉴스 유형별 표시값입니다.
DVL_FLAG_PATH = os.path.join(VECTOR_DIR, "korean_dvl_flags.npy")

PROBLEM_TYPE_COLUMNS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

# Naive Bayes에서 반복 실험할 하이퍼파라미터 후보들입니다.
NB_GRID = {
    "model_type": ["multinomial", "complement", "bernoulli"],
    "alpha": [
        0.001,
        0.002,
        0.003,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.65,
        0.75,
        0.9,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ],
    "fit_prior": [True, False],
    "class_prior": [
        None,
        [0.5, 0.5],
        [0.48, 0.52],
        [0.45, 0.55],
        [0.42, 0.58],
        [0.4, 0.6],
        [0.35, 0.65],
        [0.3, 0.7],
    ],
    "norm": [True, False],
    "binarize": [0.0, 0.001, 0.003, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05],
}

LOG_PATH = os.path.join(SCRIPT_DIR, "naive_V1_log.txt")

BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model_naive_V1.pkl")

BEST_CONFIG_PATH = os.path.join(SCRIPT_DIR, "best_config_naive_V1.txt")

RANDOM_STATE = 42

# 전체 데이터 중 20%를 test set으로 사용합니다.
TEST_SIZE = 0.2

# 기본 튜닝 실행 횟수입니다.
DEFAULT_MAX_TUNING_RUNS = 2000

# NB_GRID의 후보를 실제 실행 가능한 조합으로 만들어 하나씩 반환합니다.
def iter_param_grid(grid):
    
    keys = list(grid.keys())

    grouped_configs = {model_type: [] for model_type in grid["model_type"]}

    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))

        if is_valid_nb_config(params):
            grouped_configs[params["model_type"]].append(params)

    max_group_size = max(len(configs) for configs in grouped_configs.values())

    for index in range(max_group_size):
        for model_type in grid["model_type"]:
            configs = grouped_configs[model_type]
            if index < len(configs):
                yield configs[index]

# Naive Bayes 모델별로 사용할 수 없는 옵션 조합을 걸러냅니다.
def is_valid_nb_config(params):

    model_type = params["model_type"]

    if params["class_prior"] is not None and not params["fit_prior"]:
        return False
    if model_type == "multinomial":
        return params["norm"] is True and params["binarize"] == 0.0
    if model_type == "complement":
        return params["binarize"] == 0.0
    if model_type == "bernoulli":
        return params["norm"] is True
    return False

# 환경변수에서 양의 정수를 읽습니다. 없거나 잘못되면 기본값을 돌려줍니다.
def get_env_int(name, default=None):
    
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = int(value)
    except ValueError:
        # 숫자로 바꿀 수 없는 값이면 안전하게 기본값을 사용합니다.
        return default

    # 0 이하의 값은 실행 제한값으로 쓰기 애매하므로 기본값을 사용합니다.
    return parsed if parsed > 0 else default

# 환경변수에서 0과 1 사이의 실수를 읽습니다. 없거나 잘못되면 기본값을 사용합니다.
def get_env_float(name, default):
    
    # 예: TEST_SIZE=0.3처럼 test set 비율을 바꿀 수 있습니다.
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = float(value)
    except ValueError:
        return default

    # test_size는 0보다 크고 1보다 작아야 합니다.
    return parsed if 0 < parsed < 1 else default

# 메시지를 콘솔과 로그 파일 양쪽에 동시에 남깁니다.
def append_log(message):
    
    # flush=True는 장시간 실행 중에도 출력이 바로 보이도록 합니다.
    print(message, flush=True)

    # 로그 파일을 append 모드로 열어 기존 로그 뒤에 한 줄씩 추가합니다.
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
        log_file.flush()

# 파라미터 dict를 로그에 쓰기 좋은 한 줄 문자열로 바꿉니다.
def format_params(params):
    
    formatted = []

    for key, value in params.items():
        if value is None:
            display_value = "default"
        elif isinstance(value, list):
            display_value = "/".join(str(item) for item in value)
        else:
            display_value = value

        formatted.append(f"{key}={display_value}")

    return ", ".join(formatted)

# 학습에 필요한 세 개의 vector 파일이 실제로 있는지 확인합니다.
def validate_input_files():
    
    missing = [
        path for path in [TFIDF_PATH, LABEL_PATH, DVL_FLAG_PATH]
        if not os.path.exists(path)
    ]

    # 하나라도 없으면 뒤에서 더 어려운 오류가 나기 전에 바로 종료합니다.
    if missing:
        print("Required vector files were not found:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

# TF-IDF 행렬, 라벨, 문제 유형 플래그를 메모리로 읽고 기본 검사를 수행합니다.
def load_vector_data():
    
    # 먼저 파일 존재 여부부터 확인합니다.
    validate_input_files()

    # .npz로 저장된 scipy sparse matrix를 읽습니다.
    # TF-IDF는 대부분 0이기 때문에 sparse 형식이 메모리 절약에 중요합니다.
    X = sparse.load_npz(TFIDF_PATH)

    # 라벨과 문제 유형 플래그는 numpy 배열로 저장되어 있습니다.
    y = np.load(LABEL_PATH)
    dvl_flags = np.load(DVL_FLAG_PATH)

    # scikit-learn 모델에 안정적으로 넣기 위해 CSR sparse matrix로 맞춥니다.
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    y = np.asarray(y).astype(int).ravel()

    dvl_flags = np.asarray(dvl_flags)

    if dvl_flags.ndim == 1:
        dvl_flags = dvl_flags.reshape(-1, 1)

    row_counts = {
        "TF-IDF": X.shape[0],
        "labels": y.shape[0],
        "dvl_flags": dvl_flags.shape[0],
    }

    # 값에 이상이 있을 경우 강제로 종료합니다.
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Input row counts do not match: {row_counts}")

    if X.min() < 0:
        raise ValueError("Naive Bayes requires non-negative TF-IDF features.")

    # 빠른 테스트가 필요할 때 MAX_VECTOR_ROWS 환경변수로 일부 행만 샘플링할 수 있습니다.
    max_rows = get_env_int("MAX_VECTOR_ROWS")
    if max_rows and max_rows < X.shape[0]:
        # stratify=y를 넣어 라벨 비율이 크게 깨지지 않게 일부 데이터만 뽑습니다.
        X, _, y, _, dvl_flags, _ = train_test_split(
            X,
            y,
            dvl_flags,
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    return X, y, dvl_flags

# 파라미터 dict를 받아 실제 scikit-learn Naive Bayes 모델 객체를 만듭니다.
def make_model(params):
    
    # 세 Naive Bayes 모델이 공통으로 받는 옵션입니다.
    common = {
        "alpha": params["alpha"],
        "fit_prior": params["fit_prior"],
        "class_prior": params["class_prior"],
    }

    # MultinomialNB: TF-IDF 텍스트 분류의 기본 기준 모델입니다.
    if params["model_type"] == "multinomial":
        return MultinomialNB(**common)

    # ComplementNB: 클래스 불균형이 있는 텍스트 분류에서 자주 비교해볼 만한 모델입니다.
    if params["model_type"] == "complement":
        return ComplementNB(norm=params["norm"], **common)

    # BernoulliNB: TF-IDF feature를 임계값 기준으로 0/1처럼 보고 학습합니다.
    if params["model_type"] == "bernoulli":
        return BernoulliNB(binarize=params["binarize"], **common)

    # 여기까지 왔다면 NB_GRID에 알 수 없는 모델명이 들어간 것입니다.
    raise ValueError(f"Unknown model_type: {params['model_type']}")

# dvl_flags 열 개수에 맞춰 문제 유형 이름 리스트를 만듭니다.
def get_problem_type_names(dvl_flags):
    
    names = PROBLEM_TYPE_COLUMNS[:dvl_flags.shape[1]]

    if len(names) < dvl_flags.shape[1]:
        names.extend(
            f"problem_type_{index + 1}"
            for index in range(len(names), dvl_flags.shape[1])
        )

    return names

# 문제 유형별로 모델이 fake라고 잡아낸 비율을 계산합니다.
def score_problem_types(y_pred, dvl_flags):
    
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

        # 해당 유형 sample에 대한 예측값만 꺼냅니다.
        type_y_pred = y_pred[mask]

        # 문제 유형에 속한 sample 중 fake(1)로 예측된 비율입니다. = 재현율(Recall)
        fake_recall = float(np.mean(type_y_pred == 1))

        scores[problem_type] = {
            "sample_count": sample_count,
            "fake_recall": fake_recall,
        }

    return scores

# 모델 1개를 학습하고 모든 평가 지표를 계산합니다.
def fit_and_score_model(params, X_train, X_test, y_train, y_test, dvl_test):
    
    # 이 run의 학습/평가 시간을 재기 시작합니다.
    start_time = time.time()

    # 현재 하이퍼파라미터 조합에 맞는 모델 객체를 만듭니다.
    model = make_model(params)

    # 학습 중 발생하는 warning을 기록합니다.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)

    # test set에 대한 예측 라벨입니다.
    y_pred = model.predict(X_test)

    # 전체 정답률입니다.
    accuracy = accuracy_score(y_test, y_pred)

    # macro 평균은 클래스별 precision/recall/f1을 구한 뒤 단순 평균합니다.
    # 데이터 불균형이 있을 때 accuracy보다 모델 품질을 더 균형 있게 볼 수 있습니다.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # confusion matrix를 tn, fp, fn, tp 순서로 펼칩니다.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except ValueError:
            roc_auc = None

    # main 함수에서 로그와 best model 저장에 쓸 정보를 dict로 묶어 반환합니다.
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
        "problem_type_scores": score_problem_types(y_pred, dvl_test),
        "warning_count": len(caught_warnings),
        "fit_seconds": time.time() - start_time,
    }

# 현재까지 가장 좋은 모델의 설정과 점수를 txt 파일로 저장합니다.
def write_best_config(best_result):
    
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        config_file.write("Best model configuration - Naive Bayes V1 pre-vectorized\n")
        config_file.write("=" * 60 + "\n")

        # 핵심 평가 지표를 사람이 읽기 좋게 4자리까지 저장합니다.
        config_file.write(f"Macro F1: {best_result['f1']:.4f}\n")
        config_file.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        config_file.write(f"Precision: {best_result['precision']:.4f}\n")
        config_file.write(f"Recall: {best_result['recall']:.4f}\n")

        # ROC-AUC가 계산된 경우에만 저장합니다.
        if best_result["roc_auc"] is not None:
            config_file.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")

        # confusion matrix도 같이 저장해 오탐/미탐 균형을 확인할 수 있게 합니다.
        config_file.write(
            f"Confusion Matrix [tn, fp, fn, tp]: "
            f"[{best_result['tn']}, {best_result['fp']}, "
            f"{best_result['fn']}, {best_result['tp']}]\n"
        )

        # 가장 좋았던 Naive Bayes 하이퍼파라미터 조합입니다.
        config_file.write(f"NaiveBayes: {format_params(best_result['nb_params'])}\n")

        # 문제 유형별 fake recall도 저장합니다.
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
    
    # 데이터 로딩 중 문제가 생기면 메시지를 출력하고 종료합니다.
    try:
        X, y, dvl_flags = load_vector_data()
    except Exception as exc:
        print(f"Failed to load vector data: {exc}")
        sys.exit(1)

    # train/test로 나누려면 최소한의 행 수가 필요합니다.
    if X.shape[0] < 10:
        print("Not enough rows to split train/test data.")
        sys.exit(1)

    # 환경변수 TEST_SIZE가 있으면 그 값을 사용하고, 없으면 기본값 0.2를 씁니다.
    test_size = get_env_float("TEST_SIZE", TEST_SIZE)

    # 학습 데이터와 테스트 데이터를 나눕니다.
    # stratify=y를 사용해서 train/test 양쪽의 라벨 비율을 비슷하게 유지합니다.
    split_data = train_test_split(
        X,
        y,
        dvl_flags,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # train_test_split 결과를 각 변수에 나누어 담습니다.
    X_train, X_test, y_train, y_test, dvl_train, dvl_test = split_data

    # train 쪽 문제 유형 플래그는 현재 평가에 쓰지 않으므로 삭제합니다.
    del dvl_train

    # NB_GRID에서 실제 실행 가능한 후보 조합을 모두 만듭니다.
    nb_configs = list(iter_param_grid(NB_GRID))

    # 전체 후보 개수입니다.
    total_runs = len(nb_configs)

    # 환경변수 MAX_TUNING_RUNS가 있으면 실행 횟수를 바꿀 수 있습니다.
    max_runs = get_env_int("MAX_TUNING_RUNS", DEFAULT_MAX_TUNING_RUNS)

    # 사용자가 전체 후보보다 큰 값을 넣어도 실제 후보 개수를 넘지 않게 합니다.
    max_runs = min(max_runs, total_runs)

    # 로그 파일을 새로 만들고 이번 실행의 기본 정보를 맨 위에 기록합니다.
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("Pre-vectorized TF-IDF + Naive Bayes tuning results - V1\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"TF-IDF file: {TFIDF_PATH}\n")
        log_file.write(f"Label file: {LABEL_PATH}\n")
        log_file.write(f"DVL flag file: {DVL_FLAG_PATH}\n")
        log_file.write(f"Matrix shape: {X.shape}\n")
        log_file.write(f"Train rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}\n")
        log_file.write(f"Test size: {test_size}\n")
        log_file.write(f"NB configs: {len(nb_configs)}\n")
        log_file.write(f"Planned runs: {max_runs} / {total_runs}\n")
        log_file.write(f"MAX_VECTOR_ROWS: {get_env_int('MAX_VECTOR_ROWS') or 'all'}\n")
        log_file.write("=" * 80 + "\n")

    # 콘솔과 로그 파일에 실행 시작 정보를 남깁니다.
    append_log(f"Loaded TF-IDF matrix: {X.shape}, nnz={X.nnz}")
    append_log(f"Training rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")
    append_log(f"Running {max_runs} / {total_runs} Naive Bayes combinations")
    append_log(f"Log file: {LOG_PATH}")

    # 현재까지 가장 좋은 결과를 담을 변수입니다.
    best_result = None

    # 전체 튜닝 소요 시간을 재기 시작합니다.
    start_time = time.time()

    # 실행할 후보만큼 반복합니다.
    for run_count, nb_params in enumerate(nb_configs[:max_runs], start=1):
        append_log("")
        append_log(f"[Run {run_count}/{max_runs}] NB: {format_params(nb_params)}")

        # 특정 조합이 실패해도 전체 튜닝이 멈추지 않게 try/except로 감쌉니다.
        try:
            result = fit_and_score_model(
                nb_params,
                X_train,
                X_test,
                y_train,
                y_test,
                dvl_test,
            )
        except Exception as exc:
            append_log(f"  model failed: {exc}")
            continue

        # 이번 run의 주요 점수를 로그에 남깁니다.
        append_log(
            f"  Accuracy={result['accuracy']:.4f} | Precision={result['precision']:.4f} "
            f"| Recall={result['recall']:.4f} | Macro-F1={result['f1']:.4f}"
        )

        # ROC-AUC가 계산된 경우에만 출력합니다.
        if result["roc_auc"] is not None:
            append_log(f"  ROC-AUC={result['roc_auc']:.4f}")

        # confusion matrix도 로그에 남깁니다.
        append_log(
            f"  Confusion Matrix: tn={result['tn']}, fp={result['fp']}, "
            f"fn={result['fn']}, tp={result['tp']}"
        )

        # 학습과 평가에 걸린 시간입니다.
        append_log(f"  Fit time: {result['fit_seconds']:.1f}s")

        # warning이 있었다면 개수를 남깁니다.
        if result["warning_count"]:
            append_log(f"  warning: {result['warning_count']} warning(s)")

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
                "nb_params": nb_params.copy(),
            }

            # 현재까지 가장 좋은 모델 객체를 pkl로 저장합니다.
            joblib.dump(result["model"], BEST_MODEL_PATH)

            # 가장 좋은 모델의 설정과 점수도 txt로 저장합니다.
            write_best_config(best_result)

            append_log("  New best model saved.")

    # 전체 튜닝에 걸린 시간입니다.
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

    append_log(f"  NaiveBayes: {format_params(best_result['nb_params'])}")
    append_log(f"Saved model: {BEST_MODEL_PATH}")
    append_log(f"Saved best config: {BEST_CONFIG_PATH}")
    append_log("=" * 80)


# 이 파일을 직접 실행했을 때만 main()을 실행합니다.
# 다른 파일에서 import할 경우에는 자동 학습이 시작되지 않습니다.
if __name__ == "__main__":
    main()
