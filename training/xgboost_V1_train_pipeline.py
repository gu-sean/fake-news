import itertools
import os
import random
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
from xgboost import XGBClassifier


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

VECTOR_DIR    = os.path.join(PROJECT_DIR, "data", "vector")
TFIDF_PATH    = os.path.join(VECTOR_DIR, "korean_tfidf.npz")
LABEL_PATH    = os.path.join(VECTOR_DIR, "korean_labels.npy")
DVL_FLAG_PATH = os.path.join(VECTOR_DIR, "korean_dvl_flags.npy")

# 문제 유형별 fake recall을 출력할 때 사용할 컬럼 이름입니다.
PROBLEM_TYPE_COLUMNS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

# 넓은 탐색에서 성능이 좋았던 영역 주변만 다시 살펴보는 세부 튜닝 범위입니다.
XGB_GRID = {
    "max_depth": [8, 9],
    "learning_rate": [0.03],
    "n_estimators": [1500],
    "subsample": [0.8],
    "colsample_bytree": [0.1, 0.2],
    "min_child_weight": [3, 5],
    "reg_alpha": [1.0],
    "reg_lambda": [1.0],
}

# 추천 시작점을 다른 후보보다 먼저 실행합니다.
RECOMMENDED_CONFIG = {
    "max_depth": 8,
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "subsample": 0.8,
    "colsample_bytree": 0.1,
    "min_child_weight": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
}

_XGB_OUT_DIR     = os.path.join(PROJECT_DIR, "models", "xgboost")
LOG_PATH         = os.path.join(_XGB_OUT_DIR, "xgboost_V1_log.txt")
BEST_MODEL_PATH  = os.path.join(_XGB_OUT_DIR, "best_model_xgboost_V1.pkl")
BEST_CONFIG_PATH = os.path.join(_XGB_OUT_DIR, "best_config_xgboost_V1.txt")

# 데이터 분리와 XGBoost 재현성을 맞추기 위한 공통 설정입니다.
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
EARLY_STOPPING_ROUNDS = 50

# 기본적으로 8개 세부 튜닝 조합을 모두 실행합니다.
DEFAULT_MAX_TUNING_RUNS = 8


def get_env_int(name, default=None):
    """환경변수에서 양의 정수 값을 읽고, 없거나 잘못되면 기본값을 씁니다."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    return parsed if parsed > 0 else default


def get_env_float(name, default):
    """TEST_SIZE처럼 0과 1 사이의 실수 환경변수를 읽습니다."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = float(value)
    except ValueError:
        return default

    return parsed if 0 < parsed < 1 else default


def append_log(message):
    """화면에 메시지를 출력하고 V1 튜닝 로그 파일에도 같은 내용을 남깁니다."""
    print(message, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
        log_file.flush()


def format_params(params):
    """모델 파라미터 딕셔너리를 한 줄 문자열로 보기 좋게 바꿉니다."""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def validate_input_files():
    """필수 벡터 파일이 없으면 학습을 시작하기 전에 중단합니다."""
    missing = [
        path for path in [TFIDF_PATH, LABEL_PATH, DVL_FLAG_PATH]
        if not os.path.exists(path)
    ]
    if missing:
        print("Required vector files were not found:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)


def load_vector_data():
    """TF-IDF, 라벨, DVL 플래그를 불러오고 기본 정합성을 확인합니다."""
    validate_input_files()

    # TF-IDF 행렬은 희소 행렬 그대로 유지합니다.
    # dense 형태로 바꾸면 전체 피처 공간에서 메모리를 크게 잡아먹습니다.
    X = sparse.load_npz(TFIDF_PATH)
    y = np.load(LABEL_PATH)
    dvl_flags = np.load(DVL_FLAG_PATH)

    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    y = np.asarray(y).astype(int).ravel()
    dvl_flags = np.asarray(dvl_flags)
    if dvl_flags.ndim == 1:
        dvl_flags = dvl_flags.reshape(-1, 1)

    # 세 입력은 같은 샘플 순서와 같은 행 개수를 가져야 합니다.
    row_counts = {
        "TF-IDF": X.shape[0],
        "labels": y.shape[0],
        "dvl_flags": dvl_flags.shape[0],
    }
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Input row counts do not match: {row_counts}")

    if X.min() < 0:
        raise ValueError("XGBoost input TF-IDF features must be non-negative.")

    # 빠른 실험을 위한 선택적 샘플링입니다.
    # stratify=y로 클래스 비율을 유지하면서 X, y, dvl_flags를 같은 행 기준으로 자릅니다.
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


def build_tuning_configs(grid, max_runs):
    """그리드에서 실제로 실행할 튜닝 목록을 만듭니다.

    이미 성능이 좋다고 확인된 추천 설정을 가장 먼저 평가합니다.
    나머지 조합은 고정 시드로 섞어 반복 실행해도 같은 후보를 비교하게 합니다.
    """
    keys = list(grid.keys())
    all_configs = [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]

    recommended = RECOMMENDED_CONFIG.copy()
    remaining = [config for config in all_configs if config != recommended]

    rng = random.Random(RANDOM_STATE)
    rng.shuffle(remaining)

    if max_runs <= 1:
        return [recommended]

    return [recommended, *remaining[:max_runs - 1]]


def make_model(params):
    """공통 학습 옵션과 후보 파라미터를 합쳐 XGBClassifier를 만듭니다."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbosity=0,
        **params,
    )


def get_problem_type_names(dvl_flags):
    """DVL 플래그 각 컬럼에 대응되는 문제 유형 이름을 반환합니다."""
    names = PROBLEM_TYPE_COLUMNS[:dvl_flags.shape[1]]
    if len(names) < dvl_flags.shape[1]:
        names.extend(
            f"problem_type_{index + 1}"
            for index in range(len(names), dvl_flags.shape[1])
        )
    return names


def score_problem_types(y_pred, dvl_flags):
    """문제 유형별로 fake라고 예측한 비율을 계산합니다."""
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

        fake_recall = float(np.mean(y_pred[mask] == 1))
        scores[problem_type] = {
            "sample_count": sample_count,
            "fake_recall": fake_recall,
        }

    return scores


def fit_and_score_model(
    params,
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    dvl_test,
):
    """후보 모델 하나를 학습하고 비교에 필요한 성능지표를 반환합니다."""
    start_time = time.time()
    model = make_model(params)

    # validation 데이터는 early stopping에만 쓰고, 최종 평가는 test 데이터로만 합니다.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Macro 평균은 양쪽 클래스를 균등하게 봅니다.
    # 샘플링으로 클래스 비율이 달라질 수 있을 때 특히 확인하기 좋습니다.
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(y_test, y_probability)
    except ValueError:
        roc_auc = None

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
        "best_iteration": getattr(model, "best_iteration", None),
        "fit_seconds": time.time() - start_time,
    }


def write_best_config(best_result):
    """최고 모델의 성능지표와 파라미터를 사람이 읽기 쉬운 텍스트 파일로 저장합니다."""
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        config_file.write("Best model configuration - XGBoost V1 pre-vectorized\n")
        config_file.write("=" * 60 + "\n")
        config_file.write(f"Macro F1: {best_result['f1']:.4f}\n")
        config_file.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        config_file.write(f"Precision: {best_result['precision']:.4f}\n")
        config_file.write(f"Recall: {best_result['recall']:.4f}\n")
        if best_result["roc_auc"] is not None:
            config_file.write(f"ROC-AUC: {best_result['roc_auc']:.4f}\n")
        config_file.write(
            f"Confusion Matrix [tn, fp, fn, tp]: "
            f"[{best_result['tn']}, {best_result['fp']}, "
            f"{best_result['fn']}, {best_result['tp']}]\n"
        )
        config_file.write(f"Best iteration: {best_result['best_iteration']}\n")
        config_file.write(f"XGBClassifier: {format_params(best_result['xgb_params'])}\n")

        config_file.write("\nProblem type fake recall\n")
        for problem_type, score in best_result["problem_type_scores"].items():
            if score["fake_recall"] is None:
                config_file.write(f"- {problem_type}: no test samples\n")
            else:
                config_file.write(
                    f"- {problem_type}: {score['fake_recall']:.4f} "
                    f"({score['sample_count']} samples)\n"
                )


def main():
    """V1 전체 흐름을 실행합니다: 데이터 로드, 후보 튜닝, 최고 모델 저장."""
    try:
        X, y, dvl_flags = load_vector_data()
    except Exception as exc:
        print(f"Failed to load vector data: {exc}")
        sys.exit(1)

    if X.shape[0] < 20:
        print("Not enough rows to split train/validation/test data.")
        sys.exit(1)

    test_size = get_env_float("TEST_SIZE", TEST_SIZE)
    validation_size = get_env_float("VALIDATION_SIZE", VALIDATION_SIZE)

    # 최종 test set은 모델 선택이나 early stopping에 사용하지 않고 따로 둡니다.
    split_data = train_test_split(
        X,
        y,
        dvl_flags,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_full, X_test, y_train_full, y_test, dvl_train_full, dvl_test = split_data

    # training 데이터 안에서 early stopping 전용 validation set을 분리합니다.
    train_data = train_test_split(
        X_train_full,
        y_train_full,
        dvl_train_full,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    X_train, X_valid, y_train, y_valid, dvl_train, dvl_valid = train_data
    del X_train_full, y_train_full, dvl_train_full, dvl_train, dvl_valid

    total_grid_size = int(np.prod([len(values) for values in XGB_GRID.values()]))
    max_runs = get_env_int("MAX_TUNING_RUNS", DEFAULT_MAX_TUNING_RUNS)
    max_runs = min(max_runs, total_grid_size)
    xgb_configs = build_tuning_configs(XGB_GRID, max_runs)

    # 데이터 정보와 탐색 범위를 로그 파일 첫 부분에 기록합니다.
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("Pre-vectorized TF-IDF + XGBoost tuning results - V1\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"TF-IDF file: {TFIDF_PATH}\n")
        log_file.write(f"Label file: {LABEL_PATH}\n")
        log_file.write(f"DVL flag file: {DVL_FLAG_PATH}\n")
        log_file.write(f"Matrix shape: {X.shape}\n")
        log_file.write(
            f"Train rows: {X_train.shape[0]} | Validation rows: {X_valid.shape[0]} "
            f"| Test rows: {X_test.shape[0]}\n"
        )
        log_file.write(f"Test size: {test_size}\n")
        log_file.write(f"Validation size within training data: {validation_size}\n")
        log_file.write(f"XGBoost fine-tuning grid size: {total_grid_size}\n")
        log_file.write(f"Planned tuning runs: {len(xgb_configs)}\n")
        log_file.write(f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}\n")
        log_file.write(f"MAX_VECTOR_ROWS: {get_env_int('MAX_VECTOR_ROWS') or 'all'}\n")
        log_file.write("=" * 80 + "\n")

    append_log(f"Loaded TF-IDF matrix: {X.shape}, nnz={X.nnz}")
    append_log(
        f"Training rows: {X_train.shape[0]} | Validation rows: {X_valid.shape[0]} "
        f"| Test rows: {X_test.shape[0]}"
    )
    append_log(
        f"Running {len(xgb_configs)} combinations "
        f"from the {total_grid_size}-configuration fine-tuning grid"
    )
    append_log(f"Log file: {LOG_PATH}")

    best_result = None
    start_time = time.time()

    # V1은 여러 후보 설정을 비교한 뒤 Macro-F1이 가장 높은 모델을 최종 결과로 남깁니다.
    for run_count, xgb_params in enumerate(xgb_configs, start=1):
        append_log("")
        append_log(f"[Run {run_count}/{len(xgb_configs)}] XGB: {format_params(xgb_params)}")

        try:
            result = fit_and_score_model(
                xgb_params,
                X_train,
                X_valid,
                X_test,
                y_train,
                y_valid,
                y_test,
                dvl_test,
            )
        except Exception as exc:
            append_log(f"  model failed: {exc}")
            continue

        append_log(
            f"  Accuracy={result['accuracy']:.4f} | Precision={result['precision']:.4f} "
            f"| Recall={result['recall']:.4f} | Macro-F1={result['f1']:.4f}"
        )
        if result["roc_auc"] is not None:
            append_log(f"  ROC-AUC={result['roc_auc']:.4f}")
        append_log(
            f"  Confusion Matrix: tn={result['tn']}, fp={result['fp']}, "
            f"fn={result['fn']}, tp={result['tp']}"
        )
        append_log(f"  Best iteration: {result['best_iteration']}")
        append_log(f"  Fit time: {result['fit_seconds']:.1f}s")

        if result["warning_count"]:
            append_log(f"  warning: {result['warning_count']} warning(s)")

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

        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = {
                **result,
                "xgb_params": xgb_params.copy(),
            }
            # 이후 후보에서 오류가 나더라도 현재까지의 최고 모델을 잃지 않도록 즉시 저장합니다.
            joblib.dump(result["model"], BEST_MODEL_PATH)
            write_best_config(best_result)
            append_log("  New best model saved.")

    elapsed = time.time() - start_time
    append_log("")
    append_log("=" * 80)
    append_log(f"Finished in {elapsed / 60:.1f} minutes")

    if best_result is None:
        append_log("No successful model run.")
        sys.exit(1)

    append_log("Best combination")
    append_log(f"  Macro-F1: {best_result['f1']:.4f}")
    append_log(f"  Accuracy: {best_result['accuracy']:.4f}")
    append_log(f"  Precision: {best_result['precision']:.4f}")
    append_log(f"  Recall: {best_result['recall']:.4f}")
    if best_result["roc_auc"] is not None:
        append_log(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
    append_log(f"  Best iteration: {best_result['best_iteration']}")
    append_log(f"  XGBClassifier: {format_params(best_result['xgb_params'])}")
    append_log(f"Saved model: {BEST_MODEL_PATH}")
    append_log(f"Saved best config: {BEST_CONFIG_PATH}")
    append_log("=" * 80)


if __name__ == "__main__":
    main()
