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
from xgboost import XGBClassifier


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

VECTOR_DIR    = os.path.join(PROJECT_DIR, "data", "vector")
TFIDF_PATH    = os.path.join(VECTOR_DIR, "korean_tfidf.npz")
LABEL_PATH    = os.path.join(VECTOR_DIR, "korean_labels.npy")
DVL_FLAG_PATH = os.path.join(VECTOR_DIR, "korean_dvl_flags.npy")

_XGB_OUT_DIR = os.path.join(PROJECT_DIR, "models", "xgboost")
MODEL_PATH   = os.path.join(_XGB_OUT_DIR, "best_model_xgboost_V2.pkl")
METRICS_PATH = os.path.join(_XGB_OUT_DIR, "performance_xgboost_V2.txt")

# 문제 유형별 fake recall을 출력할 때 사용할 컬럼 이름입니다.
PROBLEM_TYPE_COLUMNS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

# 설정은 리스트 안에 하나만 둡니다.
# 이 파이프라인은 여러 조합을 반복 탐색하지 않고 이 설정 하나만 바로 학습합니다.
TUNING_CONFIGS = [
    {
        "max_depth": 9,
        "learning_rate": 0.03,
        "n_estimators": 1500,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "min_child_weight": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    }
]

# 데이터 분리와 XGBoost 재현성을 맞추기 위한 공통 설정입니다.
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
EARLY_STOPPING_ROUNDS = 50


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


def format_params(params):
    """모델 파라미터 딕셔너리를 한 줄 문자열로 보기 좋게 바꿉니다."""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def validate_input_files():
    """필수 벡터 파일이 없으면 학습을 시작하기 전에 중단합니다."""
    missing = [
        path
        for path in [TFIDF_PATH, LABEL_PATH, DVL_FLAG_PATH]
        if not os.path.exists(path)
    ]
    if missing:
        print("Required vector files were not found:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)


def load_vector_data():
    """TF-IDF, 라벨, DVL 플래그를 불러오고 정합성을 확인합니다."""
    validate_input_files()

    # TF-IDF 행렬은 희소 행렬 그대로 유지해 메모리 사용량을 줄입니다.
    X = sparse.load_npz(TFIDF_PATH)
    y = np.load(LABEL_PATH)
    dvl_flags = np.load(DVL_FLAG_PATH)

    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    y = np.asarray(y).astype(int).ravel()
    dvl_flags = np.asarray(dvl_flags)
    if dvl_flags.ndim == 1:
        dvl_flags = dvl_flags.reshape(-1, 1)

    # 모든 배열은 같은 샘플을 같은 순서로 담고 있어야 합니다.
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
    # 클래스 비율을 유지하면서 X, y, dvl_flags에서 같은 행을 뽑습니다.
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


def make_model(params):
    """V2에서 한 번 학습할 XGBClassifier를 만듭니다."""
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

    for index, problem_type in enumerate(get_problem_type_names(dvl_flags)):
        mask = dvl_flags[:, index].astype(int) == 1
        sample_count = int(mask.sum())
        scores[problem_type] = {
            "sample_count": sample_count,
            "fake_recall": float(np.mean(y_pred[mask] == 1))
            if sample_count
            else None,
        }

    return scores


def train_and_evaluate(
    params,
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    dvl_test,
):
    """V2의 고정 설정을 한 번 학습하고 최종 test 성능지표를 모읍니다."""
    start_time = time.time()
    model = make_model(params)

    # validation 데이터는 early stopping에만 쓰고,
    # test 데이터는 최종 평가를 위해 건드리지 않습니다.
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

    # Macro 평균은 real/fake 클래스를 균등하게 반영합니다.
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
        "params": params.copy(),
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

# 학습 후 화면에 출력하고 파일로 저장할 성능 리포트 문자열을 만듭니다.
def build_metrics_text(result, matrix_shape, split_shapes):
    
    X_train_shape, X_valid_shape, X_test_shape = split_shapes
    lines = [
        "XGBoost V2 single-configuration training results",
        "=" * 60,
        f"Matrix shape: {matrix_shape}",
        (
            f"Train rows: {X_train_shape[0]} | Validation rows: {X_valid_shape[0]} "
            f"| Test rows: {X_test_shape[0]}"
        ),
        f"XGBClassifier: {format_params(result['params'])}",
        "",
        f"Macro F1: {result['f1']:.4f}",
        f"Accuracy: {result['accuracy']:.4f}",
        f"Precision: {result['precision']:.4f}",
        f"Recall: {result['recall']:.4f}",
    ]

    if result["roc_auc"] is not None:
        lines.append(f"ROC-AUC: {result['roc_auc']:.4f}")

    lines.extend(
        [
            (
                "Confusion Matrix [tn, fp, fn, tp]: "
                f"[{result['tn']}, {result['fp']}, {result['fn']}, {result['tp']}]"
            ),
            f"Best iteration: {result['best_iteration']}",
            f"Fit time: {result['fit_seconds']:.1f}s",
            f"Warnings: {result['warning_count']}",
            "",
            "Problem type fake recall",
        ]
    )

    for problem_type, score in result["problem_type_scores"].items():
        if score["fake_recall"] is None:
            lines.append(f"- {problem_type}: no test samples")
        else:
            lines.append(
                f"- {problem_type}: {score['fake_recall']:.4f} "
                f"({score['sample_count']} samples)"
            )

    return "\n".join(lines) + "\n"

# V2 전체 흐름을 실행합니다: 데이터 로드, 고정 설정 1회 학습, 결과 저장.
def main():
    
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

    # 먼저 train/test로 나누고, train 안에서 early stopping용 validation을 다시 나눕니다.
    X_train_full, X_test, y_train_full, y_test, dvl_train_full, dvl_test = (
        train_test_split(
            X,
            y,
            dvl_flags,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )
    X_train, X_valid, y_train, y_valid, _, _ = train_test_split(
        X_train_full,
        y_train_full,
        dvl_train_full,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    del X_train_full, y_train_full, dvl_train_full

    # V2는 의도적으로 첫 번째 설정 하나만 사용합니다.
    # 그리드 반복문이나 탐색 로그는 만들지 않습니다.
    params = TUNING_CONFIGS[0]
    print(f"Training XGBoost V2 with: {format_params(params)}", flush=True)
    print(
        f"Rows - train: {X_train.shape[0]}, validation: {X_valid.shape[0]}, "
        f"test: {X_test.shape[0]}",
        flush=True,
    )

    try:
        result = train_and_evaluate(
            params,
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            dvl_test,
        )
    except Exception as exc:
        print(f"Model training failed: {exc}")
        sys.exit(1)

    # 사람이 읽는 리포트를 저장하기 전에 학습된 모델부터 pkl로 저장합니다.
    joblib.dump(result["model"], MODEL_PATH)
    # xgb_model_ko.pkl도 동일하게 갱신 (detector.py / eval_all_models.py가 이 경로를 사용)
    xgb_model_ko_path = os.path.join(_XGB_OUT_DIR, "xgb_model_ko.pkl")
    joblib.dump(result["model"], xgb_model_ko_path)

    metrics_text = build_metrics_text(
        result,
        X.shape,
        (X_train.shape, X_valid.shape, X_test.shape),
    )
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(metrics_text)

    print()
    print(metrics_text, end="")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved performance metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
