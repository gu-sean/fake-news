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
TEST_SIZE = 0.2
MAX_ITER = 3000
DEFAULT_MAX_TUNING_RUNS = 12


def iter_param_grid(grid):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        if is_valid_lr_config(params):
            yield params


def is_valid_lr_config(params):
    penalty = params["penalty"]
    solver = params["solver"]
    l1_ratio = params["l1_ratio"]

    if solver == "liblinear":
        return penalty in {"l1", "l2"} and l1_ratio is None
    if solver == "saga":
        if penalty in {"l1", "l2"}:
            return l1_ratio is None
        if penalty == "elasticnet":
            return l1_ratio is not None
    return False


def get_env_int(name, default=None):
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    return parsed if parsed > 0 else default


def get_env_float(name, default):
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = float(value)
    except ValueError:
        return default

    return parsed if 0 < parsed < 1 else default


def append_log(message):
    print(message, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
        log_file.flush()


def format_params(params):
    formatted = []
    for key, value in params.items():
        formatted.append(f"{key}={value if value is not None else 'default'}")
    return ", ".join(formatted)


def validate_input_files():
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
    validate_input_files()

    X = sparse.load_npz(TFIDF_PATH)
    y = np.load(LABEL_PATH)
    dvl_flags = np.load(DVL_FLAG_PATH)

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
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Input row counts do not match: {row_counts}")

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
    return LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver=params["solver"],
        class_weight=params["class_weight"],
        tol=params["tol"],
        l1_ratio=params["l1_ratio"] if params["penalty"] == "elasticnet" else None,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def get_problem_type_names(dvl_flags):
    names = PROBLEM_TYPE_COLUMNS[:dvl_flags.shape[1]]
    if len(names) < dvl_flags.shape[1]:
        names.extend(
            f"problem_type_{index + 1}"
            for index in range(len(names), dvl_flags.shape[1])
        )
    return names


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


def fit_and_score_model(params, X_train, X_test, y_train, y_test, dvl_test):
    start_time = time.time()

    model = make_model(params)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)

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


def write_best_config(best_result):
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        config_file.write("Best model configuration - Logistic V4.2 pre-vectorized\n")
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
        config_file.write(f"LogisticRegression: {format_params(best_result['lr_params'])}\n")
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
    try:
        X, y, dvl_flags = load_vector_data()
    except Exception as exc:
        print(f"Failed to load vector data: {exc}")
        sys.exit(1)

    if X.shape[0] < 10:
        print("Not enough rows to split train/test data.")
        sys.exit(1)

    test_size = get_env_float("TEST_SIZE", TEST_SIZE)
    split_data = train_test_split(
        X,
        y,
        dvl_flags,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test, y_train, y_test, dvl_train, dvl_test = split_data
    del dvl_train

    lr_configs = list(iter_param_grid(LR_GRID))
    total_runs = len(lr_configs)
    max_runs = get_env_int("MAX_TUNING_RUNS", DEFAULT_MAX_TUNING_RUNS)
    max_runs = min(max_runs, total_runs)

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

    append_log(f"Loaded TF-IDF matrix: {X.shape}, nnz={X.nnz}")
    append_log(f"Training rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")
    append_log(f"Running {max_runs} / {total_runs} LogisticRegression combinations")
    append_log(f"Log file: {LOG_PATH}")

    best_result = None
    start_time = time.time()

    for run_count, lr_params in enumerate(lr_configs[:max_runs], start=1):
        append_log("")
        append_log(f"[Run {run_count}/{max_runs}] LR: {format_params(lr_params)}")

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
        append_log(f"  Fit time: {result['fit_seconds']:.1f}s")

        if result["convergence_warning_count"]:
            append_log(
                f"  warning: convergence warning "
                f"({result['convergence_warning_count']} time(s))"
            )

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
                "lr_params": lr_params.copy(),
            }
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
    append_log(f"  LogisticRegression: {format_params(best_result['lr_params'])}")
    append_log(f"Saved model: {BEST_MODEL_PATH}")
    append_log(f"Saved best config: {BEST_CONFIG_PATH}")
    append_log("=" * 80)


if __name__ == "__main__":
    main()
