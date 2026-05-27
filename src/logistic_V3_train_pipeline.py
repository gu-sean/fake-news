import itertools
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

TRAIN_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "subset_01.csv"),
    os.path.join(PROJECT_DIR, "dataset", "subset_01.csv"),
    os.path.join(PROJECT_DIR, "DevideCSVs", "subsets", "subset_01.csv"),
]

TEXT_COLUMNS = ["title", "content"]
LABEL_COLUMN = "label"
PROBLEM_TYPE_COLUMNS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

TFIDF_GRID = {
    "ngram_range": [(1, 1), (1, 2), (1, 3)],
    "max_features": [10000, 30000, 50000, 100000],
    "min_df": [1, 2, 3, 5],
    "max_df": [0.85, 0.9, 0.95],
    # None means sklearn default analyzer ("word").
    "analyzer": ["char"],
}

LR_GRID = {
    "C": [0.1, 0.5, 1, 2, 4, 8],
    "penalty": ["l1", "l2"],
    # The source note appears to mean "saga"; sklearn has no "sega" solver.
    "solver": ["liblinear", "saga"],
    "class_weight": ["balanced"],
}

LOG_PATH = os.path.join(SCRIPT_DIR, "log.txt")
BEST_VECTORIZER_PATH = os.path.join(SCRIPT_DIR, "best_vectorizer.pkl")
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pkl")
BEST_CONFIG_PATH = os.path.join(SCRIPT_DIR, "best_config.txt")
RANDOM_STATE = 42
MAX_ITER = 3000


def find_train_file():
    for filename in TRAIN_CANDIDATES:
        if os.path.exists(filename):
            return filename
    return None


def iter_param_grid(grid):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def make_vectorizer(params):
    vectorizer_params = {
        "ngram_range": params["ngram_range"],
        "max_features": params["max_features"],
        "min_df": params["min_df"],
        "max_df": params["max_df"],
        "sublinear_tf": True,
        "norm": "l2",
    }
    if params["analyzer"] is not None:
        vectorizer_params["analyzer"] = params["analyzer"]
    return TfidfVectorizer(**vectorizer_params)


def make_model(params):
    return LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver=params["solver"],
        class_weight=params["class_weight"],
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )


def append_log(message):
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
        log_file.flush()


def format_params(params):
    formatted = []
    for key, value in params.items():
        formatted.append(f"{key}={value if value is not None else 'default'}")
    return ", ".join(formatted)


def score_problem_types(y_true, y_pred, problem_types):
    scores = {}
    for column in problem_types.columns:
        mask = problem_types[column].fillna(0).astype(int) == 1
        if mask.sum() == 0:
            scores[column] = "no test samples"
            continue

        type_y_true = y_true[mask.to_numpy()]
        type_y_pred = y_pred[mask.to_numpy()]
        if len(np.unique(type_y_true)) < 2:
            scores[column] = "only one label in test samples"
            continue

        scores[column] = f1_score(
            type_y_true,
            type_y_pred,
            average="macro",
            zero_division=0,
        )
    return scores


def write_best_config(best_result):
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        config_file.write("Best model configuration\n")
        config_file.write("=" * 60 + "\n")
        config_file.write(f"Macro F1: {best_result['f1']:.4f}\n")
        config_file.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        config_file.write(f"Precision: {best_result['precision']:.4f}\n")
        config_file.write(f"Recall: {best_result['recall']:.4f}\n")
        config_file.write(f"TF-IDF: {format_params(best_result['tfidf_params'])}\n")
        config_file.write(f"LogisticRegression: {format_params(best_result['lr_params'])}\n")


def main():
    train_filename = find_train_file()
    if train_filename is None:
        print("Training data was not found. Checked:")
        for candidate in TRAIN_CANDIDATES:
            print(f"  - {candidate}")
        sys.exit(1)

    print(f"Loading training data: {train_filename}")
    try:
        total_df = pd.read_csv(train_filename)
    except Exception as exc:
        print(f"Failed to load training data: {exc}")
        sys.exit(1)

    missing_columns = [col for col in TEXT_COLUMNS + [LABEL_COLUMN] if col not in total_df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        sys.exit(1)

    available_problem_types = [
        column for column in PROBLEM_TYPE_COLUMNS if column in total_df.columns
    ]

    total_df = total_df.dropna(subset=TEXT_COLUMNS + [LABEL_COLUMN]).reset_index(drop=True)
    if len(total_df) < 10:
        print("Not enough rows to split train/test data.")
        sys.exit(1)

    X = total_df["title"].astype(str) + " " + total_df["content"].astype(str)
    y = total_df[LABEL_COLUMN].astype(int).to_numpy()
    problem_type_df = total_df[available_problem_types].copy()

    split_data = train_test_split(
        X,
        y,
        problem_type_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test, y_train, y_test, _, problem_type_test = split_data

    tfidf_configs = list(iter_param_grid(TFIDF_GRID))
    lr_configs = list(iter_param_grid(LR_GRID))
    total_runs = len(tfidf_configs) * len(lr_configs)
    max_runs_env = os.getenv("MAX_TUNING_RUNS")
    max_runs = int(max_runs_env) if max_runs_env else total_runs
    max_runs = min(max_runs, total_runs)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("TF-IDF + LogisticRegression tuning results\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Training file: {train_filename}\n")
        log_file.write(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")
        log_file.write(f"Planned runs: {max_runs} / {total_runs}\n")
        log_file.write(f"Problem type columns: {available_problem_types or 'none'}\n")
        log_file.write("=" * 80 + "\n")

    append_log(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    append_log(f"Running {max_runs} / {total_runs} tuning combinations")
    append_log(f"Log file: {LOG_PATH}")

    best_result = None
    run_count = 0
    start_time = time.time()

    for tfidf_index, tfidf_params in enumerate(tfidf_configs, start=1):
        if run_count >= max_runs:
            break

        append_log("")
        append_log(f"[TF-IDF {tfidf_index}/{len(tfidf_configs)}] {format_params(tfidf_params)}")

        try:
            vectorizer = make_vectorizer(tfidf_params)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
        except Exception as exc:
            append_log(f"  vectorizer failed: {exc}")
            continue

        for lr_params in lr_configs:
            if run_count >= max_runs:
                break

            run_count += 1
            append_log(f"[Run {run_count}/{max_runs}] LR: {format_params(lr_params)}")

            try:
                model = make_model(lr_params)
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X_train_tfidf, y_train)
                convergence_warnings = [
                    warning for warning in caught_warnings
                    if issubclass(warning.category, ConvergenceWarning)
                ]
            except Exception as exc:
                append_log(f"  model failed: {exc}")
                continue

            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
            problem_type_scores = score_problem_types(y_test, y_pred, problem_type_test)

            append_log(
                f"  Accuracy={accuracy:.4f} | Precision={precision:.4f} "
                f"| Recall={recall:.4f} | Macro-F1={f1:.4f}"
            )
            if convergence_warnings:
                append_log(f"  warning: convergence warning ({len(convergence_warnings)} time(s))")

            if problem_type_scores:
                append_log("  Problem type Macro-F1:")
                for problem_type, score in problem_type_scores.items():
                    if isinstance(score, float):
                        append_log(f"    - {problem_type}: {score:.4f}")
                    else:
                        append_log(f"    - {problem_type}: {score}")

            if best_result is None or f1 > best_result["f1"]:
                best_result = {
                    "f1": f1,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "tfidf_params": tfidf_params.copy(),
                    "lr_params": lr_params.copy(),
                }
                joblib.dump(vectorizer, BEST_VECTORIZER_PATH)
                joblib.dump(model, BEST_MODEL_PATH)
                write_best_config(best_result)
                append_log("  New best model saved.")

    elapsed = time.time() - start_time
    append_log("")
    append_log("=" * 80)
    append_log(f"Finished {run_count} run(s) in {elapsed / 60:.1f} minutes")

    if best_result is None:
        append_log("No successful model run.")
        sys.exit(1)

    append_log("Best combination")
    append_log(f"  Macro-F1: {best_result['f1']:.4f}")
    append_log(f"  Accuracy: {best_result['accuracy']:.4f}")
    append_log(f"  Precision: {best_result['precision']:.4f}")
    append_log(f"  Recall: {best_result['recall']:.4f}")
    append_log(f"  TF-IDF: {format_params(best_result['tfidf_params'])}")
    append_log(f"  LogisticRegression: {format_params(best_result['lr_params'])}")
    append_log(f"Saved vectorizer: {BEST_VECTORIZER_PATH}")
    append_log(f"Saved model: {BEST_MODEL_PATH}")
    append_log(f"Saved best config: {BEST_CONFIG_PATH}")
    append_log("=" * 80)


if __name__ == "__main__":
    main()
