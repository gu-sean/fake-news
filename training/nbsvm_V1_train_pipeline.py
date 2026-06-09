"""Fast LinearSVC tuning pipeline for the pre-vectorized Korean TF-IDF data."""

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VECTOR_DIR = SCRIPT_DIR / "vector"
LOG_DIR = PROJECT_DIR / "log"
PKL_DIR = PROJECT_DIR / "pkl"
CSV_DIR = PROJECT_DIR / "csv"
JSON_DIR = PROJECT_DIR / "json"
TFIDF_PATH = VECTOR_DIR / "korean_tfidf.npz"
LABEL_PATH = VECTOR_DIR / "korean_labels.npy"
DVL_FLAG_PATH = VECTOR_DIR / "korean_dvl_flags.npy"

LOG_PATH = LOG_DIR / "nbsvm_V1_log.txt"
BEST_MODEL_PATH = PKL_DIR / "best_model_nbsvm_V1.pkl"
BEST_ARTIFACT_PATH = PKL_DIR / "best_artifact_nbsvm_V1.pkl"
BEST_CONFIG_PATH = LOG_DIR / "best_config_nbsvm_V1.txt"
RESULTS_CSV_PATH = CSV_DIR / "performance_nbsvm_V1.csv"
RESULTS_JSON_PATH = JSON_DIR / "performance_nbsvm_V1.json"
PREDICTIONS_PATH = CSV_DIR / "nbsvm_V1_test_predictions.csv"

PROBLEM_TYPE_COLUMNS = [
    "stat_distortion",
    "causal_error",
    "emotional_provocation",
    "source_lack",
    "img_mismatch",
]

RANDOM_STATE = 42
BASE_TEST_SIZE = 0.2
FINAL_TEST_SIZE_WITHIN_HOLDOUT = 0.5

# A preliminary local search around the imported C=5.0 showed that stronger
# regularization consistently improved validation F1, so V1 focuses here.
C_CANDIDATES = [0.075, 0.1, 0.125, 0.15, 0.2]
BASE_PARAMS = {
    "penalty": "l2",
    "loss": "squared_hinge",
    "class_weight": "balanced",
    "dual": False,
    "tol": 1e-4,
    "max_iter": 5000,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quickly tune LinearSVC around C=5.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=len(C_CANDIDATES),
        help="Maximum number of C candidates to run.",
    )
    parser.add_argument(
        "--max-vector-rows",
        type=int,
        default=None,
        help="Optional stratified total row limit for a smoke test.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.02,
        help="Validation decision-threshold search step from -0.5 to 0.5.",
    )
    return parser.parse_args()


def append_log(message=""):
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(message + "\n")
        file.flush()


def ensure_output_directories():
    for directory in [LOG_DIR, PKL_DIR, CSV_DIR, JSON_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_inputs():
    missing = [
        str(path)
        for path in [TFIDF_PATH, LABEL_PATH, DVL_FLAG_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Required files were not found:\n  - " + "\n  - ".join(missing))


def load_data(max_rows=None):
    validate_inputs()
    X = sparse.load_npz(TFIDF_PATH)
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    y = np.asarray(np.load(LABEL_PATH)).astype(int).ravel()
    flags = np.asarray(np.load(DVL_FLAG_PATH))
    if flags.ndim == 1:
        flags = flags.reshape(-1, 1)

    counts = {"X": X.shape[0], "y": len(y), "flags": len(flags)}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Input row counts do not match: {counts}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Labels must be binary 0/1, received {np.unique(y).tolist()}.")

    if max_rows and 20 <= max_rows < len(y):
        X, _, y, _, flags, _ = train_test_split(
            X,
            y,
            flags,
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    return X, y, flags


def split_data(X, y, flags):
    X_train, X_holdout, y_train, y_holdout, _, flags_holdout = train_test_split(
        X,
        y,
        flags,
        test_size=BASE_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test, flags_val, flags_test = train_test_split(
        X_holdout,
        y_holdout,
        flags_holdout,
        test_size=FINAL_TEST_SIZE_WITHIN_HOLDOUT,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, flags_val, flags_test


def make_model(C):
    return LinearSVC(C=C, random_state=RANDOM_STATE, **BASE_PARAMS)


def metric_summary(y_true, scores, threshold):
    prediction = (scores > threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="macro", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "class_0_f1": float(f1[0]),
        "class_1_f1": float(f1[1]),
        "class_0_recall": float(recall[0]),
        "class_1_recall": float(recall[1]),
        "class_0_support": int(support[0]),
        "class_1_support": int(support[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_val, scores, step):
    if not 0 < step <= 0.5:
        raise ValueError("--threshold-step must be greater than 0 and no greater than 0.5.")
    best = None
    for threshold in np.arange(-0.5, 0.500001, step):
        metrics = metric_summary(y_val, scores, threshold)
        rank = (metrics["macro_f1"], metrics["accuracy"], -abs(threshold))
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "metrics": metrics}
    return best["metrics"]


def problem_type_scores(y_true, prediction, flags):
    results = {}
    names = PROBLEM_TYPE_COLUMNS[: flags.shape[1]]
    names.extend(
        f"problem_type_{index + 1}" for index in range(len(names), flags.shape[1])
    )
    for index, name in enumerate(names):
        mask = flags[:, index].astype(bool)
        if not np.any(mask):
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[mask], prediction[mask], average="macro", zero_division=0
        )
        fake_mask = y_true[mask] == 1
        fake_recall = (
            float(np.mean(prediction[mask][fake_mask] == 1))
            if np.any(fake_mask)
            else None
        )
        results[name] = {
            "samples": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], prediction[mask])),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "fake_recall": fake_recall,
        }
    return results


def write_results_csv(results):
    metric_fields = [
        "C",
        "threshold",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "roc_auc",
        "fit_seconds",
        "n_iter",
        "convergence_warnings",
    ]
    dvl_fields = []
    for problem_type in PROBLEM_TYPE_COLUMNS:
        dvl_fields.extend(
            [
                f"{problem_type}_accuracy",
                f"{problem_type}_macro_f1",
                f"{problem_type}_fake_recall",
                f"{problem_type}_samples",
            ]
        )
    fields = metric_fields + dvl_fields
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "C": result["C"],
                **{key: result["validation"][key] for key in metric_fields[1:7]},
                "fit_seconds": result["fit_seconds"],
                "n_iter": result["n_iter"],
                "convergence_warnings": result["convergence_warnings"],
            }
            for problem_type, metrics in result["validation_problem_types"].items():
                for metric_name in ["accuracy", "macro_f1", "fake_recall", "samples"]:
                    row[f"{problem_type}_{metric_name}"] = metrics[metric_name]
            writer.writerow(row)


def save_best_outputs(best, test_metrics, y_test, test_scores, flags_test, all_results):
    model = best["model"]
    threshold = best["validation"]["threshold"]
    prediction = (test_scores > threshold).astype(int)
    problem_scores = problem_type_scores(y_test, prediction, flags_test)

    joblib.dump(model, BEST_MODEL_PATH, compress=3)
    artifact = {
        "artifact_version": 1,
        "name": "nbsvm_V1",
        "model": model,
        "threshold": threshold,
        "params": model.get_params(),
        "expected_features": int(model.n_features_in_),
        "positive_class": 1,
    }
    joblib.dump(artifact, BEST_ARTIFACT_PATH, compress=3)

    report = {
        "best_C": best["C"],
        "best_params": model.get_params(),
        "validation": best["validation"],
        "test": test_metrics,
        "problem_types": problem_scores,
        "candidates": [
            {
                "C": result["C"],
                "validation": result["validation"],
                "validation_problem_types": result["validation_problem_types"],
                "fit_seconds": result["fit_seconds"],
                "n_iter": result["n_iter"],
                "convergence_warnings": result["convergence_warnings"],
            }
            for result in all_results
        ],
    }
    RESULTS_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "Best model configuration - nbsvm V1 pre-vectorized LinearSVC",
        "=" * 68,
        f"C: {best['C']}",
        f"Threshold selected on validation: {threshold:.4f}",
        f"Final test Macro F1: {test_metrics['macro_f1']:.6f}",
        f"Final test Accuracy: {test_metrics['accuracy']:.6f}",
        f"Final test Macro precision: {test_metrics['macro_precision']:.6f}",
        f"Final test Macro recall: {test_metrics['macro_recall']:.6f}",
        f"Final test ROC-AUC: {test_metrics['roc_auc']:.6f}",
        (
            "Confusion Matrix [tn, fp, fn, tp]: "
            f"[{test_metrics['tn']}, {test_metrics['fp']}, "
            f"{test_metrics['fn']}, {test_metrics['tp']}]"
        ),
        "",
        "Problem type performance",
    ]
    for name, metrics in problem_scores.items():
        lines.append(
            f"- {name}: samples={metrics['samples']}, accuracy={metrics['accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"fake_recall={metrics['fake_recall']:.6f}"
        )
    BEST_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(test_scores, -35, 35)))
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_label", "predicted_label", "decision_score", "sigmoid_score"])
        writer.writerows(zip(y_test, prediction, test_scores, probabilities))


def main():
    args = parse_args()
    ensure_output_directories()
    LOG_PATH.write_text(
        "nbsvm V1 quick LinearSVC tuning log\n" + "=" * 80 + "\n",
        encoding="utf-8",
    )
    started = time.time()
    X, y, flags = load_data(args.max_vector_rows)
    X_train, X_val, X_test, y_train, y_val, y_test, flags_val, flags_test = split_data(
        X, y, flags
    )
    candidates = C_CANDIDATES[: max(1, min(args.max_runs, len(C_CANDIDATES)))]

    append_log(f"Loaded TF-IDF: shape={X.shape}, nnz={X.nnz:,}")
    append_log(
        f"Rows: train={len(y_train):,}, validation={len(y_val):,}, test={len(y_test):,}"
    )
    append_log(f"C candidates: {candidates}")

    results = []
    best = None
    for index, C in enumerate(candidates, start=1):
        append_log()
        append_log(f"[Run {index}/{len(candidates)}] LinearSVC C={C}")
        model = make_model(C)
        fit_started = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        fit_seconds = time.time() - fit_started
        convergence_warnings = sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        validation_scores = model.decision_function(X_val)
        validation = tune_threshold(y_val, validation_scores, args.threshold_step)
        validation_prediction = (validation_scores > validation["threshold"]).astype(int)
        validation_problem_types = problem_type_scores(
            y_val, validation_prediction, flags_val
        )
        result = {
            "C": C,
            "model": model,
            "validation": validation,
            "validation_problem_types": validation_problem_types,
            "fit_seconds": fit_seconds,
            "n_iter": int(model.n_iter_),
            "convergence_warnings": convergence_warnings,
        }
        results.append(result)
        append_log(
            f"  Validation: Macro-F1={validation['macro_f1']:.6f} | "
            f"Accuracy={validation['accuracy']:.6f} | ROC-AUC={validation['roc_auc']:.6f} "
            f"| threshold={validation['threshold']:.4f}"
        )
        append_log(
            f"  Fit time={fit_seconds:.1f}s | iterations={model.n_iter_} "
            f"| convergence warnings={convergence_warnings}"
        )
        append_log("  DVL validation performance:")
        for problem_type, metrics in validation_problem_types.items():
            append_log(
                f"    - {problem_type}: samples={metrics['samples']} | "
                f"Accuracy={metrics['accuracy']:.6f} | Macro-F1={metrics['macro_f1']:.6f} "
                f"| Fake recall={metrics['fake_recall']:.6f}"
            )
        rank = (validation["macro_f1"], validation["accuracy"], validation["roc_auc"])
        if best is None or rank > best["rank"]:
            best = {**result, "rank": rank}
            joblib.dump(model, BEST_MODEL_PATH, compress=3)
            append_log("  New best model saved.")

    write_results_csv(results)
    test_scores = best["model"].decision_function(X_test)
    test_metrics = metric_summary(y_test, test_scores, best["validation"]["threshold"])
    save_best_outputs(best, test_metrics, y_test, test_scores, flags_test, results)

    append_log()
    append_log("=" * 80)
    append_log(f"Best C: {best['C']}")
    append_log(f"Best validation Macro-F1: {best['validation']['macro_f1']:.6f}")
    append_log(f"Final test Macro-F1: {test_metrics['macro_f1']:.6f}")
    append_log(f"Final test Accuracy: {test_metrics['accuracy']:.6f}")
    append_log(f"Final test ROC-AUC: {test_metrics['roc_auc']:.6f}")
    append_log("Final test DVL performance:")
    test_prediction = (test_scores > best["validation"]["threshold"]).astype(int)
    for problem_type, metrics in problem_type_scores(
        y_test, test_prediction, flags_test
    ).items():
        append_log(
            f"  - {problem_type}: samples={metrics['samples']} | "
            f"Accuracy={metrics['accuracy']:.6f} | Macro-F1={metrics['macro_f1']:.6f} "
            f"| Fake recall={metrics['fake_recall']:.6f}"
        )
    append_log(f"Saved model: {BEST_MODEL_PATH}")
    append_log(f"Saved artifact: {BEST_ARTIFACT_PATH}")
    append_log(f"Elapsed: {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
