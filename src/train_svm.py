import os
from datetime import datetime

from sklearn.svm import LinearSVC

from data_utils import prepare_data
from eval_utils import (
    append_to_log,
    evaluate_model,
    log_evaluation,
    save_results_json,
    save_results_pickle,
)
from plot_utils import save_confusion_matrix, save_f1_bar_chart

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("outputs", f"SVM_baseline_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run_log.txt")

    append_to_log(log_path, f"Run directory: {run_dir}")
    append_to_log(log_path, "Model: LinearSVC")
    append_to_log(log_path, "Settings: max_iter=5000, tol=1e-3")
    append_to_log(log_path, "-" * 60)

    print("Preparing data...")
    X_train, X_test, y_train, y_test, class_names = prepare_data()

    append_to_log(log_path, f"Classes: {class_names}")
    append_to_log(log_path, f"Train samples: {X_train.shape[0]}")
    append_to_log(log_path, f"Test samples: {X_test.shape[0]}")
    append_to_log(log_path, f"Feature dimension: {X_train.shape[1]}")
    append_to_log(log_path, "-" * 60)

    print("Training baseline SVM...")
    model = LinearSVC(max_iter=5000, tol=1e-3, verbose=1)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    eval_results = evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        label="Baseline SVM"
    )

    log_evaluation(log_path, "Baseline SVM", eval_results)

    save_confusion_matrix(
        eval_results["confusion_matrix"],
        class_names,
        "Baseline SVM Confusion Matrix",
        os.path.join(run_dir, "confusion_matrix.webp")
    )

    save_f1_bar_chart(
        eval_results["report_dict"],
        class_names,
        "Baseline SVM Per-Class F1-score",
        os.path.join(run_dir, "f1_scores.webp")
    )

    results = {
        "model": "LinearSVC",
        "settings": {
            "max_iter": 5000,
            "tol": 1e-3
        },
        "accuracy": eval_results["accuracy"],
        "class_names": class_names,
        "classification_report": eval_results["report_dict"],
        "confusion_matrix": eval_results["confusion_matrix"].tolist(),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape)
    }

    save_results_json(results, os.path.join(run_dir, "baseline_results.json"))
    save_results_pickle(results, os.path.join(run_dir, "baseline_results.pkl"))

    print(f"\nSaved baseline results to: {run_dir}")

if __name__ == "__main__":
    main()