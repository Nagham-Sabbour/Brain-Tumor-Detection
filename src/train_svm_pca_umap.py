import os
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import umap

from data_utils import load_full_dataset, split_and_scale_data
from eval_utils import (
    append_to_log,
    evaluate_model,
    log_evaluation,
    save_results_json,
    save_results_pickle,
)
from plot_utils import (
    save_confusion_matrix,
    save_pca_plot,
    save_pca_umap_plot,
    save_multiseed_plot,
    save_method_comparison_bar,
)

def main():
    # Experiment configuration
    seeds = [0, 1, 2]
    pca_dims = [50, 100, 200]
    umap_dims = [20, 50, 100]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("outputs", f"SVM_comparisons_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run_log.txt")

    append_to_log(log_path, f"Run directory: {run_dir}")
    append_to_log(log_path, "Model: LinearSVC")
    append_to_log(log_path, "Settings: max_iter=5000, tol=1e-3")
    append_to_log(log_path, f"Seeds: {seeds}")
    append_to_log(log_path, f"PCA dims: {pca_dims}")
    append_to_log(log_path, f"UMAP dims: {umap_dims}")
    append_to_log(log_path, "-" * 40)

    results = {
        "baseline": [],
        "pca": {d: [] for d in pca_dims},
        "umap": {d: [] for d in umap_dims},
        "details": {
            "baseline": [],
            "pca": {d: [] for d in pca_dims},
            "umap": {d: [] for d in umap_dims}
        }
    }

    print("Loading and preprocessing full dataset once...")
    X, y, class_names = load_full_dataset()

    # Run experiments
    for seed in seeds:
        print("\n" + "-" * 40)
        print(f"Running experiments with seed {seed}")
        print("-" * 40)

        append_to_log(log_path, "-" * 40)
        append_to_log(log_path, f"Running experiments with seed {seed}")
        append_to_log(log_path, "-" * 40)

        X_train, X_test, y_train, y_test = split_and_scale_data(
            X,
            y,
            test_size=0.2,
            random_state=seed
        )

        append_to_log(log_path, f"Classes: {class_names}")
        append_to_log(log_path, f"Train samples: {X_train.shape[0]}")
        append_to_log(log_path, f"Test samples: {X_test.shape[0]}")
        append_to_log(log_path, f"Feature dimension: {X_train.shape[1]}")

        # Baseline SVM
        print("Training baseline SVM...")
        svm = LinearSVC(max_iter=5000, tol=1e-3)
        svm.fit(X_train, y_train)

        baseline_eval = evaluate_model(
            svm,
            X_test,
            y_test,
            class_names,
            label="Baseline SVM",
            seed=seed
        )
        log_evaluation(log_path, "Baseline SVM", baseline_eval, seed=seed)

        results["baseline"].append(baseline_eval["accuracy"])
        results["details"]["baseline"].append({
            "seed": seed,
            "accuracy": baseline_eval["accuracy"],
            "report": baseline_eval["report_dict"],
            "confusion_matrix": baseline_eval["confusion_matrix"].tolist()
        })

        if seed == 0:
            save_confusion_matrix(
                baseline_eval["confusion_matrix"],
                class_names,
                f"Baseline SVM Confusion Matrix (seed={seed})",
                os.path.join(run_dir, f"cm_baseline_svm_seed{seed}.webp")
            )

        # PCA Experiments
        for d in pca_dims:
            print(f"Training PCA + SVM (d={d})...")
            pca = PCA(n_components=d)

            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            svm = LinearSVC(max_iter=5000, tol=1e-3)
            svm.fit(X_train_pca, y_train)

            pca_eval = evaluate_model(
                svm,
                X_test_pca,
                y_test,
                class_names,
                label=f"PCA + SVM ({d})",
                seed=seed
            )
            log_evaluation(log_path, f"PCA + SVM ({d})", pca_eval, seed=seed)

            results["pca"][d].append(pca_eval["accuracy"])
            results["details"]["pca"][d].append({
                "seed": seed,
                "accuracy": pca_eval["accuracy"],
                "report": pca_eval["report_dict"],
                "confusion_matrix": pca_eval["confusion_matrix"].tolist()
            })

            if seed == 0:
                safe_name = f"cm_pca_svm_{d}_seed{seed}.webp"
                save_confusion_matrix(
                    pca_eval["confusion_matrix"],
                    class_names,
                    f"PCA + SVM ({d}) Confusion Matrix (seed={seed})",
                    os.path.join(run_dir, safe_name)
                )

        # UMAP Experiment
        for d in umap_dims:
            print(f"Training UMAP + SVM (d={d})...")
            reducer = umap.UMAP(n_components=d, random_state=seed)

            X_train_umap = reducer.fit_transform(X_train)
            X_test_umap = reducer.transform(X_test)

            svm = LinearSVC(max_iter=5000, tol=1e-3)
            svm.fit(X_train_umap, y_train)

            umap_eval = evaluate_model(
                svm,
                X_test_umap,
                y_test,
                class_names,
                label=f"UMAP + SVM ({d})",
                seed=seed
            )

            log_evaluation(log_path, f"UMAP + SVM ({d})", umap_eval, seed=seed)

            results["umap"][d].append(umap_eval["accuracy"])
            results["details"]["umap"][d].append({
                "seed": seed,
                "accuracy": umap_eval["accuracy"],
                "report": umap_eval["report_dict"],
                "confusion_matrix": umap_eval["confusion_matrix"].tolist()
            })

            if seed == 0:
                save_confusion_matrix(
                    umap_eval["confusion_matrix"],
                    class_names,
                    f"UMAP + SVM ({d}) Confusion Matrix (seed={seed})",
                    os.path.join(run_dir, f"cm_umap_svm_{d}_seed{seed}.webp")
                )

    # Results
    print("\n" + "-" * 40)
    print("Final Mean Results Across Seeds")
    print("-" * 40)

    append_to_log(log_path, "-" * 40)
    append_to_log(log_path, "Final Mean Results Across Seeds")
    append_to_log(log_path, "-" * 40)

    baseline_mean = np.mean(results["baseline"])
    baseline_std = np.std(results["baseline"])
    print(f"Baseline SVM: {baseline_mean:.4f} ± {baseline_std:.4f}")
    append_to_log(log_path, f"Baseline SVM: {baseline_mean:.4f} ± {baseline_std:.4f}")

    pca_means = []
    for d in pca_dims:
        mean_acc = np.mean(results["pca"][d])
        std_acc = np.std(results["pca"][d])
        pca_means.append(mean_acc)
        print(f"PCA({d}): {mean_acc:.4f} ± {std_acc:.4f}")
        append_to_log(log_path, f"PCA({d}): {mean_acc:.4f} ± {std_acc:.4f}")

    umap_means = []
    for d in umap_dims:
        mean_acc = np.mean(results["umap"][d])
        std_acc = np.std(results["umap"][d])
        umap_means.append(mean_acc)
        print(f"UMAP({d}): {mean_acc:.4f} ± {std_acc:.4f}")
        append_to_log(log_path, f"UMAP({d}): {mean_acc:.4f} ± {std_acc:.4f}")

    # Plot PCA results
    save_pca_plot(
        pca_dims,
        pca_means,
        baseline_mean,
        os.path.join(run_dir, "pca.webp")
    )

    # Plot PCA vs UMAP vs Baseline
    save_pca_umap_plot(
        pca_dims,
        pca_means,
        baseline_mean,
        umap_dims,
        umap_means,
        os.path.join(run_dir, "pca_umap.webp")
    )

    # Multi-seed plot
    save_multiseed_plot(
        seeds,
        results["baseline"],
        results["pca"],
        results["umap"],
        os.path.join(run_dir, "pca_umap_multiseed.webp")
    )

    # Bar chart of mean accuracies
    labels = (
        ["Baseline"]
        + [f"PCA {d}" for d in pca_dims]
        + [f"UMAP {d}" for d in umap_dims]
    )

    means = (
        [baseline_mean]
        + pca_means
        + umap_means
    )

    save_method_comparison_bar(
        labels,
        means,
        os.path.join(run_dir, "method_comparison.webp")
    )

    # Save results
    save_results_json(results, os.path.join(run_dir, "pca_umap_results.json"))
    save_results_pickle(results, os.path.join(run_dir, "pca_umap_results.pkl"))

    print(f"\nSaved plots, logs, and results to: {run_dir}")

if __name__ == "__main__":
    main()