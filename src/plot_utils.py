import matplotlib.pyplot as plt

def save_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    
def save_f1_bar_chart(report_dict, class_names, title, filename):
    f1_scores = [report_dict[class_name]["f1-score"] for class_name in class_names]

    plt.figure(figsize=(7, 5))
    plt.bar(class_names, f1_scores)
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def save_pca_plot(pca_dims, pca_means, baseline_mean, filename):
    plt.figure()
    plt.plot(pca_dims, pca_means, marker="o", label="PCA + SVM")
    plt.axhline(y=baseline_mean, linestyle="--", label="Baseline SVM")
    plt.xlabel("PCA Components")
    plt.ylabel("Accuracy")
    plt.title("SVM + PCA Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_pca_umap_plot(pca_dims, pca_means, baseline_mean, umap_dims, umap_means, filename):
    plt.figure()
    plt.plot(pca_dims, pca_means, marker="o", label="PCA + SVM")
    plt.axhline(y=baseline_mean, linestyle=":", label="Baseline SVM")

    for d, mean in zip(umap_dims, umap_means):
        plt.axhline(y=mean, linestyle="--", label=f"UMAP {d} + SVM")

    plt.xlabel("PCA Components")
    plt.ylabel("Accuracy")
    plt.title("PCA vs UMAP vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_multiseed_plot(seeds, baseline_results, pca_results, umap_results, filename):
    plt.figure()
    plt.plot(seeds, baseline_results, marker="o", label="Baseline")

    for d, values in pca_results.items():
        plt.plot(seeds, values, marker="o", label=f"PCA {d}")

    for d, values in umap_results.items():
        plt.plot(seeds, values, marker="o", label=f"UMAP {d}")

    plt.xlabel("Seed")
    plt.ylabel("Accuracy")
    plt.title("Multi-seed Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_method_comparison_bar(labels, means, filename):
    plt.figure()
    plt.bar(labels, means)
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy by Method")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()