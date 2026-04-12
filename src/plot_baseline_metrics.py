import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Hardcoded metrics from our best baseline run (seed 0)
    class_names = ["Glioma", "Meningioma", "No tumor", "Pituitary"]
    precision = [0.90, 0.78, 0.94, 0.91]
    recall = [0.87, 0.82, 0.89, 0.95]
    f1_score = [0.88, 0.80, 0.92, 0.93]

    output_dir = "outputs/SVM_comparisons_2026-04-05_19-47-53"
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(len(class_names))
    width = 0.25

    # Grouped bar chart
    plt.figure(figsize=(9, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1_score, width, label="F1-score")

    plt.xticks(x, class_names, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Baseline SVM Per-Class Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grouped_bar_chart.webp"), bbox_inches="tight")
    plt.close()

    # Line plot with markers
    plt.figure(figsize=(8, 4.5))
    plt.plot(class_names, precision, marker="o", label="Precision")
    plt.plot(class_names, recall, marker="s", label="Recall")
    plt.plot(class_names, f1_score, marker="^", label="F1-score")

    plt.ylim(0.7, 1.0)
    plt.margins(x=0.02)
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Baseline SVM Per-Class Metrics", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "line_plot_metrics.webp"), bbox_inches="tight")
    plt.close()

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()