import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import kagglehub
import umap
from sklearn.preprocessing import StandardScaler

seeds = [1, 2, 3, 4, 5]

# Download latest version
path = kagglehub.dataset_download("ishans24/brain-tumor-dataset")
# print(path)
# exit()

IMG_SIZE = 128

def load_data(base_path):
    X = []
    y = []
    class_names = sorted(os.listdir(base_path))

    for label, cls in enumerate(class_names):
        class_path = os.path.join(base_path, cls)

        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), class_names

X, y, class_names = load_data(path)

# TASK 1 — Flatten + Normalize

X = X.reshape(len(X), -1)
X = X / 255.0

# For storing results across seeds
baseline_scores = []
pca_scores = {50: [], 100: [], 200: [], 300: []}
umap_scores = {2: [], 8: [], 16: [], 20: []}

# MULTI-SEED EXPERIMENT
for seed in seeds:

    print(f"\n===== Running seed {seed} =====")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Scaling (IMPORTANT for PCA & UMAP)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # BASELINE — No PCA / No UMAP
    # ----
    svm_no_pca = LinearSVC(max_iter=5000)
    svm_no_pca.fit(X_train, y_train)

    acc_no_pca = svm_no_pca.score(X_test, y_test)
    baseline_scores.append(acc_no_pca)

    print(f"Baseline Accuracy: {acc_no_pca:.4f}")

    # PCA EXPERIMENT
    pca_components = [50, 100, 200, 300]

    for n in pca_components:
        print(f"\n--- PCA with {n} components ---")

        pca = PCA(n_components=n, svd_solver='randomized')

        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        svm = LinearSVC(max_iter=5000)
        svm.fit(X_train_pca, y_train)

        acc = svm.score(X_test_pca, y_test)

        print(f"Accuracy (PCA {n}): {acc:.4f}")

        pca_scores[n].append(acc)

    # UMAP EXPERIMENT
    umap_components = [2, 8, 16, 20]

    np.random.seed(seed) # To generate random variables

    for n in umap_components:
        print(f"\n--- UMAP with {n} components ---")

        umap_model = umap.UMAP(
            n_components=n,
            n_neighbors=40,
            min_dist=0.0,
            metric='euclidean',
            random_state=42
        )

        idx = np.random.choice(len(X_train), 3000, replace=False)
        X_train_subset = X_train[idx]

        umap_model.fit(X_train_subset)

        X_train_umap = umap_model.transform(X_train)
        X_test_umap = umap_model.transform(X_test)

        svm = LinearSVC(max_iter=5000, C=0.1)
        svm.fit(X_train_umap, y_train)

        acc = svm.score(X_test_umap, y_test)

        print(f"Accuracy (UMAP {n}): {acc:.4f}")

        umap_scores[n].append(acc)

# Compute mean/std
def mean_std(arr):
    return np.mean(arr), np.std(arr)

baseline_mean, baseline_std = mean_std(baseline_scores)

pca_components = [50, 100, 200, 300]
umap_components = [2, 8, 16, 20]

pca_means = {n: mean_std(pca_scores[n])[0] for n in pca_components}
pca_stds  = {n: mean_std(pca_scores[n])[1] for n in pca_components}

umap_means = {n: mean_std(umap_scores[n])[0] for n in umap_components}
umap_stds  = {n: mean_std(umap_scores[n])[1] for n in umap_components}

# FINAL RESULTS
print("\n=== FINAL RESULTS ===")
print(f"No Reduction: {baseline_mean:.4f} ± {baseline_std:.4f}")

print("\nPCA Results:")
for n in pca_components:
    print(f"PCA {n}: {pca_means[n]:.4f} ± {pca_stds[n]:.4f}")

print("\nUMAP Results:")
for n in umap_components:
    print(f"UMAP {n}: {umap_means[n]:.4f} ± {umap_stds[n]:.4f}")

# Visualization

methods = ['No Reduction'] + \
          [f'PCA ({n})' for n in pca_components] + \
          [f'UMAP ({n})' for n in umap_components]

means = [baseline_mean] + \
        [pca_means[n] for n in pca_components] + \
        [umap_means[n] for n in umap_components]

stds = [baseline_std] + \
       [pca_stds[n] for n in pca_components] + \
       [umap_stds[n] for n in umap_components]

plt.figure(figsize=(12,5))
plt.bar(methods, means, yerr=stds, capsize=5)
plt.ylabel("Accuracy")
plt.title("Comparison: PCA vs UMAP vs No Reduction")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')

for i, v in enumerate(means):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=8)

plt.tight_layout()
plt.show()
