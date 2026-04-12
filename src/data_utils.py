import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IMG_SIZE = 64   # Can increase to 128
DATASET_NAME = "ishans24/brain-tumor-dataset"
LOCAL_DATASET_DIR = "data/brain_tumor_dataset"


def get_dataset_path():
    if os.path.exists(LOCAL_DATASET_DIR):
        print(f"Using local dataset at: {LOCAL_DATASET_DIR}")
        return LOCAL_DATASET_DIR

    print("Local dataset not found. Downloading from Kaggle...")
    return kagglehub.dataset_download(DATASET_NAME)


def load_data(base_path, img_size=IMG_SIZE):
    X = []
    y = []

    class_names = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    )

    print(f"Found classes: {class_names}")

    total_loaded = 0
    for label, cls in enumerate(class_names):
        class_path = os.path.join(base_path, cls)
        image_names = os.listdir(class_path)

        print(f"Loading class '{cls}' ({len(image_names)} images)...")

        for i, img_name in enumerate(image_names, start=1):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)
            total_loaded += 1

            if i % 500 == 0:
                print(f"  Loaded {i}/{len(image_names)} images from '{cls}'...")

    print(f"Finished loading data. Total images loaded: {total_loaded}")
    return np.array(X), np.array(y), class_names


def preprocess_images(X):
    print("Flattening images...")
    X = X.reshape(len(X), -1)

    print("Normalizing pixel values to [0, 1]...")
    X = X / 255.0

    print(f"Preprocessed data shape: {X.shape}")
    return X


def load_full_dataset():
    path = get_dataset_path()
    X, y, class_names = load_data(path)
    X = preprocess_images(X)
    return X, y, class_names


def split_and_scale_data(X, y, test_size=0.2, random_state=50):
    print(f"Splitting data into train/test sets with seed={random_state}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    scaler = StandardScaler()
    print("Fitting StandardScaler on training data...")
    X_train = scaler.fit_transform(X_train)

    print("Transforming test data...")
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def prepare_data(test_size=0.2, random_state=50):
    X, y, class_names = load_full_dataset()
    X_train, X_test, y_train, y_test = split_and_scale_data(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, class_names