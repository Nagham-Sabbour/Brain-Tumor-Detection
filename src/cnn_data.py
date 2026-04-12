import os
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


IMG_SIZE = 64
LOCAL_DATASET_DIR = "data/brain_tumor_dataset"


class BrainTumorDataset(Dataset):
    """
    Simple PyTorch dataset for MRI images.
    Returns:
        image tensor of shape (1, 64, 64)
        label as int
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        # convert to uint8 for torchvision transforms
        image = (image * 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            # add channel dimension: (64,64) -> (1,64,64)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)
        return image, label

def load_images_from_folders(base_path: str, img_size: int = IMG_SIZE):
    """
    Loads grayscale images from folder structure like:
    data/brain_tumor_dataset/
        glioma/
        meningioma/
        no_tumor/
        pituitary/
    """
    images = []
    labels = []

    class_names = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    )

    print(f"Found classes: {class_names}")

    total_loaded = 0
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        file_names = os.listdir(class_path)

        print(f"Loading '{class_name}' with {len(file_names)} images...")

        for i, file_name in enumerate(file_names, start=1):
            img_path = os.path.join(class_path, file_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # normalize pixel values to [0,1]

            images.append(img)
            labels.append(label)
            total_loaded += 1

            if i % 500 == 0:
                print(f"  Loaded {i}/{len(file_names)} from {class_name}")

    print(f"Finished loading {total_loaded} images.")
    return np.array(images), np.array(labels), class_names


def create_data_splits(
    images: np.ndarray,
    labels: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Creates train/val/test split with stratification.
    Final split is:
    - 70% train
    - 15% val
    - 15% test
    """
    # first split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # then split remaining into train/val
    val_ratio_adjusted = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_stats(y_train, y_val, y_test, class_names):
    print("\nSplit sizes:")
    print(f"Train: {len(y_train)}")
    print(f"Val:   {len(y_val)}")
    print(f"Test:  {len(y_test)}")

    def count_labels(y):
        return {class_names[k]: v for k, v in sorted(Counter(y).items())}

    print("\nClass distribution:")
    print("Train:", count_labels(y_train))
    print("Val:  ", count_labels(y_val))
    print("Test: ", count_labels(y_test))


def get_dataloaders(
    batch_size: int = 32,
    img_size: int = IMG_SIZE,
    random_state: int = 42,
):
    if not os.path.exists(LOCAL_DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found at '{LOCAL_DATASET_DIR}'. "
            "Please make sure the dataset is downloaded there."
        )

    images, labels, class_names = load_images_from_folders(
        LOCAL_DATASET_DIR, img_size=img_size
    )

    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        images,
        labels,
        random_state=random_state,
    )

    print_split_stats(y_train, y_val, y_test, class_names)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0,translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transform)
    val_dataset = BrainTumorDataset(X_val, y_val, transform=test_transform)
    test_dataset = BrainTumorDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names