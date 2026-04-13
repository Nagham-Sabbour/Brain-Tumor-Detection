import os
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from cnn_data import get_dataloaders
from cnn_model import BrainTumorCNN

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS device.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    print("Using CPU.")
    return torch.device("cpu")

def append_to_log(log_path, text):
    with open(log_path, "a") as f:
        f.write(text + "\n")

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

def save_training_curves(history, run_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_curve.png"))
    plt.close()

def evaluate_model(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def main():
    # config
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50
    random_state = 42

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("outputs", f"CNN_baseline_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run_log.txt")

    append_to_log(log_path, f"Run directory: {run_dir}")
    append_to_log(log_path, "Model: BrainTumorCNN")
    append_to_log(log_path, f"Batch size: {batch_size}")
    append_to_log(log_path, f"Learning rate: {learning_rate}")
    append_to_log(log_path, f"Epochs: {num_epochs}")
    append_to_log(log_path, "-" * 60)

    # data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=batch_size,
        random_state=random_state,
        save_split_dir=run_dir
    )

    device = get_device()

    # model
    model = BrainTumorCNN(num_classes=len(class_names)).to(device)
    weights = torch.tensor([1.0, 1.2, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(model)
    append_to_log(log_path, str(model))

    # training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(run_dir, "best_cnn_model.pt")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, _, _ = evaluate_model(
            model, val_loader, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        msg = (
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        print(msg)
        append_to_log(log_path, msg)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
            print("  Saved new best model.")
            append_to_log(log_path, "Saved new best model.")
        else:
            epochs_without_improvement += 1

    # save training plots
    save_training_curves(history, run_dir)

    # load best model and test
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    append_to_log(log_path, "\nFinal Test Results")
    append_to_log(log_path, f"Test Loss: {test_loss:.4f}")
    append_to_log(log_path, f"Test Accuracy: {test_acc:.4f}")
    append_to_log(log_path, report_text)
    append_to_log(log_path, "Confusion Matrix:")
    append_to_log(log_path, str(cm))

    save_confusion_matrix(
        cm,
        class_names,
        "CNN Confusion Matrix",
        os.path.join(run_dir, "cnn_confusion_matrix.png"),
    )

    results = {
        "model": "BrainTumorCNN",
        "accuracy": test_acc,
        "test_loss": test_loss,
        "class_names": class_names,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "settings": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "random_state": random_state,
        },
    }

    with open(os.path.join(run_dir, "cnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(run_dir, "cnn_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved CNN results to: {run_dir}")

if __name__ == "__main__":
    main()