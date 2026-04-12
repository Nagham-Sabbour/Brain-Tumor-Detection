import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def append_to_log(log_path, text):
    with open(log_path, "a") as f:
        f.write(text + "\n")

def evaluate_model(model, X_test, y_test, class_names, label, seed=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_test, y_pred)

    header = f"{label} Evaluation"
    if seed is not None:
        header += f" (seed={seed})"

    print("\n" + "-" * 42)
    print(header)
    print("-" * 42)
    print(f"Accuracy: {acc:.4f}")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }

def log_evaluation(log_path, label, eval_results, seed=None):
    header = f"{label} Evaluation"
    if seed is not None:
        header += f" (seed={seed})"

    append_to_log(log_path, "-" * 42)
    append_to_log(log_path, header)
    append_to_log(log_path, "-" * 42)
    append_to_log(log_path, f"Accuracy: {eval_results['accuracy']:.4f}")
    append_to_log(log_path, eval_results["report_text"])
    append_to_log(log_path, "Confusion Matrix:")
    append_to_log(log_path, str(eval_results["confusion_matrix"]))
    append_to_log(log_path, "")

def save_results_json(results, filepath):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

def save_results_pickle(results, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(results, f)