import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, accuracy_score)
from src.common import load_datasets, load_model_bundle

def main():
    _, _, test = load_datasets()
    model = load_model_bundle()
    ys, scores = [], []
    for images, labels in test:
        scores.extend(model.predict(images, verbose=0).reshape(-1).tolist())
        ys.extend(labels.numpy().reshape(-1).astype(int).tolist())
    y_true, y_score = np.array(ys), np.array(scores)
    y_pred = (y_score >= 0.5).astype(int)
    names = ["NORMAL", "PNEUMONIA"]
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=names).plot(values_format="d")
    plt.title("Confusion Matrix — Test Set"); plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=160); plt.close()
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "test_samples": int(len(y_true))}
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score)); metrics["roc_auc"] = auc
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Test Set"); plt.legend(); plt.tight_layout()
        plt.savefig("reports/roc_curve.png", dpi=160); plt.close()
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(report); print(json.dumps(metrics, indent=2)); print("Evaluation files saved in reports/")

if __name__ == "__main__":
    main()
