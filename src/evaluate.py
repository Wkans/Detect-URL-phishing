"""
evaluate.py
-----------
Đánh giá mô hình Random Forest (đầu ra cuối cùng của pipeline lai) trên tập test.
In ra: Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

from config import RF_MODEL_PATH


def evaluate():
    data = np.load("../data/test_fused_features.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    rf_model = joblib.load(RF_MODEL_PATH)

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    print("===== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST =====")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nBáo cáo chi tiết:\n", classification_report(y_test, y_pred, target_names=["Benign", "Phishing"]))

    # ----- Confusion Matrix -----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"])
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title("Confusion Matrix - Hybrid CNN-LSTM + Random Forest")
    plt.tight_layout()
    plt.savefig("../data/confusion_matrix.png", dpi=150)
    plt.show()

    # ----- ROC Curve -----
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("../data/roc_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    evaluate()
