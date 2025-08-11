# evaluation.py

import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder
) -> Dict[str, Any]:
    """
    Mengevaluasi performa model menggunakan metrik evaluasi.

    Args:
        y_true: Array label sebenarnya
        y_pred: Array hasil prediksi
        label_encoder: LabelEncoder untuk mengembalikan label asli

    Returns:
        Dictionary berisi nilai akurasi, precision, recall, F1-score, dan classification report
    """
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Label benar atau prediksi kosong, evaluasi tidak dapat dilakukan.")
    
    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Classification report dengan label asli
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=label_encoder.inverse_transform(np.unique(y_true)), 
        zero_division=0
    )

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "classification_report": report
    }
