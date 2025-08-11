import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
from sklearn.base import ClassifierMixin


def split_data(
    X: Any, 
    y: Any, 
    test_size: float = 0.4, 
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Membagi dataset menjadi data latih dan data uji.

    Args:
        X: Matriks fitur (contoh: hasil TF-IDF)
        y: Label target
        test_size: Proporsi data uji (default 0.2)
        random_state: Nilai seed untuk pengacakan

    Returns:
        Tuple berisi X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_naive_bayes(X_train: Any, y_train: Any) -> MultinomialNB:
    """
    Melatih model klasifikasi Naive Bayes (MultinomialNB).

    Args:
        X_train: Fitur latih
        y_train: Label latih

    Returns:
        Model Naive Bayes yang telah dilatih
    """
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Data latih kosong, tidak bisa melatih model.")
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(model: ClassifierMixin, X_test: Any) -> np.ndarray:
    """
    Melakukan prediksi menggunakan model Naive Bayes.

    Args:
        model: Model klasifikasi yang telah dilatih
        X_test: Fitur data uji

    Returns:
        Array hasil prediksi label
    """
    if X_test.shape[0] == 0:
        raise ValueError("Data uji kosong, tidak bisa melakukan prediksi.")
    
    return model.predict(X_test)


def prediction_distribution(
    y_pred: np.ndarray, 
    label_encoder: LabelEncoder
) -> Dict[str, float]:
    """
    Menghitung distribusi hasil prediksi dalam persentase tiap kelas.

    Args:
        y_pred: Array hasil prediksi (label numerik)
        label_encoder: LabelEncoder untuk mengembalikan label asli

    Returns:
        Dictionary berisi label asli dan persentase kemunculannya
    """
    if y_pred.size == 0:
        return {"empty": 0.0}

    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    percentages = {
        label_encoder.inverse_transform([label])[0]: round((count / total) * 100, 2)
        for label, count in zip(unique, counts)
    }
    return percentages


