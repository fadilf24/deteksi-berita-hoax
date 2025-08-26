from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np


def train_naive_bayes(X_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """
    Melatih model Gaussian Naive Bayes dengan data training.
    
    Parameters:
        X_train (np.ndarray): Fitur data training (dense array).
        y_train (np.ndarray): Label data training.
    
    Returns:
        GaussianNB: Model Naive Bayes yang sudah terlatih.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(model: GaussianNB, X_test: np.ndarray):
    """
    Melakukan prediksi menggunakan model Gaussian Naive Bayes.
    
    Parameters:
        model (GaussianNB): Model Naive Bayes yang sudah dilatih.
        X_test (np.ndarray): Fitur data uji (dense array).
    
    Returns:
        tuple: 
            - y_pred (np.ndarray): Label hasil prediksi.
            - y_prob (np.ndarray): Probabilitas prediksi untuk tiap kelas.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return y_pred, y_prob


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mengevaluasi akurasi model berdasarkan data uji.
    
    Parameters:
        y_test (np.ndarray): Label sebenarnya.
        y_pred (np.ndarray): Label hasil prediksi model.
    
    Returns:
        float: Nilai akurasi (0 - 1).
    """
    return accuracy_score(y_test, y_pred)
