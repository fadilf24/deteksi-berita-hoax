from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Membagi data menjadi data latih dan data uji.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
def train_naive_bayes(X_train, y_train):
    """
    Melatih Gaussian Naive Bayes dengan data training.
    Pastikan input dikonversi ke dense array.
    """
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def predict_naive_bayes(model, X_test):
    """
    Melakukan prediksi menggunakan model Gaussian Naive Bayes.
    Pastikan input dikonversi ke dense array.
    """
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    return model.predict(X_test), model.predict_proba(X_test)

def evaluate_model(model, X_test, y_test):
    """
    Menghitung akurasi model Gaussian Naive Bayes.
    """
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    return model.score(X_test, y_test)

