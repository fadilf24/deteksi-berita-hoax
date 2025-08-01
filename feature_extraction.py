import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def combine_text_columns(df, col1='T_text', col2=None):
    """
    Gabungkan dua kolom teks (hasil tokenisasi/stemming) jika diberikan.
    Default hanya gabungkan 'T_text' jika 'col2' None.
    """
    df = df.copy()
    if col1 not in df.columns:
        raise ValueError(f"Kolom '{col1}' tidak ditemukan dalam DataFrame.")
    
    df[col1] = df[col1].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    if col2 and col2 in df.columns:
        df[col2] = df[col2].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        df['gabungan'] = df[col1] + ' ' + df[col2]
    else:
        df['gabungan'] = df[col1]
    
    return df

def tfidf_transform(text_series):
    """
    Ubah teks dalam Series menjadi TF-IDF matrix.
    Kembalikan: matrix fitur (sparse matrix) dan vectorizer.
    """
    text_series = text_series.fillna('').astype(str)
    text_series = text_series.apply(lambda x: x.strip())

    if text_series.empty or all(text_series == ''):
        raise ValueError("Tidak ada dokumen valid setelah preprocessing.")

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(text_series)

    return features, vectorizer

def tfidf_to_dataframe(tfidf_matrix, vectorizer):
    """
    Konversi matrix TF-IDF ke DataFrame.
    """
    return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

def save_tfidf_output(df_preprocessed, tfidf_df, preprocessed_path='preprocessed_data.csv', tfidf_path='tfidf_output.csv'):
    """
    Simpan hasil preprocessing dan TF-IDF dalam format CSV.
    """
    try:
        df_preprocessed.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_path}")
    except Exception as e:
        print(f"Gagal menyimpan preprocessed data: {e}")

    try:
        tfidf_df.to_csv(tfidf_path, index=False)
        print(f"TF-IDF output saved to {tfidf_path}")
    except Exception as e:
        print(f"Gagal menyimpan TF-IDF data: {e}")
