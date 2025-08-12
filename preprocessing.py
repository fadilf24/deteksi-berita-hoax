import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# -----------------------------------
# Inisialisasi global: Stopword dan Stemmer
# -----------------------------------

def init_stopwords():
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())

def init_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stop_words = init_stopwords()
stemmer = init_stemmer()

# -----------------------------------
# Fungsi Preprocessing Teks Tunggal
# -----------------------------------

def cleansing(text):
    """
    Tahap 6: Cleansing - Membersihkan teks dari URL, angka, simbol, dan karakter non-alfabet.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def case_folding(text):
    """
    Tahap 7: Case folding - Mengubah teks menjadi huruf kecil.
    """
    return text.lower()

def tokenize(text):
    """
    Tahap 8: Tokenizing - Memecah teks menjadi token.
    """
    return wordpunct_tokenize(text)

def remove_stopwords(tokens):
    """
    Tahap 9: Stopword Removal - Menghapus kata-kata umum.
    """
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    """
    Tahap 10: Stemming - Mengembalikan kata ke bentuk dasar.
    """
    kalimat = " ".join(tokens)
    hasil_stem = stemmer.stem(kalimat)
    return hasil_stem.split()

def filter_token_length(tokens, min_len=4, max_len=25):
    """
    Tahap 11: Filter token berdasarkan panjang karakter.
    """
    return [t for t in tokens if min_len <= len(t) <= max_len]

def preprocess_text(text):
    """
    Proses gabungan 6-11 untuk kolom DataFrame.
    """
    if not isinstance(text, str):
        text = str(text)

    text = cleansing(text)
    text = case_folding(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming_tokens(tokens)
    tokens = filter_token_length(tokens)

    return " ".join(tokens)

# -----------------------------------
# Preprocessing untuk DataFrame
# -----------------------------------

def load_and_clean_data(df1, df2):
    """
    Gabungan Dataset:
    1. Tambah label jika tidak ada
    2. Pilih kolom penting
    3. Sesuaikan nama kolom
    4. Gabungkan dataset
    """

    if 'label' not in df2.columns:
        df2['label'] = 'Non-Hoax'
    else:
        df2['label'] = df2['label'].fillna('Non-Hoax')

    #df2 = df2.rename(columns={
    #    "Judul": "judul",
    #    "Isi": "narasi",
    #    "Konten": "narasi",
    #    "T_judul": "judul",
    #    "T_isi": "narasi",
    #    "T_label": "label",
    #    "Label": "label"
    #})

    expected_cols = ["judul", "narasi", "label"]
    df2 = df2[[col for col in expected_cols if col in df2.columns]]
    df1 = df1[[col for col in expected_cols if col in df1.columns]]

    df1 = df1.loc[:, ~df1.columns.duplicated()]
    df2 = df2.loc[:, ~df2.columns.duplicated()]

    df = pd.concat([df1, df2], ignore_index=True)

    kolom_tidak_dipakai = ["ID", "Tanggal", "tanggal", "Link", "nama file gambar"]
    df = df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], errors="ignore")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("?", np.nan)

    df.dropna(subset=["judul", "narasi", "Label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def preprocess_dataframe(df):
    """
    Gabungkan judul + narasi jadi 'text', lalu buat 'T_text' hasil preprocessing.
    """
    df["judul"] = df["judul"].astype(str)
    df["narasi"] = df["narasi"].astype(str)

    df["text"] = df["judul"] + " " + df["narasi"]
    df["T_text"] = df["text"].apply(preprocess_text)

    return df

def preprocess_with_steps(text):
    """
    Menampilkan hasil tiap tahapan preprocessing (untuk debug atau edukasi).
    """
    hasil = {}
    hasil['original'] = text
    hasil['cleansing'] = cleansing(text)
    hasil['case_folding'] = case_folding(hasil['cleansing'])
    hasil['tokenizing'] = tokenize(hasil['case_folding'])
    hasil['stopword_removal'] = remove_stopwords(hasil['tokenizing'])
    hasil['stemming'] = stemming_tokens(hasil['stopword_removal'])
    hasil['filtering'] = filter_token_length(hasil['stemming'])
    hasil['final'] = " ".join(hasil['filtering'])
    return hasil



