import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.figure_factory as ff
import io
import re
import json
import uuid
from datetime import datetime
import pytz
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from streamlit_option_menu import option_menu
from fpdf import FPDF
import firebase_admin
from firebase_admin import credentials, db
from langdetect import detect_langs, DetectorFactory

# Import modul custom
from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data, preprocess_with_steps
from feature_extraction import combine_text_columns, tfidf_transform
from classification import split_data, train_naive_bayes, predict_naive_bayes
from interpretation import configure_gemini, analyze_with_gemini

DetectorFactory.seed = 0  # agar hasil deteksi bahasa konsisten

# ✅ Validasi teks bahasa Indonesia
def is_indonesian(text, min_prob=0.90):
    if not text.strip():
        return False
    try:
        detections = detect_langs(text)
        return any(lang.lang == "id" and lang.prob >= min_prob for lang in detections)
    except:
        return False

# ✅ Validasi teks input minimal
def is_valid_text(text):
    words = re.findall(r'\w+', text)
    return len(words) >= 5 and any(len(word) > 3 for word in words)

# ✅ Konfigurasi halaman Streamlit
st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🔎", layout="wide")

# ✅ Konfigurasi Firebase
firebase_cred = st.secrets["FIREBASE_KEY"]
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(firebase_cred)
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://deteksi-berita-hoaks-default-rtdb.asia-southeast1.firebasedatabase.app/"
        })
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

def simpan_ke_firebase(data):
    try:
        tz = pytz.timezone("Asia/Jakarta")
        waktu_wib = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        data["timestamp"] = waktu_wib
        ref = db.reference("prediksi_hoaks")
        ref.child(str(uuid.uuid4())).set(data)
    except Exception as e:
        st.error(f"Gagal menyimpan ke Firebase: {e}")

def read_predictions_from_firebase():
    try:
        ref = db.reference("prediksi_hoaks")
        data = ref.get()
        if isinstance(data, dict):
            return pd.DataFrame(data.values())
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca data dari Firebase: {e}")
        return pd.DataFrame()

# ✅ Sidebar Navigasi
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Deteksi Hoaks", "Dataset", "Split Data", "Preprocessing", "Evaluasi Model", "Riwayat Prediksi", "Info Sistem"],
        icons=["search", "folder", "shuffle", "tools", "bar-chart", "clock-history", "cpu"],
        default_index=0,
        orientation="vertical"
    )

st.title("📰 Deteksi Berita Hoaks (Naive Bayes + LLM)")

# ✅ Load dataset
@st.cache_data
def load_dataset():
    try:
        df1 = pd.read_csv("Data_latih.csv")
        df2 = pd.read_csv("detik_data_.csv")
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return pd.DataFrame(), pd.DataFrame()
    return df1, df2

@st.cache_data
def prepare_data(df1, df2):
    df = load_and_clean_data(df1, df2)
    df = preprocess_dataframe(df)
    df = combine_text_columns(df)
    label_map = {"Hoax": 0, "Non-Hoax": 1, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    return df

@st.cache_data
def extract_features_and_model(df):
    # Transformasi TF-IDF
    X, vectorizer = tfidf_transform(df["T_text"])
    X = X.toarray()  # GaussianNB butuh dense array
    y = df["label"].values

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_naive_bayes(X_train, y_train)
    y_pred = predict_naive_bayes(model, X_test)

    return model, vectorizer, X_test, y_test, y_pred

# ✅ Load Data dan Model
try:
    df1, df2 = load_dataset()
    if df1.empty or df2.empty:
        st.stop()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)
except Exception as e:
    st.error(f"Gagal memuat atau memproses data:\n{e}")
    st.stop()

hasil_semua = []

# ✅ Halaman Split Data
def show_split_data_page(df, vectorizer):
    st.header("📊 Split Data & Distribusi Label")
    label_mapping = {1: "Non-Hoax", 0: "Hoax"}
    df["label_text"] = df["label"].map(label_mapping)

    X = df["T_text"]
    y = df["label_text"]
    X_train, X_test, y_train, y_test = split_data(X, y)

    st.subheader("Distribusi Label - Data Train")
    st.dataframe(y_train.value_counts().reset_index().rename(columns={"index": "Label", "label_text": "Jumlah"}))

    st.subheader("Distribusi Label - Data Test")
    st.dataframe(y_test.value_counts().reset_index().rename(columns={"index": "Label", "label_text": "Jumlah"}))

    st.info(f"Jumlah data latih: {len(X_train)} | Jumlah data uji: {len(X_test)}")

    st.subheader("Total Bobot TF-IDF Data Uji per Label")
    X_test_tfidf = vectorizer.transform(X_test)
    mask_hoax = (y_test == "Hoax").values
    mask_nonhoax = (y_test == "Non-Hoax").values
    tfidf_sum_hoax = X_test_tfidf[mask_hoax].sum()
    tfidf_sum_nonhoax = X_test_tfidf[mask_nonhoax].sum()
    jumlah_fitur_unik = (X_test_tfidf.sum(axis=0) > 0).sum()

    st.write(f"Jumlah fitur unik (kata unik) di data uji: {int(jumlah_fitur_unik)}")
    st.write(f"Total Bobot TF-IDF Data Uji untuk Label 'Hoax'    : {tfidf_sum_hoax:.4f}")
    st.write(f"Total Bobot TF-IDF Data Uji untuk Label 'Non-Hoax': {tfidf_sum_nonhoax:.4f}")

# ✅ Menu Deteksi Hoaks
if selected == "Deteksi Hoaks":
    st.subheader("Masukkan Teks Berita:")
    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...", height=200)

    if st.button("Analisis Berita"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        elif not is_valid_text(user_input):
            st.warning("Masukkan teks yang lengkap dan valid, bukan hanya satu kata atau karakter acak.")
        elif not is_indonesian(user_input):
            st.warning("❌ Teks harus ditulis dalam Bahasa Indonesia.")
        else:
            with st.spinner("Memproses teks dan memprediksi..."):
                processed = preprocess_text(user_input)
                vectorized = vectorizer.transform([processed]).toarray()  # ubah ke dense
                prediction = model.predict(vectorized)[0]
                probas = model.predict_proba(vectorized)[0]
                label_map = {1: "Non-Hoax", 0: "Hoax"}
                pred_label = label_map[prediction]

            st.success(f"Prediksi: **{pred_label}**")

            st.subheader("Keyakinan Model:")
            df_proba = pd.DataFrame({"Label": ["Non-Hoax", "Hoax"], "Probabilitas": probas})
            fig = px.pie(df_proba, names="Label", values="Probabilitas",
                         title="Distribusi Probabilitas Prediksi",
                         color_discrete_sequence=["green", "red"])
            st.plotly_chart(fig, use_container_width=True)

            try:
                result = analyze_with_gemini(
                    text=user_input,
                    predicted_label=pred_label,
                    used_links=[],
                    distribution={"Non-Hoax": f"{probas[1]*100:.1f}",
                                  "Hoax": f"{probas[0]*100:.1f}"}
                )
            except Exception:
                st.warning("LLM gagal dianalisis, hanya menampilkan hasil Naive Bayes.")
                result = {}

            hasil_baru = {
                "Input": user_input,
                "Preprocessed": processed,
                "Prediksi Model": pred_label,
                "Probabilitas Non-Hoax": f"{probas[1]*100:.2f}%",
                "Probabilitas Hoax": f"{probas[0]*100:.2f}%",
                "Kebenaran LLM": result.get("kebenaran"),
                "Alasan LLM": result.get("alasan"),
                "Ringkasan Berita": result.get("ringkasan"),
                "Perbandingan": result.get("perbandingan_kebenaran"),
                "Penjelasan Koreksi": result.get("penjelasan_koreksi")
            }

            simpan_ke_firebase(hasil_baru)
            hasil_semua.append(pd.DataFrame([hasil_baru]))
            st.success("Hasil disimpan ke Firebase Realtime Database")

    if hasil_semua:
        df_hasil = pd.concat(hasil_semua, ignore_index=True)
        csv = df_hasil.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Unduh Hasil (.csv)", data=csv, file_name="hasil_deteksi_berita.csv", mime="text/csv")

elif selected == "Dataset":
    st.subheader("Dataset Kaggle:")
    st.dataframe(df1)
    st.subheader("Dataset Detik.com:")
    st.dataframe(df2)

elif selected == "Split Data":
    show_split_data_page(df, vectorizer)

# ✅ Menu Evaluasi Model
elif selected == "Evaluasi Model":
    st.subheader("Evaluasi Model Naive Bayes")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

    st.subheader("Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, target_names=["Non-Hoax", "Hoax"], zero_division=0)
    st.text(report)

    st.subheader("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Non-Hoax", "Hoax"]
    z = cm
    z_text = [[str(y) for y in x] for x in z]
    fig_cm = ff.create_annotated_heatmap(z, x=labels, y=labels, annotation_text=z_text, colorscale="Blues")
    fig_cm.update_layout(xaxis_title="Prediksi", yaxis_title="Aktual", title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    df_eval = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df_eval["Hasil"] = np.where(df_eval["Actual"] == df_eval["Predicted"], "Benar", "Salah")
    df_eval["T_text"] = df.loc[y_test.index, "T_text"].values
    st.dataframe(df_eval[["T_text", "Actual", "Predicted", "Hasil"]].head())

# ✅ Menu Riwayat Prediksi
elif selected == "Riwayat Prediksi":
    st.subheader("🕒 Riwayat Prediksi")
    df_riwayat = read_predictions_from_firebase()
    if not df_riwayat.empty:
        df_riwayat["timestamp"] = pd.to_datetime(df_riwayat["timestamp"])
        df_riwayat = df_riwayat.sort_values("timestamp", ascending=False).reset_index(drop=True)
        kolom_utama = [
            "Input", "Prediksi Model", "Probabilitas Non-Hoax", "Probabilitas Hoax",
            "Kebenaran LLM", "Alasan LLM", "Ringkasan Berita", "Perbandingan", "Penjelasan Koreksi", "timestamp"
        ]
        tampilkan = [col for col in kolom_utama if col in df_riwayat.columns]
        df_tampil = df_riwayat[tampilkan]
        if not df_tampil.empty:
            df_tampil.insert(0, "No", range(1, len(df_tampil) + 1))
        st.dataframe(df_tampil, use_container_width=True)
        csv_data = df_tampil.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Riwayat (.csv)", data=csv_data, file_name="riwayat_prediksi_firebase.csv", mime="text/csv")
    else:
        st.info("Belum ada data prediksi yang disimpan.")
