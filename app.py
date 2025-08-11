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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from streamlit_option_menu import option_menu
from fpdf import FPDF
import firebase_admin
from firebase_admin import credentials, db

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data, preprocess_with_steps
from feature_extraction import combine_text_columns, tfidf_transform
from classification import split_data, train_naive_bayes, predict_naive_bayes
from interpretation import configure_gemini, analyze_with_gemini

from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0  # agar konsisten hasil
#validasi teks bahsa indonesia
def is_indonesian(text, min_prob=0.90):
    try:
        detections = detect_langs(text)
        for lang in detections:
            if lang.lang == "id" and lang.prob >= min_prob:
                return True
        return False
    except:
        return False
        
st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🔎", layout="wide")

# ✅ Konfigurasi Firebase
firebase_cred = dict(st.secrets["FIREBASE_KEY"])
if not firebase_admin._apps:
    print("Initializing Firebase...")
    firebase_cred = dict(st.secrets["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_cred)
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://deteksi-berita-hoaks-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })
else:
    print("Firebase already initialized.")

def simpan_ke_firebase(data):
    tz = pytz.timezone("Asia/Jakarta")
    waktu_wib = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    data["timestamp"] = waktu_wib
    ref = db.reference("prediksi_hoaks")
    ref.child(str(uuid.uuid4())).set(data)

def read_predictions_from_firebase():
    try:
        ref = db.reference("prediksi_hoaks")
        data = ref.get()
        return pd.DataFrame(data.values()) if data else pd.DataFrame()
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

@st.cache_data
def load_dataset():
    return pd.read_csv("Data_latih.csv"), pd.read_csv("detik_data.csv")

@st.cache_data
def prepare_data(df1, df2):
    df = load_and_clean_data(df1, df2)
    df = preprocess_dataframe(df)
    df = combine_text_columns(df)
    label_map = {"Hoax": 1, "Non-Hoax": 0, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    return df

@st.cache_data
def extract_features_and_model(df):
    X, vectorizer = tfidf_transform(df["T_text"])
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_naive_bayes(X_train, y_train)

    # Predict
    y_pred = predict_naive_bayes(model, X_test)

    return model, vectorizer, X_test, y_test, y_pred
def is_valid_text(text):
    words = re.findall(r'\w+', text)
    return len(words) >= 5 and any(len(word) > 3 for word in words)

# ✅ Load Data dan Model
try:
    df1, df2 = load_dataset()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)
except Exception as e:
    st.error(f"Gagal memuat atau memproses data:\n{e}")
    st.stop()

hasil_semua = []

def show_split_data_page(df):
    st.header("📊 Split Data & Distribusi Label")

    # Mapping label angka ke teks
    label_mapping = {1: "non-hoax", 0: "hoax"}
    df["label_text"] = df["label"].map(label_mapping)

    # Ambil fitur dan label teks
    X = df["T_text"]
    y = df["label_text"]

    # Split data (pakai default 20% test size dari classification.py)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Distribusi label pada data train
    st.subheader("Distribusi Label - Data Train")
    train_dist = y_train.value_counts().reset_index()
    train_dist.columns = ["Label", "Jumlah"]
    st.dataframe(train_dist)

    # Distribusi label pada data test
    st.subheader("Distribusi Label - Data Test")
    test_dist = y_test.value_counts().reset_index()
    test_dist.columns = ["Label", "Jumlah"]
    st.dataframe(test_dist)

    # Ringkasan total
    st.info(f"Jumlah data latih: {len(X_train)} | Jumlah data uji: {len(X_test)}")

# ✅ Menu Deteksi Hoaks
if selected == "Deteksi Hoaks":
    st.subheader("Masukkan Teks Berita:")
    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...", height=200)

# ✅ Menu Deteksi Hoaks
if selected == "Deteksi Hoaks":
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
                vectorized = vectorizer.transform([processed])
                prediction = model.predict(vectorized)[0]
                probas = model.predict_proba(vectorized)[0]
                label_map = {1: "Hoax", 0: "Non-Hoax"}
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
                    distribution={"Non-Hoax": f"{probas[0]*100:.1f}",
                                  "Hoax": f"{probas[1]*100:.1f}"}
                )
                with st.expander("Hasil Interpretasi LLM"):
                    st.write(result.get('output_mentah', 'Tidak tersedia'))

                hasil_baru = {
                    "Input": user_input,
                    "Preprocessed": processed,
                    "Prediksi Model": pred_label,
                    "Probabilitas Non-Hoax": f"{probas[0]*100:.2f}%",
                    "Probabilitas Hoax": f"{probas[1]*100:.2f}%",
                    "Kebenaran LLM": result.get("kebenaran"),
                    "Alasan LLM": result.get("alasan"),
                    "Ringkasan Berita": result.get("ringkasan"),
                    "Perbandingan": result.get("perbandingan_kebenaran"),
                    "Penjelasan Koreksi": result.get("penjelasan_koreksi")
                }

                simpan_ke_firebase(hasil_baru)
                hasil_semua.append(pd.DataFrame([hasil_baru]))
                st.success("Hasil disimpan ke Firebase Realtime Database")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat menggunakan LLM:\n{e}")


if hasil_semua:
    df_hasil = pd.concat(hasil_semua, ignore_index=True)
    csv = df_hasil.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Unduh Hasil (.csv)", data=csv, file_name="hasil_deteksi_berita.csv", mime="text/csv")

    # ✅ Tambahan: Download PDF
    if st.button("⬇️ Unduh Hasil (.pdf)"):
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Hasil Deteksi Berita Hoaks', 0, 1, 'C')
                self.ln(5)
            def chapter_title(self, title):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, title, 0, 1)
            def chapter_body(self, body):
                self.set_font('Arial', '', 11)
                self.multi_cell(0, 8, body)
                self.ln()

        pdf = PDF()
        pdf.add_page()

        for idx, row in df_hasil.iterrows():
            pdf.chapter_title(f"Data #{idx+1}")
            pdf.chapter_body(f"Teks Asli:\n{row['Input']}")
            pdf.chapter_body(f"Preprocessed:\n{row['Preprocessed']}")
            pdf.chapter_body(f"Prediksi Model: {row['Prediksi Model']}")
            pdf.chapter_body(f"Probabilitas Non-Hoax: {row['Probabilitas Non-Hoax']}")
            pdf.chapter_body(f"Probabilitas Hoax: {row['Probabilitas Hoax']}")
            pdf.chapter_body(f"Kebenaran LLM: {row.get('Kebenaran LLM', '-')}")
            pdf.chapter_body(f"Alasan LLM:\n{row.get('Alasan LLM', '-')}")
            pdf.chapter_body(f"Ringkasan Berita:\n{row.get('Ringkasan Berita', '-')}")
            pdf.chapter_body(f"Perbandingan:\n{row.get('Perbandingan', '-')}")
            pdf.chapter_body(f"Penjelasan Koreksi:\n{row.get('Penjelasan Koreksi', '-')}")
            pdf.ln(5)

        pdf_output = pdf.output(dest='S').encode('latin-1')
        st.download_button(
            label="📄 Download PDF",
            data=pdf_output,
            file_name="hasil_deteksi_berita.pdf",
            mime="application/pdf"
        )

# ✅ Menu Dataset
elif selected == "Dataset":
    st.subheader("Dataset Kaggle:")
    st.dataframe(df1)
    st.subheader("Dataset Detik.com:")
    st.dataframe(df2)

elif selected == "Split Data":
    show_split_data_page(df)

# menu preprocessing
elif selected == "Preprocessing":
    st.subheader("🔧 Tahapan Preprocessing Dataset")

    # 1️⃣ Penambahan Atribut Label pada Dataset Detik
    st.markdown("### 1️⃣ Penambahan Atribut Label pada Dataset Detik")
    st.write("Menambahkan atribut label pada dataset detik.com.")

    # Tambahkan label jika belum ada
    if 'label' not in df2.columns:
        df2['label'] = 'Non-Hoax'
        st.success("✅ Kolom `label` berhasil ditambahkan ke dataset Detik.com dengan nilai default 'Non-Hoax'")
    else:
        st.info(" 1 Kolom `label` sudah ada di Detik.com, tidak perlu ditambahkan lagi.")

    # Tampilkan hasilnya
    st.dataframe(df2.head())

    st.markdown("### 2️ Pemilihan Atribut yang Digunakan")
    st.write("Atribut yang dipilih untuk digunakan dalam analisis adalah: `judul`, `narasi`, dan `label`.")

    st.subheader("📄 Dataset Kaggle")
    st.dataframe(df1[["judul", "narasi", "label"]].head())

    st.subheader("📄 Dataset Detik.com")
    st.dataframe(df2[["Judul", "Konten", "label"]].head())

    st.markdown("### Penggabungan Dataset Kaggle + Detik.com")
    st.dataframe(df[["judul", "narasi", "label"]].head(), use_container_width=True)

    st.markdown("### Penambahan Atribut `text` (Gabungan Judul + Narasi)")
    st.dataframe(df[["text"]].head(), use_container_width=True)

    st.markdown("### 🔎 Contoh Proses Lengkap Preprocessing")

    # Ambil contoh teks
    contoh_teks = df["text"].iloc[0]

    # Proses preprocessing bertahap
    hasil = preprocess_with_steps(contoh_teks)

    # Buat list data untuk setiap tahap
    data = [
        {"Tahapan Preprocessing": "Cleansing", "Before": contoh_teks, "After": hasil["cleansing"]},
        {"Tahapan Preprocessing": "Case Folding", "Before": hasil["cleansing"], "After": hasil["case_folding"]},
        {"Tahapan Preprocessing": "Tokenizing", "Before": hasil["case_folding"], "After": hasil["tokenizing"]},
        {"Tahapan Preprocessing": "Stopword Removal", "Before": hasil["tokenizing"], "After": hasil["stopword_removal"]},
        {"Tahapan Preprocessing": "Stemming", "Before": hasil["stopword_removal"], "After": hasil["stemming"]},
        {"Tahapan Preprocessing": "Filter Tokens", "Before": hasil["stemming"], "After": hasil["filtering"]},  # ✅ disesuaikan
        {"Tahapan Preprocessing": "Final Result (T_text)", "Before": hasil["filtering"], "After": hasil["final"]},
    ]

    # Ubah jadi DataFrame
    df_steps = pd.DataFrame(data)

    # Tampilkan tabel
    st.dataframe(df_steps, use_container_width=True)



# ✅ Menu Evaluasi Model
elif selected == "Evaluasi Model":
    st.subheader("Evaluasi Model Naive Bayes")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

    st.subheader("Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, target_names=["Non-Hoax", "Hoax"])
    st.text(report)

    st.subheader("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Non-Hoax", "Hoax"]
    z = cm
    z_text = [[str(y) for y in x] for x in z]

    fig_cm = ff.create_annotated_heatmap(
        z,
        x=labels,
        y=labels,
        annotation_text=z_text,
        colorscale="Blues"
    )

    fig_cm.update_layout(
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        title="Confusion Matrix"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Visualisasi Prediksi:")
    df_eval = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df_eval["Hasil"] = np.where(df_eval["Actual"] == df_eval["Predicted"], "Benar", "Salah")
    hasil_count = df_eval["Hasil"].value_counts().reset_index()
    hasil_count.columns = ["Hasil", "Jumlah"]
    fig_eval = px.pie(hasil_count, names="Hasil", values="Jumlah", title="Distribusi Prediksi Benar vs Salah",
                      color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_eval, use_container_width=True)

    st.subheader("Contoh Data Salah Prediksi:")
    salah = df_eval[df_eval["Hasil"] == "Salah"]
    st.dataframe(salah.head() if not salah.empty else pd.DataFrame([{"Info": "Semua prediksi benar!"}]))

# ✅ Menu Riwayat Prediksi
elif selected == "Riwayat Prediksi":
    st.subheader("🕒 Riwayat Prediksi")

    df_riwayat = read_predictions_from_firebase()
    if not df_riwayat.empty:
        df_riwayat["timestamp"] = pd.to_datetime(df_riwayat["timestamp"])
        df_riwayat = df_riwayat.sort_values("timestamp", ascending=False).reset_index(drop=True)
        df_riwayat.index += 1

        kolom_utama = [
            "Input", "Prediksi Model", "Probabilitas Non-Hoax", "Probabilitas Hoax",
            "Kebenaran LLM", "Alasan LLM", "Ringkasan Berita", "Perbandingan", "Penjelasan Koreksi", "timestamp"
        ]
        tampilkan = [col for col in kolom_utama if col in df_riwayat.columns]
        df_tampil = df_riwayat[tampilkan]
        df_tampil.insert(0, "No", range(1, len(df_tampil) + 1))

        st.dataframe(df_tampil, use_container_width=True)

        csv_data = df_tampil.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Riwayat (.csv)", data=csv_data, file_name="riwayat_prediksi_firebase.csv", mime="text/csv")
    else:
        st.info("Belum ada data prediksi yang disimpan.")

elif selected == "Info Sistem":
    import platform
    import psutil
    import socket
    import cpuinfo

    st.subheader("💻 Informasi Sistem (Streamlit Cloud)")

    # CPUINFO
    cpu = cpuinfo.get_cpu_info()
    st.markdown("### 🧠 CPU Detail")
    st.write("Brand:", cpu.get("brand_raw"))
    st.write("Architecture:", cpu.get("arch"))
    st.write("Bits:", cpu.get("bits"))
    st.write("Cores (logical):", psutil.cpu_count(logical=True))
    st.write("Cores (physical):", psutil.cpu_count(logical=False))

    # Frekuensi CPU
    freq = psutil.cpu_freq()
    if freq:
        st.write(f"Frequency: {freq.current:.2f} MHz / Max: {freq.max:.2f} MHz")

    # MEMORY
    st.markdown("### 💾 RAM")
    vm = psutil.virtual_memory()
    st.write(f"Total RAM: {vm.total/1024**3:.2f} GB")
    st.write(f"Available: {vm.available/1024**3:.2f} GB")
    st.write(f"Usage: {vm.percent}%")

    # DISK
    st.markdown("### 💽 Disk")
    disk = psutil.disk_usage('/')
    st.write(f"Total Disk: {disk.total/1024**3:.2f} GB")
    st.write(f"Free Disk: {disk.free/1024**3:.2f} GB")
    st.write(f"Usage: {disk.percent}%")

    # GPU (Nvidia) – optional safe import
    st.markdown("### ⚙️ GPU Nvidia (jika ada)")
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                st.write(f"Name: {gpu.name}")
                st.write(f"Load: {gpu.load*100:.1f}%")
                st.write(f"Memory Free: {gpu.memoryFree}MB")
                st.write(f"Memory Total: {gpu.memoryTotal}MB")
        else:
            st.write("GPU Nvidia tidak terdeteksi.")
    except ImportError:
        st.write("Library GPUtil tidak tersedia.")

    # GPU (AMD) – optional safe import
    st.markdown("### 🔧 GPU AMD (jika ada)")
    try:
        import pyamdgpuinfo
        amd_info = pyamdgpuinfo.get_gpu_information()
        if amd_info:
            st.json(amd_info)
        else:
            st.write("GPU AMD tidak terdeteksi.")
    except ImportError:
        st.write("Library pyamdgpuinfo tidak tersedia.")
    except Exception as e:
        st.write("Tidak dapat membaca GPU AMD:", e)

    # OS & Python
    st.markdown("### 🛠️ Sistem Operasi & Python")
    st.write("OS:", platform.platform())
    st.write("Python:", platform.python_version())

    # Jaringan
    st.markdown("### 🌐 Jaringan")
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        st.write("Hostname:", hostname)
        st.write("IP:", ip)
    except:
        st.write("Tidak dapat mengambil informasi jaringan.")




