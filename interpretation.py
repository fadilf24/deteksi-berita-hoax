import google.generativeai as genai
import re

def configure_gemini(api_key):
    """
    Mengatur API key untuk Google Gemini.
    """
    genai.configure(api_key=api_key)

def analyze_with_gemini(text, predicted_label, used_links=None, distribution=None):
    """
    Menganalisis teks berita menggunakan LLM berdasarkan hasil prediksi model Naive Bayes.
    Mengembalikan hasil interpretasi, serta menjelaskan jika hasil LLM berbeda dengan prediksi model.
    """

    distribusi_str = ""
    if distribution:
        distribusi_str = "\nDistribusi Prediksi Model (dalam persen):\n"
        distribusi_str += "\n".join([f"- {label}: {percentage}%" for label, percentage in distribution.items()])

    prompt = f"""
Teks berikut adalah sebuah berita.

Tugas Anda:
1. Tentukan apakah berita ini termasuk 'Hoax' atau 'Non-Hoax'. Jawaban diawali dengan: "Kebenaran: ..."
2. Jelaskan secara singkat mengapa Anda menilai demikian. Jika Anda menggunakan referensi atau sumber, sebutkan secara umum saja tanpa mencantumkan link. Jawaban diawali dengan: "Alasan: ..."
3. Buat ringkasan isi berita maksimal dalam 5 kalimat. Jawaban diawali dengan: "Ringkasan: ..."

Prediksi model Naive Bayes untuk berita ini: {predicted_label}
{distribusi_str}

Teks Berita:
{text}
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    response_text = response.text.strip()

    # Inisialisasi hasil
    kebenaran_val = None
    alasan_val = None
    ringkasan_val = None

    try:
        kebenaran_match = re.search(r"Kebenaran:\s*(Hoax|Non[- ]?Hoax)", response_text, re.IGNORECASE)
        if kebenaran_match:
            kebenaran_val = kebenaran_match.group(1).strip().replace("-", " ")

        alasan_match = re.search(r"Alasan:\s*(.*?)(?:\n(?:Ringkasan|$))", response_text, re.DOTALL | re.IGNORECASE)
        if alasan_match:
            alasan_val = alasan_match.group(1).strip()

        ringkasan_match = re.search(r"Ringkasan:\s*(.*?)(?:\n|$)", response_text, re.DOTALL | re.IGNORECASE)
        if ringkasan_match:
            ringkasan_val = ringkasan_match.group(1).strip()

    except Exception as e:
        alasan_val = f"Gagal memproses respons LLM: {e}"

    # Bandingkan hasil prediksi dengan interpretasi LLM
    pred_label_clean = predicted_label.strip().lower().replace("-", " ") if predicted_label else ""
    llm_label_clean = kebenaran_val.lower() if kebenaran_val else ""
    perbandingan = "sesuai" if pred_label_clean == llm_label_clean else "berbeda"

    # Penjelasan koreksi versi ramah pengguna
    penjelasan_koreksi = None
    if perbandingan == "berbeda":
        penjelasan_koreksi = (
            f"Model otomatis memprediksi bahwa berita ini adalah **{predicted_label}**, "
            f"namun hasil analisis oleh kecerdasan buatan (LLM) menyatakan bahwa berita ini termasuk **{kebenaran_val}**.\n\n"
            f"Perbedaan ini mungkin terjadi karena model otomatis seperti Naive Bayes hanya melihat pola kata dan frekuensi kata dalam teks, "
            f"tanpa memahami arti atau isi cerita secara menyeluruh.\n\n"
            f"Sebaliknya, model LLM (Large Language Model) seperti Gemini dapat membaca teks dan memahami maksud dari isi berita, "
            f"termasuk apakah informasi tersebut terdengar masuk akal atau justru mencurigakan. Dengan kemampuan itu, LLM bisa memberikan analisis yang lebih mendekati pemahaman manusia.\n\n"
            f"**Alasan dari LLM:** {alasan_val or 'Tidak tersedia'}"
        )

    return {
        "kebenaran": kebenaran_val,
        "alasan": alasan_val,
        "ringkasan": ringkasan_val,
        "output_mentah": response_text,
        "perbandingan_kebenaran": perbandingan,
        "penjelasan_koreksi": penjelasan_koreksi
    }




