import google.generativeai as genai
import re

def configure_gemini(api_key):
    """
    Mengatur API key untuk Google Gemini.
    """
    genai.configure(api_key=api_key)

def analyze_with_gemini(text, predicted_label, used_links=None, distribution=None):
    """
    Menganalisis teks berita menggunakan LLM berdasarkan hasil prediksi model Naive Bayes
    dengan prompt yang disesuaikan agar sesuai instruksi user.
    """

    distribusi_str = ""
    if distribution:
        distribusi_str = "\nDistribusi Prediksi Model (dalam persen):\n"
        distribusi_str += "\n".join([f"- {label}: {percentage}%" for label, percentage in distribution.items()])

    # ✅ Prompt sesuai instruksi user
    prompt = f"""
From now on, you are an expert in misinformation detection and content analysis, tasked to classify Indonesian news articles and explain the reasoning.
You will receive a news text in Indonesian. Your task is to determine if the news is Hoax or Non-Hoax and provide reasoning in Indonesian:
1. Answer in one word with either ‘Hoax’ or ‘Non-Hoax’, then explain why in 2-3 sentences.
2. Write a concise summary of the news in Indonesian with maximum 5 sentences covering the key points.

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
        # Ambil label Hoax/Non-Hoax (cari kata pertama Hoax atau Non-Hoax)
        kebenaran_match = re.search(r"\b(Hoax|Non[- ]?Hoax)\b", response_text, re.IGNORECASE)
        if kebenaran_match:
            kebenaran_val = kebenaran_match.group(1).strip().replace("-", " ")

        # Ambil alasan (kalimat setelah label sampai sebelum ringkasan)
        # Asumsi: jawaban 1 dan 2 dipisah baris atau paragraf
        alasan_match = re.search(r"(?:Hoax|Non[- ]?Hoax)\s*(.*?)((?:Ringkasan|$))", response_text, re.DOTALL | re.IGNORECASE)
        if alasan_match:
            alasan_raw = alasan_match.group(1).strip()
            # Bersihkan agar hanya 2-3 kalimat alasan (bukan ringkasan)
            alasan_sentences = re.split(r'(?<=[.!?])\s+', alasan_raw)
            alasan_val = " ".join(alasan_sentences[:3]).strip()

        # Ambil ringkasan (baris terakhir atau setelah kata kunci 'Ringkasan' jika ada)
        summary_candidates = re.split(r'\n+', response_text)
        if len(summary_candidates) > 1:
            # Ambil paragraf terakhir sebagai ringkasan
            ringkasan_val = summary_candidates[-1].strip()
            # Batasi maksimal 5 kalimat
            ringkasan_sentences = re.split(r'(?<=[.!?])\s+', ringkasan_val)
            ringkasan_val = " ".join(ringkasan_sentences[:5]).strip()

    except Exception as e:
        alasan_val = f"Gagal memproses respons LLM: {e}"

    # Bandingkan hasil prediksi dengan interpretasi LLM
    pred_label_clean = predicted_label.strip().lower().replace("-", " ") if predicted_label else ""
    llm_label_clean = kebenaran_val.lower() if kebenaran_val else ""
    perbandingan = "sesuai" if pred_label_clean == llm_label_clean else "berbeda"

    # Penjelasan koreksi jika berbeda
    penjelasan_koreksi = None
    if perbandingan == "berbeda":
        penjelasan_koreksi = (
            f"Model otomatis memprediksi bahwa berita ini adalah **{predicted_label}**, "
            f"namun hasil analisis oleh LLM menyatakan bahwa berita ini termasuk **{kebenaran_val}**.\n\n"
            f"Perbedaan ini mungkin terjadi karena model Naive Bayes hanya menganalisis pola kata, "
            f"sedangkan LLM memahami konteks semantik teks secara menyeluruh.\n\n"
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
