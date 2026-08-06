import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import pandas as pd

# ============================================================
# KONFIGURASI
# ============================================================
HF_REPO_ID = "desif5943-blip/klasifikasi-ai-asli"

# Nilai accuracy / loss / precision / recall / f1_score di bawah ini
# diambil dari hasil evaluasi TAHAP 2 (fine-tuning) pada DATA UJI
# (test set, 2000 gambar) sesuai log training masing-masing model.
# precision / recall / f1_score menggunakan nilai "macro avg" dari
# classification_report.
MODELS_INFO = {
    "EfficientNetB0": {
        "filename": "best_efficientnet_gradual.keras",
        "input_size": (224, 224),
        "preprocess_fn": tf.keras.applications.efficientnet.preprocess_input,
        "accuracy":  0.8395,
        "loss":      0.4705,
        "precision": 0.84,
        "recall":    0.84,
        "f1_score":  0.84,
    },
    "Xception": {
        "filename": "best_xception_gradual.keras",
        "input_size": (224, 224),
        "preprocess_fn": tf.keras.applications.xception.preprocess_input,
        "accuracy":  0.9110,
        "loss":      0.2583,
        "precision": 0.91,
        "recall":    0.91,
        "f1_score":  0.91,
    },
    "ResNet50": {
        "filename": "best_resnet50_gradual.keras",
        "input_size": (224, 224),
        "preprocess_fn": tf.keras.applications.resnet50.preprocess_input,
        "accuracy":  0.8715,
        "loss":      0.3486,
        "precision": 0.87,
        "recall":    0.87,
        "f1_score":  0.87,
    },
}

CLASS_NAMES = ["Citra Asli", "Citra Generatif AI"]

st.set_page_config(
    page_title="Klasifikasi Citra Asli vs Citra Generatif AI",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD MODEL (di-cache, hanya download sekali)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    info = MODELS_INFO[model_key]
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=info["filename"]
    )
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": info["preprocess_fn"]},
        safe_mode=False
    )
    return model

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict(model, image: Image.Image, input_size):
    img = image.convert("RGB").resize(input_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    return pred

def parse_pred(pred):
    if pred.shape[0] == 1:
        prob_ai   = float(pred[0])
        prob_asli = 1 - prob_ai
    else:
        prob_asli = float(pred[0])
        prob_ai   = float(pred[1])
    label      = CLASS_NAMES[0] if prob_asli > prob_ai else CLASS_NAMES[1]
    confidence = max(prob_asli, prob_ai) * 100
    return label, confidence, prob_asli, prob_ai

# ============================================================
# UI — HEADER
# ============================================================
st.title("🤖 Klasifikasi Citra Asli vs Citra Generatif AI")
st.write(
    "Upload gambar lalu klik **Klasifikasi Semua Model** — "
    "ketiga model akan berjalan sekaligus dan hasilnya "
    "ditampilkan berdampingan untuk perbandingan."
)
st.divider()

uploaded_file = st.file_uploader(
    "Upload gambar (jpg/jpeg/png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Gambar yang diupload",
                 use_container_width=True)
    with col_info:
        st.info(
            "Klik tombol di bawah untuk menjalankan ketiga model sekaligus.\n\n"
            "Model yang pertama kali digunakan akan didownload dari "
            "Hugging Face Hub dan disimpan di cache — "
            "proses ini hanya terjadi satu kali per sesi."
        )

    st.divider()

    if st.button("🔍 Klasifikasi Semua Model",
                 type="primary", use_container_width=True):

        # ── Jalankan ketiga model ──────────────────────────────
        hasil = {}
        progress_bar = st.progress(0, text="Mempersiapkan model...")

        for idx, model_key in enumerate(MODELS_INFO):
            progress_bar.progress(
                (idx) / len(MODELS_INFO),
                text=f"Memuat & menjalankan {model_key}..."
            )
            model = load_model(model_key)
            pred  = predict(
                model, image, MODELS_INFO[model_key]["input_size"]
            )
            label, confidence, prob_asli, prob_ai = parse_pred(pred)
            hasil[model_key] = {
                "label":      label,
                "confidence": confidence,
                "prob_asli":  prob_asli,
                "prob_ai":    prob_ai,
            }

        progress_bar.progress(1.0, text="Selesai!")

        # ── Hasil ketiga model berdampingan ───────────────────
        st.subheader("📊 Hasil Klasifikasi Ketiga Model")
        col1, col2, col3 = st.columns(3)

        for col, model_key in zip([col1, col2, col3], MODELS_INFO):
            info = MODELS_INFO[model_key]
            h    = hasil[model_key]

            with col:
                st.markdown(f"### {model_key}")
                st.divider()

                warna  = "🟢" if h["label"] == CLASS_NAMES[0] else "🔴"
                st.markdown(f"**Hasil: {warna} {h['label']}**")
                st.metric("Tingkat Keyakinan", f"{h['confidence']:.2f}%")

                st.write("**Probabilitas per Kelas:**")
                st.write("Citra Asli")
                st.progress(h["prob_asli"])
                st.caption(f"{h['prob_asli'] * 100:.2f}%")

                st.write("Citra Generatif AI")
                st.progress(h["prob_ai"])
                st.caption(f"{h['prob_ai'] * 100:.2f}%")

                st.divider()
                st.write("**Kinerja Model (Data Uji):**")
                st.metric("Akurasi",   f"{info['accuracy']  * 100:.2f}%")
                st.metric("Loss",      f"{info['loss']:.4f}")
                st.metric("Precision", f"{info['precision'] * 100:.2f}%")
                st.metric("Recall",    f"{info['recall']    * 100:.2f}%")
                st.metric("F1-Score",  f"{info['f1_score']  * 100:.2f}%")
                st.caption(
                    f"Nilai kinerja dari hasil evaluasi "
                    f"model {model_key} pada data uji."
                )

        # ── Tabel ringkasan perbandingan ──────────────────────
        st.divider()
        st.subheader("📋 Tabel Ringkasan Perbandingan Ketiga Model")

        rows = []
        for model_key in MODELS_INFO:
            info = MODELS_INFO[model_key]
            h    = hasil[model_key]
            rows.append({
                "Model":              model_key,
                "Hasil Klasifikasi":  h["label"],
                "Keyakinan (%)":      f"{h['confidence']:.2f}",
                "Prob. Asli (%)":     f"{h['prob_asli']  * 100:.2f}",
                "Prob. AI (%)":       f"{h['prob_ai']    * 100:.2f}",
                "Akurasi (%)":        f"{info['accuracy']  * 100:.2f}",
                "Loss":               f"{info['loss']:.4f}",
                "Precision (%)":      f"{info['precision'] * 100:.2f}",
                "Recall (%)":         f"{info['recall']    * 100:.2f}",
                "F1-Score (%)":       f"{info['f1_score']  * 100:.2f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Kesimpulan otomatis ───────────────────────────────
        st.divider()
        st.subheader("💡 Kesimpulan Perbandingan")

        model_terbaik = max(
            MODELS_INFO,
            key=lambda k: MODELS_INFO[k]["accuracy"]
        )
        label_mayoritas = max(
            set(h["label"] for h in hasil.values()),
            key=lambda l: sum(
                1 for h in hasil.values() if h["label"] == l
            )
        )
        jumlah_setuju = sum(
            1 for h in hasil.values()
            if h["label"] == label_mayoritas
        )

        st.success(
            f"**{jumlah_setuju} dari 3 model** sepakat bahwa gambar ini "
            f"adalah **{label_mayoritas}**.\n\n"
            f"Model dengan akurasi tertinggi pada data uji adalah "
            f"**{model_terbaik}** "
            f"({MODELS_INFO[model_terbaik]['accuracy']*100:.2f}%)."
        )

st.divider()
st.caption(
    "Model: EfficientNetB0, Xception, ResNet50 — "
    "Transfer learning dengan teknik gradual unfreezing (2 tahap: "
    "classifier-only lalu fine-tuning bertahap). | "
    "Repositori model: desif5943-blip/klasifikasi-ai-asli (Hugging Face Hub)"
)
