import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ============================================================
# KONFIGURASI — sesuaikan dengan repo Hugging Face kamu
# ============================================================
HF_REPO_ID = "USERNAME_HF/NAMA_REPO_HF"  # ganti ini

MODELS_INFO = {
    "EfficientNetB0": {
        "filename": "best_efficientnet_gradual.keras",
        "input_size": (224, 224),
        "preprocess_fn": tf.keras.applications.efficientnet.preprocess_input,
        "accuracy": 0.00,   # ganti dengan hasil test accuracy kamu
        "loss": 0.0000,     # ganti dengan hasil test loss kamu
        "precision": 0.00,
        "recall": 0.00,
        "f1_score": 0.00,
    },
    "Xception": {
        "filename": "best_xception_gradual.keras",
        "input_size": (224, 224),
        "preprocess_fn": tf.keras.applications.xception.preprocess_input,
        "accuracy": 0.00,
        "loss": 0.0000,
        "precision": 0.00,
        "recall": 0.00,
        "f1_score": 0.00,
    },
    "ResNet50": {
        "filename": "best_resnet50_gradual.keras",
        "input_size": (224, 224),
        "preprocess_fn": tf.keras.applications.resnet50.preprocess_input,
        "accuracy": 0.00,
        "loss": 0.0000,
        "precision": 0.00,
        "recall": 0.00,
        "f1_score": 0.00,
    },
}

CLASS_NAMES = ["Citra asli", "Citra Generatif AI"]

st.set_page_config(
    page_title="Klasifikasi Citra Asli vs Citra Generatif AI",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# LOAD MODEL (didownload dari Hugging Face, di-cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    info = MODELS_INFO[model_key]
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=info["filename"]
    )
    # custom_objects wajib karena model punya Lambda(preprocess_input) di dalamnya
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
    # Model sudah menangani preprocessing sendiri via Lambda layer
    # cukup kirim citra mentah (nilai piksel 0-255)
    img = image.convert("RGB").resize(input_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    return pred


# ============================================================
# UI
# ============================================================
st.title("🖼️ Klasifikasi Citra Asli vs Citra Generatif AI")
st.write(
    "Upload gambar, pilih model, lalu klik **Klasifikasi** "
    "untuk melihat hasil klasifikasi."
)

model_key = st.selectbox("Pilih model", list(MODELS_INFO.keys()))

uploaded_file = st.file_uploader(
    "Upload gambar (jpg/jpeg/png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Klasifikasi", type="primary"):
        info = MODELS_INFO[model_key]

        with st.spinner(f"Memuat model {model_key} dan memproses gambar..."):
            model = load_model(model_key)
            pred  = predict(model, image, info["input_size"])

        # Tangani output softmax (2 neuron) atau sigmoid (1 neuron)
        if pred.shape[0] == 1:
            prob_ai   = float(pred[0])
            prob_asli = 1 - prob_ai
        else:
            prob_asli = float(pred[0])
            prob_ai   = float(pred[1])

        label      = CLASS_NAMES[0] if prob_asli > prob_ai else CLASS_NAMES[1]
        confidence = max(prob_asli, prob_ai) * 100

        st.divider()
        st.subheader(f"Hasil: **{label}**")
        st.write(f"Tingkat keyakinan: **{confidence:.2f}%**")

        st.write("Citra Asli")
        st.progress(prob_asli)
        st.caption(f"{prob_asli * 100:.2f}%")

        st.write("Citra Generatif AI")
        st.progress(prob_ai)
        st.caption(f"{prob_ai * 100:.2f}%")

        # ── Kinerja Model ──────────────────────────────────────
        st.divider()
        st.subheader("Kinerja Model")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Akurasi Model", f"{info['accuracy'] * 100:.2f}%")
        with col2:
            st.metric("Loss Model", f"{info['loss']:.4f}")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Precision", f"{info['precision'] * 100:.2f}%")
        with col4:
            st.metric("Recall", f"{info['recall'] * 100:.2f}%")
        with col5:
            st.metric("F1-Score", f"{info['f1_score'] * 100:.2f}%")

        st.caption(
            f"Nilai di atas merupakan hasil evaluasi model {model_key} "
            "pada data uji."
        )

st.divider()
st.caption(
    "Model: EfficientNetB0, Xception, ResNet50 — "
    "Transfer learning dengan teknik gradual unfreezing."
)
