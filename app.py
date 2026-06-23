import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ============================================================
# KONFIGURASI - GANTI INI sesuai repo Hugging Face kamu
# ============================================================
HF_REPO_ID = "desif5943-blip/klasifikasi-ai-asli"

MODELS_INFO = {
    "EfficientNet (paling ringan)": {
        "filename": "best_efficientnet_gradual.keras",
        "input_size": (224, 224),
        "preprocess_input": None,  # tidak pakai Lambda preprocess_input
        "accuracy": 0.96,
        "loss": 0.0246,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.96,
    },
    "Xception": {
        "filename": "best_xception_gradual.keras",
        "input_size": (224, 224),
        "preprocess_input": tf.keras.applications.xception.preprocess_input,
        "accuracy": 0.96,
        "loss": 0.1611,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.96,
    },
    "ResNet50": {
        "filename": "best_resnet50_gradual.keras",
        "input_size": (224, 224),
        "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
        "accuracy": 0.97,
        "loss": 0.1317,
        "precision": 0.97,
        "recall": 0.97,
        "f1_score": 0.97,
    },
}

CLASS_NAMES = ["Asli", "AI-Generated"]

st.set_page_config(page_title="Klasifikasi Citra Asli vs Citra Generatif AI", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Download model dari Hugging Face Hub (sekali saja, lalu di-cache) dan load."""
    info = MODELS_INFO[model_key]
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])

    custom_objects = None
    if info["preprocess_input"] is not None:
        custom_objects = {"preprocess_input": info["preprocess_input"]}

    model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects, safe_mode=False
    )
    return model


def predict(model, image: Image.Image, input_size):
    img = image.convert("RGB").resize(input_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    return pred


# ============================================================
# UI
# ============================================================
st.title("🖼️ Klasifikasi Citra Asli vs Citra Generatif AI")
st.write("Upload gambar, pilih model, lalu klik **Klasifikasi** untuk melihat hasil klasifikasi.")

model_key = st.selectbox("Pilih model", list(MODELS_INFO.keys()))

uploaded_file = st.file_uploader("Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Klasifikasi", type="primary"):
        info = MODELS_INFO[model_key]
        with st.spinner(f"Memuat model {model_key} dan memproses gambar..."):
            model = load_model(model_key)
            pred = predict(model, image, info["input_size"])

        if pred.shape[0] == 1:
            prob_ai = float(pred[0])
            prob_asli = 1 - prob_ai
        else:
            prob_asli, prob_ai = float(pred[0]), float(pred[1])

        label = CLASS_NAMES[0] if prob_asli > prob_ai else CLASS_NAMES[1]
        confidence = max(prob_asli, prob_ai) * 100

        st.divider()
        st.subheader(f"Hasil: **{label}**")
        st.write(f"Tingkat keyakinan: **{confidence:.2f}%**")

        st.write("Asli")
        st.progress(prob_asli)
        st.caption(f"{prob_asli * 100:.2f}%")

        st.write("AI-Generated")
        st.progress(prob_ai)
        st.caption(f"{prob_ai * 100:.2f}%")

        st.divider()
        st.subheader("Kinerja Model")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Akurasi Model", f"{info['accuracy'] * 100:.0f}%")
        with col2:
            st.metric("Loss Model", f"{info['loss']:.4f}")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Precision", f"{info['precision'] * 100:.0f}%")
        with col4:
            st.metric("Recall", f"{info['recall'] * 100:.0f}%")
        with col5:
            st.metric("F1-Score", f"{info['f1_score'] * 100:.0f}%")

        st.caption(f"Nilai-nilai di atas didapat dari hasil evaluasi model {model_key} pada data uji.")

st.divider()
st.caption("Model: EfficientNet, Xception, ResNet50 — hasil transfer learning (gradual unfreezing).")
