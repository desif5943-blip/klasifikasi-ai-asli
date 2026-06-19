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
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "input_size": (224, 224),
    },
    "Xception": {
        "filename": "best_xception_gradual.keras",
        "preprocess": tf.keras.applications.xception.preprocess_input,
        "input_size": (224, 224),
    },
    "ResNet50": {
        "filename": "best_resnet50_gradual.keras",
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "input_size": (224, 224),
    },
}

CLASS_NAMES = ["Asli", "AI-Generated"]

st.set_page_config(page_title="Deteksi Citra Asli vs AI", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Download model dari Hugging Face Hub (sekali saja, lalu di-cache) dan load."""
    info = MODELS_INFO[model_key]
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=info["filename"])
    model = tf.keras.models.load_model(model_path)
    return model


def predict(model, image: Image.Image, preprocess_fn, input_size):
    img = image.convert("RGB").resize(input_size)
    arr = np.array(img).astype("float32")
    arr = preprocess_fn(arr)
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    return pred


# ============================================================
# UI
# ============================================================
st.title("🖼️ Deteksi Citra Asli vs AI-Generated")
st.write("Upload gambar, pilih model, lalu klik **Klasifikasi** untuk melihat hasil prediksi.")

model_key = st.selectbox("Pilih model", list(MODELS_INFO.keys()))

uploaded_file = st.file_uploader("Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Klasifikasi", type="primary"):
        info = MODELS_INFO[model_key]
        with st.spinner(f"Memuat model {model_key} dan memproses gambar..."):
            model = load_model(model_key)
            pred = predict(model, image, info["preprocess"], info["input_size"])

        # Tangani output sigmoid (1 neuron) atau softmax (2 neuron)
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
st.caption("Model: EfficientNet, Xception, ResNet50 — hasil transfer learning (gradual unfreezing).")