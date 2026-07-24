# Klasifikasi Citra Asli vs AI-Generated (Streamlit + GitHub + Hugging Face)

Aplikasi web untuk mengklasifikasi gambar sebagai **Asli** atau **AI-Generated**,
menggunakan 3 model pilihan: EfficientNet, Xception, ResNet50.

Karena model `.keras` kamu berukuran besar, model disimpan di **Hugging Face Hub**
dan otomatis didownload saat aplikasi pertama kali dijalankan.

## LANGKAH 1 — Upload model ke Hugging Face Hub (sudah selesai ✅)

Repo: `desif5943-blip/klasifikasi-ai-asli`

## LANGKAH 2 — Edit konfigurasi di `app.py` (sudah selesai ✅)

`HF_REPO_ID` sudah diisi sesuai repo Hugging Face di atas.

## LANGKAH 3 — Push ke GitHub (dijalankan di TERMINAL LAPTOP)

1. Buat repo baru di GitHub (public atau private, keduanya bisa)
2. Di folder project ini, jalankan:
```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Initial commit: Streamlit app deteksi citra AI"
   git branch -M main
   git remote add origin https://github.com/USERNAME/NAMA_REPO_GITHUB.git
   git push -u origin main
```

> File model **tidak perlu** ikut di-push ke GitHub — biarkan saja di Hugging Face Hub.

## LANGKAH 4 — Deploy ke Streamlit Community Cloud

1. Buka https://share.streamlit.io dan login dengan akun GitHub kamu
2. Klik **New app**
3. Pilih repo GitHub yang baru kamu push
4. Main file path: `app.py`
5. Klik **Deploy**

## Catatan teknis penting

- **Preprocessing**: kode mengasumsikan `preprocess_input` standar tiap arsitektur + resize 224x224.
- **Output model**: kode menangani output 1 neuron (sigmoid) maupun 2 neuron (softmax).
- **RAM gratis Streamlit Cloud terbatas (~1GB)**, tapi app hanya load 1 model sesuai pilihan user, jadi aman.
