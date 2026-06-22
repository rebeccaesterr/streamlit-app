import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

import os
# --- PENTING: Tambahkan baris ini SEBELUM import tensorflow ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# --- KONFIGURASI ---
MODEL_PATHS = {
    "Inception V3": "models/model_inceptionv3.h5",
    "ResNet50": "models/model_resnet50.h5",
    "Custom CNN": "models/model_custom (1).h5" 
}

# Pastikan nama file di sini sesuai dengan yang ada di folder 'graph' kamu
GRAPH_PATHS = {
    "Inception V3": ["graph/inception_acc.png", "graph/inception_loss.png"],
    "ResNet50": ["graph/resnet_acc.png", "graph/resnet_loss.png"],
    "Custom CNN": ["graph/custom_acc.png", "graph/custom_loss.png"]
}

CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Stem End Rot', 'Healthy']

# --- FUNGSI LOAD MODEL (PERBAIKAN UTAMA) ---
@st.cache_resource
def load_model(model_name):
    model_path = MODEL_PATHS[model_name]
    try:
        # PENTING: Tambahkan compile=False untuk menghindari error "2 input tensors"
        # Ini memberitahu Keras untuk memuat arsitektur & bobot saja, tanpa konfigurasi optimizer
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

# --- FUNGSI PREPROCESSING (PERBAIKAN SIZE) ---
def preprocess_image(image, model_name):
    # 1. Resize sesuai arsitektur
    if model_name == "Inception V3":
        # PERBAIKAN: InceptionV3 WAJIB 299x299, bukan 224x224
        target_size = (299, 299)
    else:
        # ResNet50 dan Custom Model tetap 224x224
        target_size = (224, 224)
    
    # Pastikan resize menggunakan method yang benar dari PIL
    image = image.resize(target_size)
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch

    # 2. Normalisasi / Preprocessing spesifik
    if model_name == "Inception V3":
        img_array = preprocess_inception(img_array)
    elif model_name == "ResNet50":
        img_array = preprocess_resnet(img_array)
    else:
        # Custom CNN: Rescale 1./255
        img_array = img_array / 255.0
        
    return img_array

# --- UI STREAMLIT ---
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

st.title("Klasifikasi Penyakit Mangga")
st.markdown("Dashboard untuk membandingkan performa model dan melakukan prediksi penyakit.")

# Sidebar
st.sidebar.header("Konfigurasi")
selected_model_name = st.sidebar.selectbox("Pilih Model AI:", list(MODEL_PATHS.keys()))

# Bagian Atas: Grafik
st.header(f"📊 Performa Training: {selected_model_name}")
col_graph1, col_graph2 = st.columns(2)

graphs = GRAPH_PATHS[selected_model_name]

try:
    with col_graph1:
        st.image(graphs[0], caption="Grafik Akurasi", use_container_width=True)
    with col_graph2:
        st.image(graphs[1], caption="Grafik Loss", use_container_width=True)
except Exception:
    st.warning("File grafik belum direname/ditemukan di folder 'graph'.")

st.divider()

# Bagian Bawah: Prediksi
st.header("🔍 Uji Coba Prediksi")
st.write(f"Model aktif: **{selected_model_name}**")

uploaded_file = st.file_uploader("Upload gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col_img, col_res = st.columns([1, 2])
    
    image = Image.open(uploaded_file)
    with col_img:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    with col_res:
        if st.button("Mulai Deteksi"):
            with st.spinner('Sedang memproses...'):
                model = load_model(selected_model_name)
                
                if model:
                    processed_img = preprocess_image(image, selected_model_name)
                    
                    # Predict
                    predictions = model.predict(processed_img)
                    
                    pred_class_idx = np.argmax(predictions[0])
                    pred_label = CLASS_NAMES[pred_class_idx]
                    confidence = np.max(predictions[0]) * 100

                    # Tampilkan Hasil
                    st.success(f"Hasil Prediksi: **{pred_label}**")
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))
                    
                    with st.expander("Lihat detail probabilitas"):
                        st.write(dict(zip(CLASS_NAMES, predictions[0].astype(float))))