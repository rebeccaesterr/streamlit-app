import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
import json



# --- KONFIGURASI ---
MODEL_PATHS = {
    "Inception V3": "models/inc_savedmodel.keras",
    "ResNet50": "models/model_resnet50_fixed.keras",
    "Custom CNN": "models/model_custom (1).h5"
}
GRAPH_PATHS = {
    "Inception V3": ["graph/inc acc.png", "graph/inc loss.png"],
    "ResNet50": ["graph/resnet acc.png", "graph/resnet loss.png"],
    "Custom CNN": ["graph/custom model acc.png", "graph/custom model loss.png"]
}

CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Stem End Rot', 'Healthy']


class PatchedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            shape_val = kwargs.pop('batch_shape')
            
            if isinstance(shape_val, str):
                try:
                    shape_val = json.loads(shape_val.replace("null", "None"))
                except:
                    pass
            
            # Simpan ke variabel yang dimengerti Keras 2
            kwargs['batch_input_shape'] = shape_val

        # 3. Bersihkan argumen lain yang mungkin bikin error
        # Keras 3 kadang menyelipkan argumen 'dtype' sebagai string yang bikin bingung
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], str):
             # Pastikan dtype standar
             if kwargs['dtype'] not in ['float32', 'float64', 'int32']:
                 kwargs.pop('dtype') # Buang jika aneh-aneh

        super().__init__(**kwargs)

# PATCH 2: DTypePolicy (Solusi yang sudah terbukti berhasil di tahap sebelumnya)
class PatchedDTypePolicy:
    def __init__(self, **kwargs):
        pass 
    
    @classmethod
    def from_config(cls, config):
        # Kembalikan objek Policy resmi agar atribut .name tersedia
        return tf.keras.mixed_precision.Policy("float32")
    
    def get_config(self):
        return {"name": "float32"}

# PATCH 3: Conv2D (Jaga-jaga jika error pindah ke layer konvolusi)
class PatchedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, **kwargs):
        # Buang argumen Keras 3 yang tidak ada di Keras 2
        # Contoh: 'groups' default 1, tapi kalau ada di config bisa bikin masalah di versi lama
        if 'groups' in kwargs and kwargs['groups'] == 1:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# ==========================================

@st.cache_resource
def load_model(model_name):
    model_path = MODEL_PATHS[model_name]
    
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}")
        return None

    try:
        # Load dengan custom objects yang sudah diperkuat
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects={
                'InputLayer': PatchedInputLayer,
                'DTypePolicy': PatchedDTypePolicy,
                'Conv2D': PatchedConv2D, # Tambahkan ini untuk keamanan ekstra
                # Tambahkan layer umum lain jika perlu, tapi biasanya Conv2D cukup
            }
        )
        return model
    except Exception as e:
        # Tampilkan detail error yang lebih lengkap (Traceback) untuk diagnosa
        import traceback
        st.error(f"CRITICAL ERROR loading {model_name}.")
        st.code(traceback.format_exc()) # Ini akan menampilkan baris mana yang error
        return None

def preprocess_image(image, model_name):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Tentukan ukuran berdasarkan arsitektur
    if model_name == "Inception V3":
        target_size = (299, 299)
    else:
        target_size = (224, 224)
    
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    if model_name == "Inception V3":
        img_array = preprocess_inception(img_array)
    elif model_name == "ResNet50":
        img_array = preprocess_resnet(img_array)
    else:
        img_array = img_array / 255.0
    return img_array

# --- UI STREAMLIT ---
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

st.title("Klasifikasi Penyakit Mangga")
st.markdown("Dashboard untuk membandingkan performa model dan melakukan prediksi penyakit.")

st.sidebar.header("Konfigurasi")
st.sidebar.caption(f"TF Version: {tf.__version__}")
selected_model_name = st.sidebar.selectbox("Pilih Model AI:", list(MODEL_PATHS.keys()))

st.header(f"📊 Performa Training: {selected_model_name}")
col_graph1, col_graph2 = st.columns(2)

graphs = GRAPH_PATHS[selected_model_name]

if os.path.exists(graphs[0]) and os.path.exists(graphs[1]):
    with col_graph1:
        st.image(graphs[0], caption="Akurasi", use_container_width=True)
    with col_graph2:
        st.image(graphs[1], caption="Loss", use_container_width=True)
else:
    st.warning(f"File grafik tidak ditemukan. Cek folder graph.")

st.divider()

st.header("🔍 Uji Coba Prediksi")
st.write(f"Model aktif: **{selected_model_name}**")

uploaded_file = st.file_uploader("Upload gambar mangga...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col_img, col_res = st.columns([1, 2])
    image = Image.open(uploaded_file)
    with col_img:
        st.image(image, caption="Gambar Input", use_container_width=True)
    
    with col_res:
        if st.button("Mulai Deteksi", type="primary"):
            with st.spinner('Sedang memuat model & memproses...'):
                model = load_model(selected_model_name)
                
                if model:
                    try:
                        processed_img = preprocess_image(image, selected_model_name)
                        predictions = model.predict(processed_img)
                        
                        pred_class_idx = np.argmax(predictions[0])
                        pred_label = CLASS_NAMES[pred_class_idx]
                        confidence = np.max(predictions[0]) * 100

                        if confidence > 60:
                            st.success(f"Hasil: **{pred_label}**")
                        else:
                            st.warning(f"Hasil: **{pred_label}** (Kurang Yakin)")
                            
                        st.metric("Confidence", f"{confidence:.2f}%")
                        st.progress(int(confidence))
                        
                        with st.expander("Detail Probabilitas"):
                            probs = {k: f"{v*100:.2f}%" for k, v in zip(CLASS_NAMES, predictions[0])}
                            st.json(probs)
                    except Exception as e:
                        st.error(f"Error saat prediksi: {e}")
                        import traceback
                        st.caption(traceback.format_exc())