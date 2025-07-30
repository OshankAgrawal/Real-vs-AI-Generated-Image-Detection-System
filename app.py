import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import gdown

@st.cache_resource
def load_model():
    import os
    import requests

    file_url = "https://drive.google.com/uc?export=download&id=14K8dRaJBR01z1RSonNh7xCt4ghrp360l"  # file URL
    file_name = "Rebuild-model.h5"
    gdown.download(file_url, file_name, quiet=False)

    # Check file size and type before loading
    if os.path.exists(file_name):
        # st.write("Downloaded file size:", os.path.getsize(file_name), "bytes")
        with open(file_name, "rb") as f:
            signature = f.read(8)
            # st.write("File signature (first 8 bytes):", signature)

    model = tf.keras.models.load_model(file_name)
    return model


model = load_model()

# Image Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# UI part
st.set_page_config(page_title="AI Real Eyes", page_icon="👁️", layout="centered")
st.markdown("""
    <style>
        body {
            background: #0f1117;
            color: white;
        }
        .main {
            background-color: #0f1117;
        }
    </style>
""", unsafe_allow_html=True)

st.title("👁️ AI Real Eyes")
st.subheader("Detect whether an image is AI-generated or Real")

st.write("Upload an image and let our AI reveal the truth! 📷 Was it captured by a camera — or crafted by code? Put our smart detection engine to the test and uncover the reality behind the pixels!")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.write("Analyzing image...")
    
    with st.spinner("Running model..."):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0]

        # Detection part
        if len(prediction) == 1:
            confidence = float(prediction[0])  # e.g. 0.87
            if confidence >= 0.5:
                result = "🧠 AI Generated Image"
            else:
                result = "📷 Real Photograph"
        elif len(prediction) == 2:
            real_conf = prediction[0]
            ai_conf = prediction[1]
            if ai_conf > real_conf:
                result = "🧠 AI Generated Image"
                confidence = ai_conf
            else:
                result = "📷 Real Photograph"
                confidence = real_conf
        else:
            result = "❓ Unknown Model Output"
            confidence = 0.0


        # Simulate a progress bar (like your JS version)
        progress = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.001)
            progress.progress(i)

    st.success(f"**Result**: {result}")
    st.info(f"**Confidence**: {confidence * 100:.2f}%")

    st.image(image, caption="Uploaded Image", use_container_width=True)