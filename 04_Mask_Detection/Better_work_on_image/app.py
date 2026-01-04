import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import requests
import os

st.title("Face Mask Detection")
st.subheader("Upload an image to detect if a person is wearing a mask or not.")

# Load model from Hugging Face (cached)
@st.cache_resource
def load_mask_model():
    model_url = "https://huggingface.co/MrLeo0087/mask_detection2_cnn/resolve/main/mask_detector.h5"
    
    # Download model to a local file
    local_path = "mask_detector.h5"
    if not os.path.exists(local_path):
        r = requests.get(model_url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    
    # Load model
    model = load_model(local_path)
    return model

model = load_mask_model()

# Class labels
labels_dict = {0: "With Mask", 1: "Without Mask"}

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for MobileNetV2
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Make prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx] * 100
    label = f"{labels_dict[class_idx]} ({confidence:.2f}%)"

    # Display result
    st.success(label)
