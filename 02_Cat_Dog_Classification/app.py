import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat vs Dog Image Classifier")

# ----------------------------
# Load the model (cached for performance)
# ----------------------------
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/MrLeo0087/Cat-Dog-Classifier/resolve/main/cnn_mobilenetv2_model.keras"
    model_path = tf.keras.utils.get_file("cnn_mobilenetv2_model.keras", model_url)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ----------------------------
# Class labels
# ----------------------------
class_labels = {0: "Cat", 1: "Dog"}

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image, target_size=(224,224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(image):
    img_array = preprocess_image(image)
    pred = model.predict(img_array)[0][0]
    label = "Dog" if pred >= 0.5 else "Cat"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence

# ----------------------------
# Streamlit file uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload an image of a Cat or Dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict_image(image)
        st.subheader("Prediction")
        st.markdown(f"**Type:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        emoji = "🐶 Dog" if label == "Dog" else "🐱 Cat"
        st.markdown(f"### {emoji}")
