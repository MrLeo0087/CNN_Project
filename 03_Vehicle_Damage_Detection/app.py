from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

import streamlit as st

st.title('Vehicle Damage Detection')
st.subheader('Upload an image of a vehicle to detect damage.')  


@st.cache_resource
def load_model():
    model_url = 'https://huggingface.co/MrLeo0087/vehicle-damage-detection-cnn/resolve/main/car_damage.keras'
    model_path = tf.keras.utils.get_file("car_damage.keras", model_url)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    img = image.resize((224, 224))
    img_array = preprocess_input(np.array(img))
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    if predictions[0][0] > 0.5:
        st.write("The vehicle is not damaged.") 
    else:
        st.write("The vehicle is damaged.")     


