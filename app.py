import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image


IMG_SIZE = 96
SEQUENCE_LENGTH = 4

st.set_page_config(page_title="model/DeepFake Detector", layout="centered")

@st.cache_resource
def load_my_model():
    model = load_model("model/Deepfake_CNN_LSTM.h5")

    return model

model = load_my_model()


st.title(" DeepFake Detection App")
st.write("Upload an image to check if it is **Real or Fake**")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    # Convert single image → sequence
    sequence = [img] * SEQUENCE_LENGTH
    sequence = np.array(sequence)

   
    return np.expand_dims(sequence, axis=0)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_data = preprocess_image(image)

    prediction = model.predict(input_data)[0][0]

    st.subheader("Result:")

    if prediction > 0.5:
        st.error(f"🚨 Fake Image ({prediction:.2f} confidence)")
    else:
        st.success(f"✅ Real Image ({1 - prediction:.2f} confidence)")