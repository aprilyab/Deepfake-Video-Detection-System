
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 96
SEQUENCE_LENGTH = 4

st.set_page_config(page_title="DeepFake Detector", layout="centered")

@st.cache_resource
def load_my_model():
    model = load_model("model/Deepfake_CNN_LSTM_clean.h5")
    return model

model = load_my_model()

# Sidebar with instructions and about
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a CNN-LSTM deep learning model to detect deepfake images. "
    "Upload an image and click 'Predict' to see if it is real or fake."
)
st.sidebar.markdown("---")
st.sidebar.write("**Model Info:**")
st.sidebar.write("- Input size: 96x96 RGB")
st.sidebar.write("- Sequence length: 4 (frames)")
st.sidebar.write("- Architecture: CNN + LSTM")

st.title("DeepFake Detection App")
st.write("Upload an image to check if it is **Real or Fake**.")

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
    # Add a Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = model.predict(input_data)[0][0]
        st.subheader("Result:")
        if prediction > 0.5:
            st.error(f"🚨 Fake Image ({prediction:.2%} confidence)")
        else:
            st.success(f"✅ Real Image ({(1 - prediction):.2%} confidence)")
        st.progress(float(prediction) if prediction > 0.5 else 1 - float(prediction))
        st.write(f"Raw model output: {prediction:.4f}")

# Optionally, add more features below (history, feedback, etc.)