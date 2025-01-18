import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Streamlit App Title
st.title("Dog and Cat Classification")

# Sidebar for User Input
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input(
    "Model Path", r"D:\MY_PROJECTS\Dog_Cat_Classfication\dog_cat_classifier.keras"
)

# Function to load the model
@st.cache_resource
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

# Load the model at the start
if model_path:
    model = load_trained_model(model_path)
    if model:
        st.sidebar.success("Model loaded successfully!")
    else:
        st.sidebar.error("Failed to load model. Check the path.")

# Image Upload Section
uploaded_file = st.file_uploader("Upload an image of a dog or a cat", type=["jpg", "jpeg", "png"])

# Submit Button
if st.button("Submit"):
    if uploaded_file is not None and model:
        try:
            # Display Uploaded Image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess the Image to match model input
            img = img.resize((256, 256))  # Resize to match model input size (256, 256)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize to [0, 1]

            # Make Prediction
            prediction = model.predict(img_array)
            class_idx = (prediction[0] > 0.5).astype("int32")  # Binary classification threshold
            confidence = prediction[0][0] if class_idx == 1 else (1 - prediction[0][0])

            # Map Class Index to Labels
            labels = {0: "Cat", 1: "Dog"}
            predicted_label = labels[class_idx[0]]

            # Display Prediction
            st.write(f"### Prediction: {predicted_label}")
            st.write(f"### Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    elif not uploaded_file:
        st.warning("Please upload an image before submitting.")
    elif not model:
        st.error("Model not loaded. Please check the model path.")
