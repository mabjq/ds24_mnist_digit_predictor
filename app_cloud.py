import streamlit as st
import numpy as np
import joblib
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import gdown
import os

st.set_page_config(
    page_title="MNIST Digit Predictor (Cloud)",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Google Drive URL for the model file 
MODEL_URL = "https://drive.google.com/uc?id=1lbazk6-Jit5xUzZqIKYJlZCXh1_YLUIp" 
MODEL_PATH = "svm_augmented_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("Model downloaded successfully!")

# Load the model with caching
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar with info
st.sidebar.title("About")
st.sidebar.write("This app predicts handwritten digits using an SVM model trained on MNIST data.")
st.sidebar.write("Precision: ~74% (based on 500 drawing attempts locally)")
st.sidebar.write("Tips: Draw clearly, upload grayscale images. Camera snapshot works best with good lighting.")
st.sidebar.write("Model Accuracy: 0.9879 on test set.")

# Main title
st.title("✏️ MNIST Digit Predictor (Cloud)")
st.write("Draw a digit, upload an image, or take a snapshot with your camera to predict a digit (0-9).")

# Tabs for drawing, uploading, and camera
tab1, tab2, tab3 = st.tabs(["Draw a Digit", "Upload an Image", "Camera Snapshot"])

# Tab 1: Drawing Canvas
with tab1:
    st.write("Draw a single digit (0-9) clearly. Use the canvas below.")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None and np.any(canvas_result.image_data):
        image = Image.fromarray(canvas_result.image_data).convert("L")
        image = np.array(image)

        image_binary = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        _, image_binary = cv2.threshold(image_binary, 50, 255, cv2.THRESH_BINARY)

        st.image(image_binary, caption="Processed Digit", use_container_width=True)

        image_array = image_binary / 255.0
        image_array = image_array.reshape(1, -1)

        prediction = model.predict(image_array)[0]
        st.markdown(f'<p style="color:#FF4B4B; font-size:20px;">Predicted Digit: {prediction}</p>', unsafe_allow_html=True)
    else:
        st.write("No drawing detected. Please draw a digit.")

# Tab 2: Image Upload
with tab2:
    st.write("Upload a grayscale image of a digit (0-9).")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        if image.size[0] > 0 and image.size[1] > 0: 
            image = np.array(image)

            avg_pixel_value = np.mean(image)
            if avg_pixel_value > 128:
                _, image_binary = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
            else:
                _, image_binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

            image_binary = cv2.resize(image_binary, (28, 28), interpolation=cv2.INTER_AREA)

            st.image(image_binary, caption="Processed Digit", use_container_width=True)

            image_array = image_binary / 255.0
            image_array = image_array.reshape(1, -1)

            prediction = model.predict(image_array)[0]
            st.markdown(f'<p style="color:#FF4B4B; font-size:20px;">Predicted Digit: {prediction}</p>', unsafe_allow_html=True)
        else:
            st.error("Invalid image file. Please upload a valid image.")

# Tab 3: Camera Snapshot (using st.camera_input)
with tab3:
    st.write("Take a snapshot with your webcam to capture a digit. Ensure good lighting and clear handwriting.")
    
    # Use st.camera_input to capture a single image from the webcam
    camera_image = st.camera_input("Take a picture of a digit")

    # Slider for threshold adjustment
    threshold_value = st.slider("Adjust Threshold", 0, 255, 100)

    if camera_image is not None:
        # Convert the captured image to grayscale
        image = Image.open(camera_image).convert("L")
        if image.size[0] > 0 and image.size[1] > 0:  
            image = np.array(image)

            # Preprocess the image
            avg_pixel_value = np.mean(image)
            if avg_pixel_value > 128:
                _, image_binary = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY_INV)
            else:
                _, image_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

            image_binary = cv2.resize(image_binary, (28, 28), interpolation=cv2.INTER_AREA)

            # Display the processed image
            st.image(image_binary, caption="Processed Digit", use_container_width=True)

            # Prepare for prediction
            image_array = image_binary / 255.0
            image_array = image_array.reshape(1, -1)

            # Predict
            prediction = model.predict(image_array)[0]
            st.markdown(f'<p style="color:#FF4B4B; font-size:20px;">Predicted Digit: {prediction}</p>', unsafe_allow_html=True)
        else:
            st.error("Invalid camera input. Please try again with a valid image.")