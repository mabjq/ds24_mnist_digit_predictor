import streamlit as st
import numpy as np
import joblib
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the best model with caching
@st.cache_resource
def load_model():
    return joblib.load("svm_augmented_model.pkl")

model = load_model()

# Sidebar with info
st.sidebar.title("About")
st.sidebar.write("This app predicts handwritten digits using an SVM model trained on MNIST data.")
st.sidebar.write("Precision: ~74% (based on 500 drawing attempts)")
st.sidebar.write("Tips: Draw clearly, use good lighting for camera, upload grayscale images.")

# Main title
st.title("✏️ MNIST Digit Predictor")
st.write("Draw a digit, upload an image, or use your camera to predict a digit (0-9).")

# Tabs for drawing, uploading, and camera
tab1, tab2, tab3 = st.tabs(["Draw a Digit", "Upload an Image", "Camera Feed"])

# Tab 1: Drawing Canvas
with tab1:
    st.write("Draw a single digit (0-9) clearly. Use the canvas below.")
    threshold_value = st.slider("Adjust Threshold", 0, 255, 50, key="canvas_threshold")
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

        # Interpolarisation
        image_binary = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
        _, image_binary = cv2.threshold(image_binary, threshold_value, 255, cv2.THRESH_BINARY)

        st.image(image_binary, caption="Processed Digit", width=280)

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

# Tab 3: Camera Feed
with tab3:
    st.write("Use your webcam to capture a digit. Adjust the threshold and press 'Start Camera'.")
    run = st.checkbox("Start Camera")
    frame_window = st.image([])

    threshold_value = st.slider("Adjust Threshold", 0, 255, 100)

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the camera. Please check your webcam.")
        else:
            predictions = []
            max_predictions = 5

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from camera.")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image_binary = cv2.resize(gray_frame, (28, 28), interpolation=cv2.INTER_AREA)
                _, image_binary = cv2.threshold(image_binary, threshold_value, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    scale_x = frame.shape[1] / 28
                    scale_y = frame.shape[0] / 28
                    x_roi, y_roi, w_roi, h_roi = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
                    cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)

                    image_array = image_binary / 255.0
                    image_array = image_array.reshape(1, -1)
                    prediction = model.predict(image_array)[0]
                    predictions.append(prediction)
                    if len(predictions) > max_predictions:
                        predictions.pop(0)
                    smoothed_prediction = int(np.mean(predictions)) if predictions else prediction
                else:
                    smoothed_prediction = None

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                caption = f"Predicted Digit: {smoothed_prediction if smoothed_prediction is not None else 'None'}"
                frame_window.image(frame_rgb, caption=caption, width=640)

            cap.release() 
            cv2.destroyAllWindows()
    if not run:
        st.write("Camera is stopped.")