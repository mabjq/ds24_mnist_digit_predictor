# Handwritten Digit Recognition with MNIST

This project develops a handwritten digit classifier using the MNIST dataset, optimized for real-world recognition. The notebook (`mnist.ipynb`) explores multiple models, with an augmented SVM achieving a test accuracy of 0.9878.

## Project Overview
- **Goal:** Recognize digits from drawings, uploads, or camera inputs using a Streamlit app.
- **Models:** Logistic Regression (0.9233), Random Forest (0.9713), SVM (0.9852), Ensemble (0.9782), Augmented SVM (0.9878).
- **Techniques:** Data augmentation to enhance robustness.

## Files
- `mnist.ipynb`: Jupyter notebook with analysis and model training.
- `app.py`: Local Streamlit app with drawing, upload, and live webcam support.
- `app_cloud.py`: Cloud-compatible Streamlit app with drawing, upload, and camera snapshot.
- `requirements.txt`: Dependencies.
- `README.md`: This file.

**Note:** The trained model (`svm_augmented_model.pkl`) is hosted on Google Drive due to GitHub size limits and is downloaded automatically by `app_cloud.py`.

## Installation

pip install -r requirements.txt

Dependencies: streamlit, numpy, scikit-learn, joblib, streamlit-drawable-canvas, pillow, opencv-python-headless, gdown.

## Usage
- Run Locally: streamlit run app.py
- Cloud Version: Live App URL (https://mnistdigitpredictor-d2m99z8cma94zjp48wppno.streamlit.app/)

## Why Augmentation?
Augmentation simulated real-world variations (camera angles, handwriting styles), increasing training data and improving generalization from 0.9852 to 0.9878.

## Future Work
- Improve precision (currently ~74% on drawings).
- Optimize app performance.


