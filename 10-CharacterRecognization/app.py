# ==========================================================
# Handwritten Character Recognition (A–Z) using MLP (Improved)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(page_title="Character Recognition using MLP", layout="wide")

st.title("🔤 Handwritten Character Recognition (A–Z)")
st.markdown("Upload a 28x28 grayscale character image for prediction.")

st.markdown("---")

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():
    data = pd.read_csv("A_Z Handwritten Data.csv", nrows=50000)
    return data

data = load_data()

X = data.iloc[:, 1:].values / 255.0   # 🔥 Normalize pixels
y = data.iloc[:, 0].values

letters = [chr(i) for i in range(65, 91)]
y_letters = np.array([letters[int(label)] for label in y])

# ==========================================================
# TRAIN MODEL (ONLY FIRST TIME)
# ==========================================================

model_exists = os.path.exists("model.pkl")

if not model_exists:

    st.info("Training model for first time...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_letters, test_size=0.2, random_state=42
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=300,          # 🔥 Increased iterations
        random_state=42,
        verbose=True
    )

    with st.spinner("Training MLP model..."):
        mlp.fit(X_train, y_train)

    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred, labels=letters)
    }

    joblib.dump((mlp, metrics), "model.pkl")
    st.success("Model trained and saved successfully!")

# ==========================================================
# LOAD MODEL
# ==========================================================

model, metrics = joblib.load("model.pkl")

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.subheader("📤 Upload Character Image")

uploaded_file = st.file_uploader(
    "Upload a handwritten character image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image)

    # 🔥 Auto-invert if background is white
    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    display_image = image_array.astype(np.uint8)

    image_flat = image_array.reshape(1, -1) / 255.0  # 🔥 Normalize

    if st.button("Analyze Character"):

        prediction = model.predict(image_flat)
        probabilities = model.predict_proba(image_flat)

        col1, col2 = st.columns(2)

        with col1:
            st.image(display_image, caption="Processed Image", width=200)

        with col2:
            st.success(f"### Predicted Character: {prediction[0]}")

            confidence = np.max(probabilities) * 100
            st.write(f"Confidence: {confidence:.2f}%")

st.markdown("---")

# ==========================================================
# TRAINING METRICS
# ==========================================================

if st.button("📊 Show Training Outputs"):

    st.write(f"Training Accuracy: {metrics['train_accuracy']*100:.2f}%")
    st.write(f"Testing Accuracy: {metrics['test_accuracy']*100:.2f}%")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        metrics['confusion_matrix'],
        cmap='Blues',
        xticklabels=letters,
        yticklabels=letters,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)
    plt.close()