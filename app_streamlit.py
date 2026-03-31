# Import required libraries
import streamlit as st                     # For building the web app UI
import numpy as np                        # For numerical operations
from tensorflow.keras.models import load_model  # To load trained deep learning models
from PIL import Image                     # For image processing
import gdown                              # To download files from Google Drive
import os                                 # For file and folder operations
from googletrans import Translator, LANGUAGES  # For multi-language support

# ---------------- Page setup ----------------
# Configure the Streamlit page (title and layout)
st.set_page_config(page_title="Brain Health Check", layout="centered")

# ---------------- Language selection ----------------
# Initialize translator
translator = Translator()

# Dropdown for selecting language
language_name = st.selectbox("🌍 Select Language", list(LANGUAGES.values()))

# Get corresponding language code
lang_code = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language_name)]

# Function to translate text dynamically
def translate_text(text):
    try:
        return translator.translate(text, dest=lang_code).text  # Translate text
    except:
        return text  # Return original text if translation fails

# ---------------- Title ----------------
# Display app title and instructions
st.title(translate_text("🧠 Brain Health Check"))
st.write(translate_text("Upload a brain MRI image to check memory condition."))

# ---------------- Model folder ----------------
# Define directory to store models
MODEL_DIR = "model"

# Create folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Model info ----------------
# Dictionary storing model file paths and Google Drive IDs
models_info = {
    "CNN": {
        "file": os.path.join(MODEL_DIR, "cnn.h5"),   # Local path
        "id": "14IV_Oiixmox5NJ8EjeEG3_tHeuSCuFZC"    # Google Drive file ID
    },
    "VGG16": {
        "file": os.path.join(MODEL_DIR, "vgg16.h5"),
        "id": "1IpMbk-H36jGS5BTxo82OeR6QRrpVsrRx"
    },
    "MobileNet": {
        "file": os.path.join(MODEL_DIR, "mobilenet.h5"),
        "id": "1iDWSXxGbwTdYnxpCyV0x9dE92dO03CLy"
    }
}

# ---------------- Load + Download Models (CACHED) ----------------
# Cache models so they load only once (improves performance)
@st.cache_resource
def load_models():
    loaded_models = {}

    # Loop through each model
    for name, info in models_info.items():
        path = info["file"]

        # Download model if not already present
        if not os.path.exists(path):
            with st.spinner(f"Downloading {name} model..."):
                url = f"https://drive.google.com/uc?id={info['id']}"
                gdown.download(url, path, quiet=False)

        # Load the model (without compiling to save time)
        loaded_models[name] = load_model(path, compile=False)

    return loaded_models

# ---------------- Load models ----------------
# Show loading spinner while models are loading
with st.spinner("Loading AI models..."):
    MODELS = load_models()

# ---------------- Class labels ----------------
# Define output classes for prediction
CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

# ---------------- Upload image ----------------
# File uploader for user to upload MRI image
file = st.file_uploader(translate_text("Select Brain Image"), type=["jpg","png","jpeg"])

# If user uploads an image
if file is not None:
    # Open image, convert to RGB, and resize to model input size
    img = Image.open(file).convert("RGB").resize((128,128))

    # Display uploaded image
    st.image(img, caption=translate_text("Your Image"), width=300)

    # Convert image to numpy array and normalize pixel values
    img_array = np.array(img)/255.0

    # Expand dimensions to match model input shape (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- Predict ----------------
    with st.spinner(translate_text("Analyzing image...")):
        probs = []

        # Get predictions from each model
        for name, model in MODELS.items():
            pred = model.predict(img_array)
            probs.append(pred[0])  # Store probabilities

        # Ensemble method: average predictions of all models
        combined_prob = np.mean(np.array(probs), axis=0)

        # Get class with highest probability
        result_index = np.argmax(combined_prob)
        result = CLASSES[result_index]

        # Confidence score
        confidence = combined_prob[result_index] * 100

    # ---------------- Message ----------------
    # Generate message based on prediction
    if result == "NonDemented":
        msg = "Normal Brain. No disease found."
    elif result == "VeryMildDemented":
        msg = "Very Early Stage. Small memory problems may start."
    elif result == "MildDemented":
        msg = "Mild Stage. Memory problems are noticeable."
    elif result == "ModerateDemented":
        msg = "Serious Stage. Strong memory problems."

    # ---------------- Output ----------------
    # Display results
    st.subheader(translate_text("Result"))
    st.write(translate_text(msg))
    st.write(f"{translate_text('Confidence')}: {confidence:.2f}%")
    st.write(translate_text("Model Used: Ensemble of CNN, VGG16, MobileNet"))

    # ---------------- Chart ----------------
    # Show probability distribution for all classes
    st.subheader(translate_text("Prediction Probabilities"))

    # Convert probabilities into dictionary format
    prob_data = {CLASSES[i]: float(combined_prob[i]) for i in range(len(CLASSES))}

    # Display bar chart
    st.bar_chart(prob_data)
