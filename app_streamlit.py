""" import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# ✅ Download model
if not os.path.exists("cnn.h5"):
    url = "https://drive.google.com/uc?id=10rWPrSDSD0t4kXo_IUAp9ijjNB2y1ILd"
    gdown.download(url, "cnn.h5", quiet=False)

st.title("Alzheimer MRI Detection")

st.write("🔄 Loading model...")

MODEL = load_model("cnn.h5")

st.write("✅ Model loaded")

CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

file = st.file_uploader("Choose an MRI Image", type=["jpg","png","jpeg"])

if file is not None:
    st.write("📂 File uploaded")

    img = Image.open(file).convert("RGB").resize((128,128))
    st.image(img, caption="Uploaded Image", width=300)

    st.write("⚙️ Processing image...")

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    st.write("🤖 Predicting...")

    pred = MODEL.predict(img)

    st.write("✅ Prediction done")

    result = CLASSES[np.argmax(pred)]

    st.success(f"Prediction: {result}")



import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
from googletrans import Translator, LANGUAGES

# Page setup
st.set_page_config(page_title="Brain Health Check", layout="centered")

# Language selection (200+ languages)
translator = Translator()
language_name = st.selectbox("🌍 Select Language", list(LANGUAGES.values()))
lang_code = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language_name)]

# Translate function
def translate_text(text):
    try:
        return translator.translate(text, dest=lang_code).text
    except:
        return text

# Title & description
st.title(translate_text("🧠 Brain Health Check"))
st.write(translate_text("Upload a brain MRI image to check memory condition."))

# Download model if not exists
MODEL_PATH = "cnn.h5"
MODEL_URL = "https://drive.google.com/uc?id=10rWPrSDSD0t4kXo_IUAp9ijjNB2y1ILd"

if not os.path.exists(MODEL_PATH):
    with st.spinner(translate_text("Downloading model...")):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model (cached)
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

MODEL = load_my_model()

# Class labels
CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

# Upload image
file = st.file_uploader(translate_text("Select Brain Image"), type=["jpg","png","jpeg"])

if file is not None:

    # Show image
    img = Image.open(file).convert("RGB").resize((128,128))
    st.image(img, caption=translate_text("Your Image"), width=300)

    # Preprocess image
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner(translate_text("Analyzing image...")):
        pred = MODEL.predict(img_array)

    result = CLASSES[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Simple result messages
    if result == "NonDemented":
        msg = "Normal Brain. No disease found."
    elif result == "VeryMildDemented":
        msg = "Very Early Stage. Small memory problems may start."
    elif result == "MildDemented":
        msg = "Mild Stage. Memory problems are noticeable."
    elif result == "ModerateDemented":
        msg = "Serious Stage. Strong memory problems."

    # Display result
    st.subheader(translate_text("Result"))
    st.write(translate_text(msg))

    # Display confidence
    st.write(f"{translate_text('Accuracy')}: {confidence:.2f}%")
     # Probability chart
    st.subheader(translate_text("Prediction Probabilities"))

    prob_data = {
        CLASSES[i]: float(pred[0][i])
        for i in range(len(CLASSES))
    }

    st.bar_chart(prob_data)
    """
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os
from googletrans import Translator, LANGUAGES

# ---------------- Page setup ----------------
st.set_page_config(page_title="Brain Health Check", layout="centered")

# ---------------- Language selection ----------------
translator = Translator()
language_name = st.selectbox("🌍 Select Language", list(LANGUAGES.values()))
lang_code = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language_name)]

def translate_text(text):
    try:
        return translator.translate(text, dest=lang_code).text
    except:
        return text

# ---------------- Title ----------------
st.title(translate_text("🧠 Brain Health Check"))
st.write(translate_text("Upload a brain MRI image to check memory condition."))

# ---------------- Model folder ----------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Model info (USE FILE IDs ONLY) ----------------
models_info = {
    "CNN": {
        "file": os.path.join(MODEL_DIR, "cnn.h5"),
        "id": "14IV_Oiixmox5NJ8EjeEG3_tHeuSCuFZC"
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
@st.cache_resource
def load_models():
    loaded_models = {}

    for name, info in models_info.items():
        path = info["file"]

        # Download only if not exists
        if not os.path.exists(path):
            with st.spinner(f"Downloading {name} model..."):
                url = f"https://drive.google.com/uc?id={info['id']}"
                gdown.download(url, path, quiet=False)

        # Load model
        loaded_models[name] = load_model(path, compile=False)

    return loaded_models

# ---------------- Load models ----------------
with st.spinner("Loading AI models..."):
    MODELS = load_models()

# ---------------- Class labels ----------------
CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

# ---------------- Upload image ----------------
file = st.file_uploader(translate_text("Select Brain Image"), type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB").resize((128,128))
    st.image(img, caption=translate_text("Your Image"), width=300)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- Predict ----------------
    with st.spinner(translate_text("Analyzing image...")):
        probs = []

        for name, model in MODELS.items():
            pred = model.predict(img_array)
            probs.append(pred[0])

        # Ensemble (average)
        combined_prob = np.mean(np.array(probs), axis=0)
        result_index = np.argmax(combined_prob)
        result = CLASSES[result_index]
        confidence = combined_prob[result_index] * 100

    # ---------------- Message ----------------
    if result == "NonDemented":
        msg = "Normal Brain. No disease found."
    elif result == "VeryMildDemented":
        msg = "Very Early Stage. Small memory problems may start."
    elif result == "MildDemented":
        msg = "Mild Stage. Memory problems are noticeable."
    elif result == "ModerateDemented":
        msg = "Serious Stage. Strong memory problems."

    # ---------------- Output ----------------
    st.subheader(translate_text("Result"))
    st.write(translate_text(msg))
    st.write(f"{translate_text('Confidence')}: {confidence:.2f}%")
    st.write(translate_text("Model Used: Ensemble of CNN, VGG16, MobileNet"))

    # ---------------- Chart ----------------
    st.subheader(translate_text("Prediction Probabilities"))
    prob_data = {CLASSES[i]: float(combined_prob[i]) for i in range(len(CLASSES))}
    st.bar_chart(prob_data)
