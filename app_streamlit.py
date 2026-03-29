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
import time

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(page_title="Brain Health Check", layout="centered")

# Language selection
translator = Translator()
language_name = st.selectbox("🌍 Select Language", list(LANGUAGES.values()))
lang_code = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language_name)]

def translate_text(text):
    try:
        return translator.translate(text, dest=lang_code).text
    except:
        return text

st.title(translate_text("🧠 Brain Health Check"))
st.write(translate_text("Upload a brain MRI image to check memory condition."))

# ---------------- DOWNLOAD MODELS ---------------- #
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# Google Drive IDs
CNN_ID = "https://drive.google.com/file/d/1ElTwfxMzzl1yaGeRPpy_YYrGXyRD4NEV/view?usp=sharing"           # replace with your Google Drive ID
VGG_ID = "https://drive.google.com/file/d/1ywWoScQeq1uULSmtsRwterh1ce0Wght2/view?usp=sharing"
MOBILE_ID = "https://drive.google.com/file/d/1UlTnRkATQcwPOcq2awxPM_bZ6rk2ZIu0/view?usp=sharing"

download_model(CNN_ID, "cnn.h5")
download_model(VGG_ID, "vgg16.h5")
download_model(MOBILE_ID, "mobilenet.h5")

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_all_models():
    cnn = load_model("cnn.h5", compile=False)
    vgg = load_model("vgg16.h5", compile=False)
    mobile = load_model("mobilenet.h5", compile=False)
    return cnn, vgg, mobile

CNN_MODEL, VGG_MODEL, MOBILE_MODEL = load_all_models()

# ---------------- MODEL ACCURACY ---------------- #
# Update with your training accuracies
model_accuracy = {
    "CNN": 0.89,
    "VGG16": 0.92,
    "MobileNet": 0.90
}

# ---------------- CLASS LABELS ---------------- #
CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

# ---------------- UPLOAD IMAGE ---------------- #
file = st.file_uploader(translate_text("Select Brain Image"), type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB")
    img_resized = img.resize((128,128))
    st.image(img_resized, caption=translate_text("Your Image"), width=300)

    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- AUTO MODEL SELECTION ---------------- #
    def select_best_model(img):
        scores = {}
        models = {
            "CNN": CNN_MODEL,
            "VGG16": VGG_MODEL,
            "MobileNet": MOBILE_MODEL
        }
        for name, model in models.items():
            start = time.time()
            pred = model.predict(img)
            end = time.time()
            inference_time = end - start
            score = model_accuracy[name] / inference_time
            scores[name] = (score, pred)
        best_model = max(scores, key=lambda x: scores[x][0])
        return best_model, scores[best_model][1]

    # Predict
    with st.spinner(translate_text("Analyzing image...")):
        best_model, pred = select_best_model(img_array)

    result = CLASSES[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # ---------------- SIMPLE RESULT MESSAGES ---------------- #
    if result == "NonDemented":
        msg = "Normal Brain. No disease found."
    elif result == "VeryMildDemented":
        msg = "Very Early Stage. Small memory problems may start."
    elif result == "MildDemented":
        msg = "Mild Stage. Memory problems are noticeable."
    elif result == "ModerateDemented":
        msg = "Serious Stage. Strong memory problems."

    st.subheader(translate_text("Result"))
    st.write(translate_text(msg))
    st.write(f"{translate_text('Confidence')}: {confidence:.2f}%")
    st.write(f"{translate_text('Model Used')}: {best_model}")

    # ---------------- PREDICTION PROBABILITIES ---------------- #
    st.subheader(translate_text("Prediction Probabilities"))
    prob_data = {CLASSES[i]: float(pred[0][i]) for i in range(len(CLASSES))}
    st.bar_chart(prob_data)
