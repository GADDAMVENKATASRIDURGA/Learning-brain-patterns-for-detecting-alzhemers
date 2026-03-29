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

# Page setup
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


# Model paths
CNN_PATH = "cnn.h5"
VGG_PATH = "vgg16.h5"
MOB_PATH = "mobilenet.h5"

# Example model links
CNN_URL = "https://drive.google.com/uc?id=10rWPrSDSD0t4kXo_IUAp9ijjNB2y1ILd"
VGG_URL = "PASTE_VGG16_LINK"
MOB_URL = "PASTE_MOBILENET_LINK"


# Download models
if not os.path.exists(CNN_PATH):
    gdown.download(CNN_URL, CNN_PATH, quiet=False)

if not os.path.exists(VGG_PATH):
    gdown.download(VGG_URL, VGG_PATH, quiet=False)

if not os.path.exists(MOB_PATH):
    gdown.download(MOB_URL, MOB_PATH, quiet=False)


st.write(translate_text("🔄 Loading AI models..."))

# Load models
@st.cache_resource
def load_models():
    cnn = load_model(CNN_PATH, compile=False)
    vgg = load_model(VGG_PATH, compile=False)
    mob = load_model(MOB_PATH, compile=False)
    return cnn, vgg, mob

CNN_MODEL, VGG_MODEL, MOB_MODEL = load_models()

st.write(translate_text("✅ Models loaded"))


CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']


file = st.file_uploader(translate_text("Select Brain Image"), type=["jpg","png","jpeg"])

if file is not None:

    img = Image.open(file).convert("RGB").resize((128,128))
    st.image(img, caption=translate_text("Your Image"), width=300)

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner(translate_text("Analyzing image...")):

        pred1 = CNN_MODEL.predict(img)
        pred2 = VGG_MODEL.predict(img)
        pred3 = MOB_MODEL.predict(img)

        # Average prediction
        pred = (pred1 + pred2 + pred3) / 3

    result = CLASSES[np.argmax(pred)]
    confidence = np.max(pred) * 100


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

    st.write(f"{translate_text('Accuracy')}: {confidence:.2f}%")

