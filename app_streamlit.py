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
"""
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# ✅ Page settings
st.set_page_config(page_title="Alzheimer Detection", layout="centered")

# ✅ Download model
if not os.path.exists("cnn.h5"):
    url = "https://drive.google.com/uc?id=10rWPrSDSD0t4kXo_IUAp9ijjNB2y1ILd"
    gdown.download(url, "cnn.h5", quiet=False)

# ✅ Title
st.title("🧠 Alzheimer MRI Detection")
st.write("Upload an MRI scan to detect Alzheimer stage using AI.")

# ✅ Load model
MODEL = load_model("cnn.h5", compile=False)

CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

# ✅ File upload
file = st.file_uploader("Choose an MRI Image", type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB").resize((128,128))
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = MODEL.predict(img)
    result = CLASSES[np.argmax(pred)]

    # ✅ Confidence
    confidence = np.max(pred) * 100

    # ✅ Better output
    if result == "NonDemented":
        st.success("🟢 Normal Brain\nNo disease found.")

    elif result == "VeryMildDemented":
        st.info("🟡 Very Early Stage\nSmall memory problems may start.")

    elif result == "MildDemented":
        st.warning("🟠 Mild Stage\nMemory problems are noticeable.")

    elif result == "ModerateDemented":
        st.error("🔴 Serious Stage\nStrong memory and thinking problems.")

    # ✅ Confidence display
    st.info(f"Confidence: {confidence:.2f}%")
