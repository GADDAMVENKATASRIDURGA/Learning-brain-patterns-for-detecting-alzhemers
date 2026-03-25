import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL = load_model("model/cnn.h5")

CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(128,128))
img = image.img_to_array(img)/255.0
img = np.expand_dims(img, axis=0)

pred = MODEL.predict(img)
result = CLASSES[np.argmax(pred)]

print("Prediction:", result)