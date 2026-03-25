from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

MODEL = load_model("model/cnn.h5")
CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join("static", file.filename)
        file.save(path)

        img = image.load_img(path, target_size=(128,128))
        img = image.img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = MODEL.predict(img)
        result = CLASSES[np.argmax(pred)]

        return render_template("result.html",
                               prediction=result,
                               img=file.filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

