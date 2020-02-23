import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import sys

app = Flask(__name__)

# 機械学習モデルの読込
model = load_model("./data/cnn_model.h5")
model.load_weights("./data/cnn_weights.h5")

graph = tf.get_default_graph()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=['POST'])
def result():
    if request.files["image"]:
        img_file = request.files["image"]
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))

        in_rows = 32
        in_cols = 32
        in_colors = 3

        labels = [
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        ]

        img = img.reshape(-1, in_rows, in_cols,
                          in_colors).astype("float32") / 255

        with graph.as_default():
            r = model.predict(img, batch_size=32, verbose=1)
            res = r[0]
            print(res, file=sys.stdout)
        return render_template("result.html", res=res, labels=labels)
    else:
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
