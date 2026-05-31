from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained model
model = load_model("rnn_model.h5")

# Load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LENGTH = 200


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review", "").strip()

    if not review:
        return render_template(
            "result.html",
            prediction="No review text provided."
        )

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([review])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post"
    )

    # Predict sentiment
    prediction_score = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    predicted_class = int(prediction_score >= 0.5)

    sentiment = "Positive" if predicted_class == 1 else "Negative"

    return render_template(
        "result.html",
        prediction=sentiment,
        score=round(float(prediction_score), 4)
    )


if __name__ == "__main__":
    app.run(debug=True)