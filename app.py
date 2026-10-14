import json
import string
from pathlib import Path
import tensorflow as tf
import streamlit as st

folder = Path(__file__).parent

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        folder / "news_rnn.keras", compile=False
    )

model = load_model()

with open(folder / "settings.json", encoding="utf-8") as f:
    settings = json.load(f)

stopwords = set(settings["stopwords"])

vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    vocabulary=settings["vocabulary"],
    output_sequence_length=settings["max_len"]
)

st.title("FlipItNews")
st.write("News classification using a vanilla RNN")

text = st.text_area("Enter a news article", height=200)

if st.button("Predict"):
    text = text.lower()

    for symbol in string.punctuation + string.digits:
        text = text.replace(symbol, " ")

    words = []
    for word in text.split():
        if word not in stopwords:
            words.append(word)

    if len(words) == 0:
        st.warning("Please enter a news article.")
    else:
        clean_text = " ".join(words)
        sequence = vectorizer(tf.constant([clean_text]))

        if not bool(tf.reduce_any(sequence > 1).numpy()):
            st.warning("No words recognized. Try a fuller article.")
        else:
            probabilities = model(sequence, training=False).numpy()[0]
            category = int(probabilities.argmax())
            st.success(settings["categories"][category])
