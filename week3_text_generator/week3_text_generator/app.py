from pathlib import Path
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="LSTM Text Generator", page_icon="✍️")
st.title("✍️ LSTM Text Generator")
st.write("Generate creative text from a word-level LSTM model.")

@st.cache_resource
def load_artifacts():
    paths = [MODEL_DIR / "text_generator.keras", MODEL_DIR / "tokenizer.pkl", MODEL_DIR / "config.pkl"]
    if not all(p.exists() for p in paths):
        return None
    model = tf.keras.models.load_model(paths[0])
    with open(paths[1], "rb") as f:
        tokenizer = pickle.load(f)
    with open(paths[2], "rb") as f:
        config = pickle.load(f)
    return model, tokenizer, config

def sample_word(probs, temperature):
    probs = np.asarray(probs, dtype=np.float64)
    logits = np.log(np.maximum(probs, 1e-8)) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def generate(model, tokenizer, seq_len, prompt, count, temperature):
    output = prompt.strip()
    for _ in range(count):
        ids = tokenizer.texts_to_sequences([output])[0][-seq_len:]
        ids = pad_sequences([ids], maxlen=seq_len, padding="pre")
        probs = model.predict(ids, verbose=0)[0]
        next_id = sample_word(probs, temperature)
        word = tokenizer.index_word.get(next_id)
        if not word:
            break
        output += " " + word
    return output

artifacts = load_artifacts()
if artifacts is None:
    st.warning("Model files are missing. Train the model first.")
    st.code("python train.py")
    st.stop()

model, tokenizer, config = artifacts
prompt = st.text_input("Starting text", "the morning")
temperature = st.slider("Temperature", 0.2, 1.5, 0.8, 0.1)
max_words = st.slider("Maximum new words", 10, 100, 40, 5)
if st.button("Generate text", type="primary"):
    if not prompt.strip():
        st.error("Enter a starting phrase first.")
    else:
        with st.spinner("Generating text..."):
            result = generate(model, tokenizer, config["sequence_length"], prompt, max_words, temperature)
        st.subheader("Generated text")
        st.write(result)
        st.caption("Small training corpora often produce repetitive or unexpected text.")
