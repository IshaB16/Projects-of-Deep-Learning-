from pathlib import Path
import pickle
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

ROOT = Path(__file__).parent
CORPUS = ROOT / "data" / "corpus.txt"
MODEL_DIR = ROOT / "model"
SEQ_LEN = 12
EPOCHS = 30

def main():
    if not CORPUS.exists():
        raise FileNotFoundError(f"Add your text corpus at {CORPUS}")
    text = CORPUS.read_text(encoding="utf-8").lower()
    text = re.sub(r"[^a-z0-9.,!?;:'\"()\-\n ]+", " ", text)
    if len(text.split()) < 100:
        raise ValueError("Please use at least 100 words in data/corpus.txt.")
    tokenizer = Tokenizer(oov_token="<OOV>", filters="")
    tokenizer.fit_on_texts([text])
    ids = tokenizer.texts_to_sequences([text])[0]
    vocab_size = len(tokenizer.word_index) + 1
    sequences = []
    for end in range(2, len(ids) + 1):
        seq = ids[max(0, end - (SEQ_LEN + 1)):end]
        if len(seq) >= 2:
            sequences.append(seq)
    padded = pad_sequences(sequences, maxlen=SEQ_LEN + 1, padding="pre")
    X, y = padded[:, :-1], padded[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
    model = Sequential([
        Embedding(vocab_size, 64),
        LSTM(128),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(X, y, epochs=EPOCHS, batch_size=64, validation_split=0.1)
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "text_generator.keras")
    with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open(MODEL_DIR / "config.pkl", "wb") as f:
        pickle.dump({"sequence_length": SEQ_LEN}, f)
    print("Training complete. Artifacts saved in model/")

if __name__ == "__main__":
    main()
