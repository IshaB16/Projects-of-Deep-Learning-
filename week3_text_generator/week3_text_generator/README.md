# Week 3 — LSTM Text Generation

An educational word-level LSTM text generator with a Streamlit interface. Users enter a starting phrase, choose temperature, and set the maximum number of generated words.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version supported by your TensorFlow installation. If installation fails on Python 3.13, try Python 3.11 with a compatible TensorFlow release.

## Prepare the corpus

Edit `data/corpus.txt`. It needs at least 100 words; a larger corpus generally works better. Use your own writing, public-domain material, or text with a license that permits this use. Do not use confidential text or copyrighted lyrics without permission.

## Train the model

```powershell
python train.py
```

This creates model artifacts in `model/`.

## Run the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## How it works

1. A tokenizer maps words to integer IDs.
2. Training sequences teach the model to predict the next word.
3. An Embedding layer represents words as vectors.
4. An LSTM learns sequential patterns.
5. Temperature adjusts how predictable or varied the sampling is.

This is a small learning project, not a production language model. Output can be repetitive or nonsensical, especially with a small corpus.
