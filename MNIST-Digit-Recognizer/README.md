# 🚀 Mini Project — Week 1: MNIST Digit Recognizer

A simple end-to-end machine-learning project that:

1. Trains a Convolutional Neural Network (CNN) on the MNIST handwritten-digit dataset.
2. Saves the trained model.
3. Provides a Flask web application with a drawing canvas.
4. Sends the drawn digit to the backend.
5. Preprocesses the drawing into a 28×28 MNIST-style image.
6. Predicts the digit in near real time and displays confidence scores.

## Tech stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pillow
- HTML
- CSS
- JavaScript
- MNIST dataset

## Project structure

```text
mnist_digit_recognizer/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── mnist_model.keras   # created after training
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## 1. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train the neural network

```bash
python train.py
```

The first run downloads MNIST automatically. Training takes a few minutes depending on your computer.

At the end, you should see something similar to:

```text
Test accuracy: 98.xx%
Model saved to: model/mnist_model.keras
```

## 4. Start the web app

```bash
python app.py
```

Open the local address shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

Draw a digit and the prediction will update automatically.

## How the project works

### Training

MNIST contains 28×28 grayscale images of handwritten digits from 0 to 9.

The CNN learns visual patterns such as edges, curves and shapes. The final softmax layer produces 10 probabilities, one for each digit.

### Prediction

The browser sends the canvas image to Flask. The backend:

- converts it to grayscale,
- crops the drawn digit,
- resizes it while preserving its shape,
- places it in a 28×28 image,
- normalizes pixel values,
- passes it to the trained CNN.

The digit with the highest probability becomes the prediction.

## Internship explanation

If your mentor asks "What did you build?", you can say:

> "I built an end-to-end MNIST digit recognition system. I trained a CNN using TensorFlow/Keras, saved the trained model, and connected it to a Flask web application. The frontend has a canvas where the user draws a digit. The backend preprocesses the drawing into the 28×28 format expected by MNIST and returns the predicted digit along with confidence scores."

## Possible improvements

- Add a prediction history.
- Show the processed 28×28 image.
- Add a probability chart.
- Improve preprocessing using centering based on the digit's center of mass.
- Deploy the application online.
- Add a model accuracy/confusion-matrix page.
