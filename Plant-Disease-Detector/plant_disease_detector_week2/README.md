# Plant Disease Detector — Week 2

A Flask image-upload app using transfer learning with MobileNetV2. It predicts a class from the dataset and displays the model's confidence score. This is an educational prototype; predictions can be wrong and are not expert agricultural advice.

## Dataset
Use an open plant-leaf image dataset such as PlantVillage. Check the host and license terms. Put images in class-named folders directly inside `dataset/`, for example:
```
dataset/
  Tomato___Early_blight/
  Tomato___healthy/
  Apple___Apple_scab/
```
Each folder should contain images for that class. Do not upload the dataset to GitHub unless redistribution is permitted.

## Setup (PowerShell)
Use Python 3.10 or 3.11 in a virtual environment:
```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python app.py
```
Open http://127.0.0.1:5000. Training uses pretrained ImageNet MobileNetV2 features, trains a classifier, then fine-tunes part of the backbone. It saves the model and class labels in `model/`. The first training run may download pretrained weights.

## Workflow
1. User uploads a leaf photo.
2. Flask converts it to RGB and resizes to 224×224.
3. The image is scaled and passed to the neural network.
4. The app shows the highest-probability class and confidence percentage.

Confidence is the model's output probability, not a guarantee of correctness.

## Internship explanation
“I used transfer learning with MobileNetV2 rather than training a large model from scratch. I trained a classification head on leaf classes, fine-tuned some pretrained layers, and connected the model to a Flask upload interface that returns a class prediction and confidence score.”

## Improvements
Try top-three predictions, a confusion matrix, more varied test photos, image preview, and comparisons with other pretrained models.
