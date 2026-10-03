from pathlib import Path
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"model/plant_disease_model.keras"
LABELS=ROOT/"model/class_names.json"
app=Flask(__name__)
model=load_model(MODEL) if MODEL.exists() else None
classes=json.loads(LABELS.read_text()) if LABELS.exists() else []

@app.route("/",methods=["GET","POST"])
def home():
    result=None; confidence=None; error=None
    if request.method=="POST":
        f=request.files.get("leaf_image")
        if not f or not f.filename: error="Choose a leaf image first."
        elif model is None: error="Model missing. Follow README.md to train it."
        else:
            try:
                im=Image.open(f).convert("RGB").resize((224,224))
                x=np.asarray(im,dtype=np.float32)[None,...]/255
                p=model.predict(x,verbose=0)[0]; i=int(np.argmax(p))
                result=classes[i].replace("___"," — ").replace("_"," ")
                confidence=float(p[i])*100
            except Exception: error="Could not read this image. Try a JPG or PNG."
    return render_template("index.html",prediction=result,confidence=confidence,error=error)
if __name__=="__main__": app.run(debug=True)
