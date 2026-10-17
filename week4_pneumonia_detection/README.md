# Week 4 — Pneumonia Detection from Chest X-rays

Educational capstone: classify chest X-ray images as `NORMAL` or `PNEUMONIA` with transfer learning (MobileNetV2).

**Important:** Educational demonstration only. This is not a medical device and must not be used to diagnose or make healthcare decisions.

## Setup (Windows PowerShell)
Use Python 3.10 or 3.11:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dataset
Dataset files are not included in this ZIP because of their size and licensing/usage terms. Obtain a public chest X-ray dataset (for example, the Kermany et al. pediatric chest X-ray dataset) from a legitimate source, review its terms, and arrange it like this:
```text
data/chest_xray/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```
Use the exact folder names above. Do not upload patient data or a dataset to a public repository unless you have permission. Keep the test set untouched during training/tuning.

## Train
```powershell
python train.py --epochs 5 --batch-size 16 --learning-rate 0.0001
```
Quick pipeline smoke test (metrics are not meaningful):
```powershell
python train.py --epochs 1 --batch-size 8 --max-train-batches 5 --max-val-batches 2
```
The default downloads ImageNet MobileNetV2 weights. To skip that download, use `--weights None`.

## Evaluate
```powershell
python evaluate.py
```
Writes metrics and plots to `reports/`: metrics JSON, classification report, confusion matrix, and ROC curve (if both classes exist).

## Streamlit app
```powershell
streamlit run app.py
```
Train the model first. Upload a JPG/PNG chest X-ray and click Analyze image.

## Flask API
```powershell
python api.py
```
Health endpoint: `http://127.0.0.1:5000/health`
PowerShell example:
```powershell
curl.exe -X POST -F "file=@C:\path\to\xray.jpeg" http://127.0.0.1:5000/predict
```

## Docker
Build and run the API:
```powershell
docker build -f Dockerfile.api -t week4-pneumonia-api .
docker run --rm -p 5000:5000 -v "${PWD}\models:/app/models" week4-pneumonia-api
```
Build and run the frontend in another terminal:
```powershell
docker build -f Dockerfile.frontend -t week4-pneumonia-frontend .
docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:5000 -v "${PWD}\models:/app/models" week4-pneumonia-frontend
```
Note: the Streamlit app runs its local model directly; the API is included as a separately deployable service.

## Project files
- `train.py`: model training
- `evaluate.py`: evaluation metrics and plots
- `app.py`: Streamlit interface
- `api.py`: Flask JSON API
- `Dockerfile.api`, `Dockerfile.frontend`: Docker images
- `report_template.md`: professional report outline

CPU training can take time. Reduce batch size to 8 if memory is limited.
