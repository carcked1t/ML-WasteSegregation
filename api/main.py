from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from PIL import Image
import torchvision.transforms as T
import json
from datetime import datetime
import time
import os
from typing import List

app = FastAPI(title="Waste Segregation API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # limit in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Classes ---
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# --- Load TorchScript Model ---
model_path = "waste_model_scripted.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = torch.jit.load(model_path, map_location=device)
model.to(device)
model.eval()

# --- Warm-up ---
with torch.no_grad():
    dummy = torch.zeros(1, 3, 160, 160).to(device)
    model(dummy)
print("[INFO] Model loaded and warmed up!")

# --- Transformations ---
transform = T.Compose([
    T.Resize((160, 160)),  # match training size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225])
])

# --- Logging ---
LOG_FILE = "predictions_log.json"

def log_prediction(data: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def home_html():
    html_path = "api/templates/index.html"
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return "<h1>Waste Segregation API Running</h1>"

@app.get("/status")
def status():
    return {"status": "Waste Segregation API Running"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []
    start_time = time.time()

    for file in files:
        try:
            image = Image.open(file.file).convert("RGB")
        except Exception as e:
            results.append({"error": f"Failed to open file {file.filename}: {str(e)}"})
            continue

        img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        result = {
            "filename": file.filename,
            "class_id": int(pred.item()),
            "class_name": CLASSES[pred.item()],
            "confidence": round(float(conf.item()), 4)
        }
        log_prediction(result)
        results.append(result)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "predictions": results,
        "latency_ms": latency_ms
    }
