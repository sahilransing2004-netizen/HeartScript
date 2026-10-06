"""
Simple FastAPI backend for HeartScript ECG image classification.
"""
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

CKPT_PATH = "../checkpoints/ecg_image_classifier_best.pt"
IMG_SIZE = 224

app = FastAPI(title="HeartScript ECG Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    label = "abnormal" if probs[1] > probs[0] else "normal"
    return {
        "label": label,
        "confidence": round(probs.max().item(), 4),
        "p_normal": round(probs[0].item(), 4),
        "p_abnormal": round(probs[1].item(), 4),
    }


# ---- Heart risk predictor (UCI tabular model) ----
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent / "risk"))
from predict_risk import predict_risk  # noqa: E402


class RiskInput(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: int = Field(ge=50, le=300)
    chol: int = Field(ge=50, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=40, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


@app.post("/risk")
def risk(data: RiskInput):
    return predict_risk(data.model_dump())
