"""
ASL Detection API
-----------------
FastAPI server that exposes the trained PyTorch landmark model
as a RESTful Web API.

Endpoints:
  GET  /              - Health check
  GET  /info          - Model info and available labels
  POST /predict       - Predict ASL letter from uploaded image
  POST /predict/landmarks - Predict from raw landmark coordinates
  DELETE /session     - Clear server-side session state
"""

import json
import io
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# =========================
# SETTINGS
# =========================
MODEL_PATH  = "models/best_asl_model.pth"
LABELS_PATH = "models/labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL
# =========================
class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.network(x)

# =========================
# LOAD MODEL + LABELS
# =========================
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

model = ASLModel(len(labels)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✅ Model loaded | Device: {DEVICE} | Classes: {len(labels)}")

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="ASL Detection API",
    description="Real-time American Sign Language detection using MediaPipe + PyTorch MLP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SCHEMAS
# =========================
class LandmarkRequest(BaseModel):
    landmarks: List[float]  # exactly 63 floats (21 points × x,y,z)

class PredictionResponse(BaseModel):
    predicted_letter: str
    confidence: float
    top3: List[dict]
    device: str
    inference_time_ms: float

# =========================
# HELPER: run model on landmarks
# =========================
def predict_from_landmarks(landmark_array: np.ndarray) -> dict:
    start = time.time()
    tensor = torch.tensor(landmark_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    elapsed = (time.time() - start) * 1000

    pred_idx    = int(np.argmax(probs))
    confidence  = float(probs[pred_idx])
    top3_idx    = np.argsort(probs)[::-1][:3]
    top3        = [{"letter": labels[i], "confidence": float(probs[i])} for i in top3_idx]

    return {
        "predicted_letter": labels[pred_idx],
        "confidence": round(confidence, 4),
        "top3": top3,
        "device": str(DEVICE),
        "inference_time_ms": round(elapsed, 2)
    }

# =========================
# ROUTES
# =========================

@app.get("/", summary="Health Check")
def health_check():
    """Returns API status and basic info."""
    return {
        "status": "ok",
        "message": "ASL Detection API is running",
        "model": "PyTorch MLP — MediaPipe Landmarks",
        "device": str(DEVICE),
        "num_classes": len(labels)
    }


@app.get("/info", summary="Model Info")
def model_info():
    """Returns model details and list of all supported ASL classes."""
    return {
        "model_architecture": "MLP (63→256→128→64→N)",
        "input_features": 63,
        "num_classes": len(labels),
        "labels": labels,
        "device": str(DEVICE),
        "confidence_threshold": 0.70
    }


@app.post("/predict", summary="Predict from Image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Upload a webcam frame (JPG/PNG).
    MediaPipe extracts hand landmarks, model predicts the ASL letter.
    Returns predicted letter, confidence, and top-3 predictions.
    """
    # Read image
    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Extract landmarks with MediaPipe
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    if not results.multi_hand_landmarks:
        raise HTTPException(status_code=422, detail="No hand detected in the image.")

    hand_lms    = results.multi_hand_landmarks[0]
    landmark_arr = []
    for lm in hand_lms.landmark:
        landmark_arr.extend([lm.x, lm.y, lm.z])

    result = predict_from_landmarks(np.array(landmark_arr, dtype=np.float32))
    return JSONResponse(content=result)


@app.post("/predict/landmarks", summary="Predict from Landmarks")
def predict_from_landmark_input(request: LandmarkRequest):
    """
    Send 63 raw landmark values (21 points × x,y,z) directly.
    Faster than /predict — no MediaPipe needed server-side.
    Useful when the client runs MediaPipe locally.
    """
    if len(request.landmarks) != 63:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 63 landmark values, got {len(request.landmarks)}."
        )
    arr    = np.array(request.landmarks, dtype=np.float32)
    result = predict_from_landmarks(arr)
    return JSONResponse(content=result)


@app.delete("/session", summary="Clear Session")
def clear_session():
    """Clears any server-side state (for future session support)."""
    return {"status": "ok", "message": "Session cleared."}


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
