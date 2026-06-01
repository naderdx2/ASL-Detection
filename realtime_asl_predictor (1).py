import json
from collections import deque, Counter

import cv2
import numpy as np

import torch
import torch.nn as nn
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# =========================
# SETTINGS
# =========================

MODEL_PATH = "models/best_asl_model.pth"
LABELS_PATH = "models/labels.json"

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# LOAD LABELS
# =========================

with open(LABELS_PATH, "r") as f:

    labels = json.load(f)

# =========================
# MODEL
# =========================

class ASLModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(63, 256),

            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, num_classes)

        )

    def forward(self, x):

        return self.network(x)

# =========================
# LOAD MODEL
# =========================

model = ASLModel(len(labels)).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully!")

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils

mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    model_complexity=1,

    min_detection_confidence=0.7,

    min_tracking_confidence=0.7

)

# =========================
# PREDICTION SMOOTHING
# =========================

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

def stable_prediction(buffer):

    if not buffer:

        return None

    return Counter(buffer).most_common(1)[0][0]

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError("Could not open camera")

print("Press q to quit")

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    shown_text = "No hand detected"

    shown_conf = ""

    if results.multi_hand_landmarks:

        hand_lms = results.multi_hand_landmarks[0]

        # =========================
        # DRAW LANDMARKS
        # =========================

        mp_draw.draw_landmarks(

            display,

            hand_lms,

            mp_hands.HAND_CONNECTIONS,

            mp_styles.get_default_hand_landmarks_style(),

            mp_styles.get_default_hand_connections_style()

        )

        # =========================
        # EXTRACT LANDMARKS
        # =========================

        landmarks = []

        for lm in hand_lms.landmark:

            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        # =========================
        # CONVERT TO TENSOR
        # =========================

        input_tensor = torch.tensor(
            landmarks,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        # =========================
        # PREDICTION
        # =========================

        with torch.no_grad():

            outputs = model(input_tensor)

            probs = torch.softmax(outputs, dim=1)

            conf, pred = torch.max(probs, 1)

            conf = conf.item()

            pred = pred.item()

        predicted_label = labels[pred]

        # =========================
        # CONFIDENCE FILTER
        # =========================

        if conf >= CONFIDENCE_THRESHOLD:

            prediction_buffer.append(predicted_label)

        stable = stable_prediction(prediction_buffer)

        if stable:

            shown_text = f"Letter: {stable}"

        else:

            shown_text = "Reading..."

        shown_conf = f"Prediction: {predicted_label} | Confidence: {conf:.2f}"

    # =========================
    # UI TEXT
    # =========================

    cv2.putText(

        display,

        shown_text,

        (20, 45),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.1,

        (0, 255, 0),

        2

    )

    cv2.putText(

        display,

        shown_conf,

        (20, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 255, 255),

        2

    )

    cv2.putText(

        display,

        "MediaPipe + PyTorch Hybrid Model",

        (20, display.shape[0] - 45),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (220, 220, 220),

        2

    )

    cv2.putText(

        display,

        "Press q to quit",

        (20, display.shape[0] - 15),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (220, 220, 220),

        2

    )

    # =========================
    # SHOW WINDOW
    # =========================

    cv2.imshow(

        "ASL Hybrid Recognition",

        display

    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# =========================
# CLEANUP
# =========================

cap.release()

cv2.destroyAllWindows()