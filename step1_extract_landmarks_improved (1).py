import os
import csv
import cv2
import mediapipe as mp

# =========================
# DATASET PATH
# =========================

DATASET_DIR = "ASL_Alphabet_Dataset/asl_alphabet_train"

CSV_FILE = "asl_landmarks.csv"

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# =========================
# CREATE CSV HEADER
# =========================

header = ["label"]

for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

with open(CSV_FILE, mode="w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(header)

# =========================
# PROCESS DATASET
# =========================

total_images = 0
saved_samples = 0
failed_images = 0

for label in os.listdir(DATASET_DIR):

    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nProcessing class: {label}")

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        total_images += 1

        image = cv2.imread(image_path)

        if image is None:

            failed_images += 1

            continue

        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe hand detection
        results = hands.process(rgb_image)

        # =========================
        # IF HAND DETECTED
        # =========================

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []

            # Extract 21 landmarks
            for lm in hand_landmarks.landmark:

                landmarks.extend([lm.x, lm.y, lm.z])

            # Create row
            row = [label] + landmarks

            # Save to CSV
            with open(CSV_FILE, mode="a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow(row)

            saved_samples += 1

        else:

            failed_images += 1

# =========================
# FINISHED
# =========================

print("\n========== DONE ==========")

print(f"Total images: {total_images}")

print(f"Saved samples: {saved_samples}")

print(f"Failed detections: {failed_images}")

hands.close()