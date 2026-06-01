import os
import json
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CREATE FOLDERS
# =========================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# SETTINGS
# =========================

CSV_FILE = "asl_landmarks.csv"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

MODEL_PATH = "models/best_asl_model.pth"
LABELS_PATH = "models/labels.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# LOAD DATA
# =========================

data = pd.read_csv(CSV_FILE)

# أول عمود = label
# باقي الأعمدة = landmarks

X = data.iloc[:, 1:].values.astype(np.float32)

y = data.iloc[:, 0].values

# =========================
# CHECK CLASS DISTRIBUTION
# =========================

print("\n===== CLASS DISTRIBUTION =====\n")

counts = Counter(y)

for cls, count in counts.items():

    print(cls, ":", count)

# حذف الكلاسات اللي عددها قليل جدًا
valid_classes = [cls for cls, count in counts.items() if count >= 10]

data = data[data.iloc[:, 0].isin(valid_classes)]

# تحديث البيانات بعد الحذف
X = data.iloc[:, 1:].values.astype(np.float32)

y = data.iloc[:, 0].values

# =========================
# LABEL ENCODING
# =========================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

class_names = encoder.classes_

# حفظ أسماء الكلاسات
with open(LABELS_PATH, "w") as f:

    json.dump(class_names.tolist(), f)

# =========================
# TRAIN / VALIDATION / TEST SPLIT
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# =========================
# DATASET CLASS
# =========================

class ASLDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X)

        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

# =========================
# DATASETS
# =========================

train_dataset = ASLDataset(X_train, y_train)

val_dataset = ASLDataset(X_val, y_val)

test_dataset = ASLDataset(X_test, y_test)

# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

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

model = ASLModel(len(class_names)).to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# =========================
# TRAINING
# =========================

best_accuracy = 0

train_losses = []

train_accuracies = []

val_accuracies = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    # =========================
    # TRAIN METRICS
    # =========================

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = correct / total

    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for val_inputs, val_labels in val_loader:

            val_inputs = val_inputs.to(DEVICE)

            val_labels = val_labels.to(DEVICE)

            val_outputs = model(val_inputs)

            _, val_preds = torch.max(val_outputs, 1)

            val_total += val_labels.size(0)

            val_correct += (val_preds == val_labels).sum().item()

    val_acc = val_correct / val_total

    # =========================
    # SAVE METRICS
    # =========================

    train_losses.append(epoch_loss)

    train_accuracies.append(epoch_acc)

    val_accuracies.append(val_acc)

    # =========================
    # PRINT RESULTS
    # =========================

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Acc: {epoch_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # =========================
    # LR SCHEDULER
    # =========================

    scheduler.step(val_acc)

    # =========================
    # SAVE EVERY EPOCH
    # =========================

    epoch_path = f"models/asl_epoch_{epoch+1}.pth"

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": val_acc
    }, epoch_path)

    # =========================
    # SAVE BEST MODEL
    # =========================

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": val_acc
        }, MODEL_PATH)

        print("Best model saved!")

# =========================
# SAVE TRAINING CURVES
# =========================

plt.figure(figsize=(8, 6))

plt.plot(train_losses)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig("results/loss_curve.png")

plt.close()

plt.figure(figsize=(8, 6))

plt.plot(train_accuracies, label="Train Accuracy")

plt.plot(val_accuracies, label="Validation Accuracy")

plt.title("Accuracy Curve")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("results/accuracy_curve.png")

plt.close()

# =========================
# LOAD BEST MODEL
# =========================

checkpoint = torch.load(MODEL_PATH)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# =========================
# EVALUATION FUNCTION
# =========================

def evaluate_model(loader, split_name):

    model.eval()

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.numpy())

    # =========================
    # ACCURACY
    # =========================

    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\n{split_name} Accuracy: {accuracy:.4f}")

    # =========================
    # CLASSIFICATION REPORT
    # =========================

    report = classification_report(
        all_labels,
        all_preds,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    with open(f"results/{split_name}_classification_report.txt", "w") as f:

        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        f.write(report)

    # =========================
    # CONFUSION MATRIX
    # =========================

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(18, 15))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title(f"{split_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(f"results/{split_name}_confusion_matrix.png")

    plt.close()

# =========================
# RUN EVALUATION
# =========================

evaluate_model(train_loader, "train")

evaluate_model(val_loader, "validation")

evaluate_model(test_loader, "test")

print("\nTraining Finished Successfully!")

print("Best Validation Accuracy:", best_accuracy)
