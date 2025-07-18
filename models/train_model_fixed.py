print("🚨 USING THIS train_model_fixed.py FILE")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore")  # Ignore all warnings

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# 🔧 Config
epochs = 45
batch_size = 16
learning_rate = 1e-4
save_dir = "models"
save_path = os.path.join(save_dir, "resnet18_neu.pth")

# 📁 Paths
train_dir = "data/NEU/raw/train/images"
val_dir = "data/NEU/raw/validation/images"

# 🧼 Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📂 Dataset and DataLoader
print("🗂️ Loading datasets...")
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = train_dataset.classes
print("✅ DataLoaders ready.")

# 🧠 Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 6)
model = model.to(device)

# ⚙️ Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 🚀 Training
print("🚀 Starting training...")
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / num_batches
    train_acc = correct / total

    # 🔍 Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    end_time = time.time()
    epoch_duration = end_time - start_time
    seconds = int(epoch_duration)
    millis = int((epoch_duration - seconds) * 1000)

    print(f"Epoch {epoch+1}/{epochs}   ⏱ {seconds}s {millis}ms/step   "
          f"- accuracy: {train_acc:.4f}   - loss: {train_loss:.4f}   "
          f"- val_accuracy: {val_acc:.4f}   - val_loss: {val_loss:.4f}")

# 💾 Save model
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")

# ✅ Accuracy
accuracy = 100 * val_acc
print(f"✅ Final Validation Accuracy: {accuracy:.2f}%")
with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
    f.write(f"{accuracy:.2f}")

# 📊 Confusion Matrix
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Raw Counts)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()
print("📊 Confusion matrix saved to models/confusion_matrix.png")

# 📦 Save class distribution
label_counts = Counter(all_labels)
class_dist = {class_names[k]: v for k, v in label_counts.items()}
with open(os.path.join(save_dir, "class_distribution.json"), "w") as f:
    json.dump(class_dist, f)
print("📦 Class distribution saved.")
print("🎉 Training complete!")




