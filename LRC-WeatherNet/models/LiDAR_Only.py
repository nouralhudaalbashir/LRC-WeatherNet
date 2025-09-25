# Complete script for LiDAR-only classification using EfficientNet-B0 with ImageNet pretrained weights

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ==== Configuration ====
base_dir = "/nfs/home/noualb20/LiDAR_Radar_Fusion/Rebuilt_LiDAR_Camera_Sorted"
output_dir = "/nfs/home/noualb20/111CleanSplit_180/Pre_OUTPUT_LiDAR180"
os.makedirs(output_dir, exist_ok=True)

checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
batch_size = 64
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manual_class_to_idx = {
    '2022_rain': 0, '2022_spring': 1, '2022_spring_snow': 2,
    '2023_early_fall': 3, '2023_fall_sunset': 4, '2023_fall_sunset_2': 5,
    '2023_late_summer': 6, '2023_snow': 7, '2023_neighborhood_fall': 8
}
idx_to_class = {v: k for k, v in manual_class_to_idx.items()}
num_classes = len(manual_class_to_idx)

lidar_means = [0.04712716]
lidar_stds = [0.16591522]

# ==== Utility Functions ====
def normalize_tensor(x):
    mean = torch.tensor(lidar_means, device=x.device).view(1, 1, 1, 1)
    std = torch.tensor(lidar_stds, device=x.device).view(1, 1, 1, 1)
    return (x - mean) / (std + 1e-6)

def apply_augmentations(x):
    if random.random() > 0.5:
        angle = random.uniform(-5, 5)
        theta = torch.tensor([
            [[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
             [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]]
        ], dtype=torch.float, device=x.device).expand(x.size(0), -1, -1)
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False)
    if random.random() > 0.5:
        x = x + torch.randn_like(x) * 0.02 * x
    return x

def calculate_class_weights(split_dir):
    class_counts = Counter()
    for class_folder in os.listdir(split_dir):
        class_idx = manual_class_to_idx[class_folder]
        class_path = os.path.join(split_dir, class_folder)
        for fname in os.listdir(class_path):
            class_counts.update([class_idx])
    counts = torch.tensor([class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float)
    weights = 1. / counts
    weights = weights / weights.sum() * num_classes
    return torch.clamp(weights, max=5.0).to(device)

class SingleFileDataset(Dataset):
    def __init__(self, split_dir, augment=False):
        self.samples = []
        for class_folder in os.listdir(split_dir):
            class_idx = manual_class_to_idx[class_folder]
            class_path = os.path.join(split_dir, class_folder)
            for fname in os.listdir(class_path):
                if fname.endswith('.pt'):
                    self.samples.append((os.path.join(class_path, fname), class_idx))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        x = torch.load(filepath, map_location='cpu')
        x = normalize_tensor(x.unsqueeze(0))
        if self.augment:
            x = apply_augmentations(x)
        return x.squeeze(0), torch.tensor(label)

def build_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=model.features[0][0].out_channels,
        kernel_size=model.features[0][0].kernel_size,
        stride=model.features[0][0].stride,
        padding=model.features[0][0].padding,
        bias=False
    )
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

# ==== Training Setup ====
model = build_model()
class_weights = calculate_class_weights(os.path.join(base_dir, "train"))
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

train_dataset = SingleFileDataset(os.path.join(base_dir, "train"), augment=True)
val_dataset = SingleFileDataset(os.path.join(base_dir, "val"), augment=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

log_path = os.path.join(output_dir, "training_logs.pkl")
csv_path = os.path.join(output_dir, "epoch_metrics.csv")
logs = pd.read_pickle(log_path) if os.path.exists(log_path) else {
    'train': {'loss': [], 'acc': [], 'f1': []},
    'val': {'loss': [], 'acc': [], 'f1': []},
    'test': {},
    'lr': []
}

metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr"])

start_epoch = 0
best_val_f1 = 0.0

# ==== Training Loop ====
for epoch in range(start_epoch, num_epochs):
    print(f"\n Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_preds, train_labels = [], []

    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += y.size(0)
        train_preds.extend(out.argmax(1).cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    train_loss /= train_total
    train_acc = train_correct / train_total
    train_f1 = f1_score(train_labels, train_preds, average='macro')

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item() * x.size(0)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)
            val_preds.extend(out.argmax(1).cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_loss /= val_total
    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    scheduler.step(val_loss)

    logs['train']['loss'].append(train_loss)
    logs['train']['acc'].append(train_acc)
    logs['train']['f1'].append(train_f1)
    logs['val']['loss'].append(val_loss)
    logs['val']['acc'].append(val_acc)
    logs['val']['f1'].append(val_f1)
    logs['lr'].append(optimizer.param_groups[0]['lr'])

    metrics_df.loc[len(metrics_df)] = [
        epoch + 1, train_loss, train_acc, train_f1,
        val_loss, val_acc, val_f1, optimizer.param_groups[0]['lr']
    ]
    metrics_df.to_csv(csv_path, index=False)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), checkpoint_path)

    print(f"\n Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f" Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

pd.to_pickle(logs, log_path)

# ==== Final Evaluation ====
test_dataset = SingleFileDataset(os.path.join(base_dir, "test"))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing Best Model"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        test_preds.extend(out.argmax(1).cpu().numpy())
        test_labels.extend(y.cpu().numpy())

acc = np.mean(np.array(test_preds) == np.array(test_labels))
f1 = f1_score(test_labels, test_preds, average='macro')
report = classification_report(test_labels, test_preds, target_names=[idx_to_class[i] for i in range(num_classes)], output_dict=True)
cm = confusion_matrix(test_labels, test_preds).tolist()

results = {
    'acc': acc,
    'f1': f1,
    'report': report,
    'confusion_matrix': cm
}

with open(os.path.join(output_dir, "best_model_test_results.json"), "w") as f:
    json.dump(results, f, indent=4)

# ==== Plots ====
logs = pd.read_pickle(log_path)

plt.figure()
plt.plot(logs['train']['loss'], label='Train Loss')
plt.plot(logs['val']['loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))

plt.figure()
plt.plot(logs['train']['acc'], label='Train Accuracy')
plt.plot(logs['val']['acc'], label='Val Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, "val_train_accuracy.png"))

plt.figure()
plt.plot(logs['train']['f1'], label='Train F1')
plt.plot(logs['val']['f1'], label='Val F1')
plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.title('F1 Score Curve')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, "F1_Score.png"))

cm_normalized = confusion_matrix(test_labels, test_preds, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[idx_to_class[i] for i in range(num_classes)],
            yticklabels=[idx_to_class[i] for i in range(num_classes)])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(output_dir, "normalized_confusion_matrix.png"))
plt.close('all')