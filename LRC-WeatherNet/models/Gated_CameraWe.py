# === Fused-RGB Dual Input Gated Fusion Script ===

import os
import json
import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ==== Configuration ====
FUSED_ROOT = "/nfs/home/noualb20/180_LiRAEarly_Sort"
RGB_ROOT = "/nfs/home/noualb20/IMG_224"
OUTPUT_DIR = "/nfs/home/noualb20/111CleanSplit_180/FusedRGB_EffB0_GatedFusion_ResultsWe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

checkpoint_path = os.path.join(OUTPUT_DIR, "best_checkpoint.pth")
csv_path = os.path.join(OUTPUT_DIR, "epoch_metrics.csv")
log_path = os.path.join(OUTPUT_DIR, "training_logs.pkl")
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

sensor_means = [0.04691519, 0.00721895, 0.00405025]
sensor_stds = [0.16557271, 0.25072036, 0.23267177]

# ==== Transforms ====
def normalize_fused(x):
    mean = torch.tensor(sensor_means).view(3, 1, 1)
    std = torch.tensor(sensor_stds).view(3, 1, 1)
    return (x - mean) / (std + 1e-6)

fused_transforms = {
    phase: lambda x: normalize_fused(x.float()) for phase in ['train', 'val', 'test']
}

rgb_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.RandomResizedCrop((224, 224), scale=(0.6, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ==== Dataset ====
class DualInputDataset(Dataset):
    def __init__(self, rgb_root, fused_root, split, transform_fused, transform_rgb):
        self.fused_root = os.path.join(fused_root, split)
        self.rgb_root = os.path.join(rgb_root, split)
        self.transform_fused = transform_fused
        self.transform_rgb = transform_rgb

        self.samples = []
        self.classes = sorted(os.listdir(self.fused_root))

        for cls in self.classes:
            fused_dir = os.path.join(self.fused_root, cls)
            rgb_dir = os.path.join(self.rgb_root, cls)
            fused_files = sorted([f for f in os.listdir(fused_dir) if f.endswith('.pt')])
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
            fused_ts = {f.split('_')[0]: f for f in fused_files}
            rgb_ts = {f.split('_')[0]: f for f in rgb_files}
            common_ts = sorted(set(fused_ts) & set(rgb_ts))

            for ts in common_ts:
                self.samples.append((
                    os.path.join(fused_dir, fused_ts[ts]),
                    os.path.join(rgb_dir, rgb_ts[ts]),
                    manual_class_to_idx[cls]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fused_path, rgb_path, label = self.samples[idx]
        fused_tensor = torch.load(fused_path, map_location='cpu')
        fused_tensor = self.transform_fused(fused_tensor)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_tensor = self.transform_rgb(rgb_img)
        return fused_tensor, rgb_tensor, label

# ==== Gated Fusion Module ====
class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.rgb_mlp = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU())
        self.fused_mlp = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(in_channels * 2, in_channels * 2), nn.Sigmoid())

    def forward(self, rgb_feat, fused_feat):
        rgb_proj = self.rgb_mlp(rgb_feat)
        fused_proj = self.fused_mlp(fused_feat)
        concat = torch.cat([rgb_proj, fused_proj], dim=1)
        gates = self.gate(concat)
        rgb_gate, fused_gate = gates.chunk(2, dim=1)
        return torch.cat([rgb_feat * rgb_gate, fused_feat * fused_gate], dim=1)

# ==== Model ====
class MultiModalFusionModel(nn.Module):
    def __init__(self, fused_in_channels=3, num_classes=9):
        super().__init__()
        self.effnet_fused = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.effnet_rgb = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.effnet_fused.features[0][0] = nn.Conv2d(fused_in_channels, 32, 3, 2, 1, bias=False)
        self.effnet_fused.classifier = nn.Identity()
        self.effnet_rgb.classifier = nn.Identity()
        self.gated_fusion = GatedFusion(in_channels=1280)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, fused_input, rgb_input):
        fused_feat = self.effnet_fused(fused_input)
        rgb_feat = self.effnet_rgb(rgb_input)
        fused = self.gated_fusion(rgb_feat, fused_feat)
        return self.classifier(fused)

# ==== Training & Evaluation Utilities ====
def create_dataloaders():
    return {
        phase: DataLoader(
            DualInputDataset(RGB_ROOT, FUSED_ROOT, phase, fused_transforms[phase], rgb_transforms[phase]),
            batch_size=batch_size, shuffle=(phase == 'train'), num_workers=4
        ) for phase in ['train', 'val', 'test']
    }

def calculate_class_weights(split_dir):
    from collections import Counter
    class_counts = Counter()
    for cls in os.listdir(split_dir):
        class_path = os.path.join(split_dir, cls)
        count = len([f for f in os.listdir(class_path) if f.endswith(".pt")])
        class_counts[manual_class_to_idx[cls]] += count
    counts = torch.tensor([class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float)
    weights = 1. / counts
    weights = weights / weights.sum() * num_classes
    return torch.clamp(weights, max=5.0).to(device)

# ==== Training Loop ====
model = MultiModalFusionModel(num_classes=num_classes).to(device)
dataloaders = create_dataloaders()
class_weights = calculate_class_weights(os.path.join(FUSED_ROOT, "train"))
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

logs = pd.read_pickle(log_path) if os.path.exists(log_path) else {
    'train': {'loss': [], 'acc': [], 'f1': []},
    'val': {'loss': [], 'acc': [], 'f1': []},
    'test': {},
    'lr': []
}

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr"])

start_epoch = 0
best_val_f1 = 0.0

for epoch in range(start_epoch, num_epochs):
    print(f"\n Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_preds, train_labels = [], []

    for fused, rgb, y in tqdm(dataloaders['train'], desc="Training"):
        fused, rgb, y = fused.to(device), rgb.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(fused, rgb)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item() * fused.size(0)
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
        for fused, rgb, y in tqdm(dataloaders['val'], desc="Validating"):
            fused, rgb, y = fused.to(device), rgb.to(device), y.to(device)
            out = model(fused, rgb)
            loss = criterion(out, y)
            val_loss += loss.item() * fused.size(0)
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

    pd.to_pickle(logs, log_path)
    with open(csv_path, "a", newline='') as f:
        csv.writer(f).writerow([epoch + 1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, optimizer.param_groups[0]['lr']])

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), checkpoint_path)

# ==== Final Test ====
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for fused, rgb, y in tqdm(dataloaders['test'], desc="Testing Best Model"):
        fused, rgb, y = fused.to(device), rgb.to(device), y.to(device)
        out = model(fused, rgb)
        test_preds.extend(out.argmax(1).cpu().numpy())
        test_labels.extend(y.cpu().numpy())

acc = np.mean(np.array(test_preds) == np.array(test_labels))
f1 = f1_score(test_labels, test_preds, average='macro')
report = classification_report(test_labels, test_preds, target_names=[idx_to_class[i] for i in range(num_classes)], output_dict=True)
cm = confusion_matrix(test_labels, test_preds)

with open(os.path.join(OUTPUT_DIR, "best_model_test_results.json"), "w") as f:
    json.dump({'acc': acc, 'f1': f1, 'report': report, 'confusion_matrix': cm.tolist()}, f, indent=4)

plt.figure(figsize=(10, 8))
sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[idx_to_class[i] for i in range(num_classes)],
            yticklabels=[idx_to_class[i] for i in range(num_classes)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, "normalized_confusion_matrix.png"))
plt.close()
