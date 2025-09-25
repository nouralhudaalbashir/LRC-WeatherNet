import os
os.environ['TORCH_HOME'] = '/nfs/home/noualb20/torch_cache'

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
import csv

# ==== Configuration ====
output_dir = "/nfs/home/noualb20/111CleanSplit_180/224RGB_9Class_Results"
os.makedirs(output_dir, exist_ok=True)

train_directory = '/nfs/home/noualb20/IMG_224/train'
val_directory = '/nfs/home/noualb20/IMG_224/val'
test_directory = '/nfs/home/noualb20/IMG_224/test'

checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
csv_log_path = os.path.join(output_dir, "epoch_metrics.csv")
log_path = os.path.join(output_dir, "training_logs.pkl")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_epochs = 30

# Manual class-to-index mapping
manual_class_to_idx = {
    '2022_rain': 0, '2022_spring': 1, '2022_spring_snow': 2,
    '2023_early_fall': 3, '2023_fall_sunset': 4, '2023_fall_sunset_2': 5,
    '2023_late_summer': 6, '2023_snow': 7, '2023_neighborhood_fall': 8
}
idx_to_class = {v: k for k, v in manual_class_to_idx.items()}
num_classes = len(manual_class_to_idx)

# ==== Transforms ====
img_height, img_width = 224, 224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.RandomResizedCrop((img_height, img_width), scale=(0.6, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ==== Dataset Loading ====
train_dataset = ImageFolder(train_directory, data_transforms['train'])
val_dataset = ImageFolder(val_directory, data_transforms['val'])
test_dataset = ImageFolder(test_directory, data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==== Model ====
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# ==== Class Weights ====
class_counts = [len(os.listdir(os.path.join(train_directory, cls))) for cls in sorted(manual_class_to_idx)]
counts = torch.tensor(class_counts, dtype=torch.float)
weights = 1. / counts
weights = weights / weights.sum() * num_classes
class_weights = torch.clamp(weights, max=5.0).to(device)

# ==== Training Setup ====
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

logs = pd.read_pickle(log_path) if os.path.exists(log_path) else {
    'train': {'loss': [], 'acc': [], 'f1': []},
    'val': {'loss': [], 'acc': [], 'f1': []},
    'test': {},
    'lr': []
}

start_epoch = 0
best_val_f1 = 0.0

# CSV Logging
if not os.path.exists(csv_log_path):
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1', 'lr'])

# ==== Training Loop ====
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_preds, train_labels = [], []

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
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
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Validating"):
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

    with open(csv_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, optimizer.param_groups[0]['lr']])

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), checkpoint_path)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

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