import os
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Class Mapping ===
manual_class_to_idx = {
    '2022_rain': 0, '2022_spring': 1, '2022_spring_snow': 2,
    '2023_early_fall': 3, '2023_fall_sunset': 4, '2023_fall_sunset_2': 5,
    '2023_late_summer': 6, '2023_snow': 7, '2023_neighborhood_fall': 8
}

# === Safe Normalization Helper ===
def normalize_tensor(tensor, mean, std, target_c):
    tensor = tensor[:, :224, :224]
    c, h, w = tensor.shape
    if c < target_c:
        padded = torch.zeros((target_c, h, w), dtype=tensor.dtype)
        padded[:c] = tensor
        tensor = padded
    elif c > target_c:
        tensor = tensor[:target_c]
    return transforms.Normalize(mean[:target_c], std[:target_c])(tensor)

# === Dataset ===
class DualEfficientNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform_fused=None, transform_radar=None, class_to_idx=None):
        self.samples = []
        self.transform_fused = transform_fused
        self.transform_radar = transform_radar
        self.class_to_idx = class_to_idx or manual_class_to_idx

        split_dir = Path(root_dir) / split
        for cls in self.class_to_idx.keys():
            fused_dir = split_dir / cls / 'fused'
            radar_dir = split_dir / cls / 'radar'
            if not fused_dir.exists() or not radar_dir.exists():
                continue

            fused_files = list(fused_dir.glob("*.pt"))
            radar_files = list(radar_dir.glob("*.pt"))

            fused_map = {f.stem.split('_')[0]: f for f in fused_files}
            radar_map = {r.stem.split('_')[0]: r for r in radar_files}

            common_ts = set(fused_map.keys()) & set(radar_map.keys())
            for ts in common_ts:
                self.samples.append((fused_map[ts], radar_map[ts], self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fused_path, radar_path, label = self.samples[idx]
        fused = torch.load(fused_path).float()
        radar = torch.load(radar_path).float()

        # Ensure CxHxW format
        if fused.dim() == 3 and fused.shape[-1] in [2, 3]:
            fused = fused.permute(2, 0, 1)
        if radar.dim() == 3 and radar.shape[-1] in [2, 3]:
            radar = radar.permute(2, 0, 1)

        if self.transform_fused:
            fused = self.transform_fused(fused)
        if self.transform_radar:
            radar = self.transform_radar(radar)

        return fused, radar, label

# === Gated Fusion ===
class ChannelGatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        combined = torch.cat([feat1, feat2], dim=1)
        gate = self.attention(combined)
        return feat1 * gate + feat2 * (1 - gate)

# === Model ===
class DualEfficientNetFusionModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.fused_net = timm.create_model('efficientnet_b0', in_chans=3, pretrained=True, features_only=False)
        self.radar_net = timm.create_model('efficientnet_b0', in_chans=2, pretrained=True, features_only=False)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(1280, 512)
        self.gating = ChannelGatedFusion(feature_dim=512)

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, fused_input, radar_input):
        fused_feat = self.fused_net.forward_features(fused_input)
        radar_feat = self.radar_net.forward_features(radar_input)

        fused_vec = self.project(self.pool(fused_feat).flatten(1))
        radar_vec = self.project(self.pool(radar_feat).flatten(1))

        fused_out = self.gating(fused_vec, radar_vec)
        return self.classifier(fused_out)

# === Dataloaders ===
def create_dataloaders(root_dir, batch_size=32):
    fused_mean = [0.04691519, 0.00721895, 0.00405025]
    fused_std = [0.16557271, 0.25072036, 0.23267177]
    radar_mean = fused_mean[:2]
    radar_std = fused_std[:2]

    normalize_fused = lambda t: normalize_tensor(t, fused_mean, fused_std, target_c=3)
    normalize_radar = lambda t: normalize_tensor(t, radar_mean, radar_std, target_c=2)

    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = DualEfficientNetDataset(
            root_dir=root_dir,
            split=split,
            transform_fused=normalize_fused,
            transform_radar=normalize_radar,
            class_to_idx=manual_class_to_idx
        )
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=4)
    return dataloaders

# === Training & Evaluation ===
def train_val_test(model, dataloaders, device, num_epochs=30, output_dir="checkpoints",
                   csv_path="log.csv", test_log_path="test_report.csv"):
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_weights = 0, None
    log_rows = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        row = {'epoch': epoch + 1}

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            y_true, y_pred, losses = [], [], []
            for fused, radar, labels in tqdm(dataloaders[phase], desc=phase.upper()):
                fused, radar, labels = fused.to(device), radar.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(fused, radar)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                preds = outputs.argmax(1)
                losses.append(loss.item())
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            row[f'{phase}_loss'] = sum(losses)/len(losses)
            row[f'{phase}_acc'] = acc
            row[f'{phase}_f1'] = f1
            print(f"{phase.upper()} — Loss: {row[f'{phase}_loss']:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

            if phase == 'val' and f1 > best_f1:
                best_f1 = f1
                best_weights = model.state_dict().copy()
                torch.save(best_weights, os.path.join(output_dir, "best_model.pt"))

        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(csv_path, index=False)
        torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch+1}.pt"))

    print("\nLoading best model...")
    model.load_state_dict(best_weights)

    # Final Test
    model.eval()
    y_true, y_pred = [], []
    for fused, radar, labels in tqdm(dataloaders['test'], desc="TEST"):
        fused, radar, labels = fused.to(device), radar.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(fused, radar)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\nTEST — Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(test_log_path)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

# === Run ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEfficientNetFusionModel()
    dataloaders = create_dataloaders("/nfs/home/noualb20/Chrono_Matched_FusedRadar", batch_size=32)

    train_val_test(
        model=model,
        dataloaders=dataloaders,
        device=device,
        num_epochs=30,
        output_dir="2LiRAFixedFinalCheckpoints",
        csv_path="2LiRAFixedFinalTrainLog.csv",
        test_log_path="2LiRAFixedFinalTestReport.csv"
    )
