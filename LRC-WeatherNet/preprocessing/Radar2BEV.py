import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==== CONFIG ====
radar_root = '/nfs/home/noualb20/LiDAR_Radar_Fusion/3Radar_180'
output_root = '/nfs/home/noualb20/3Radar_BEV_PT224'

x_range = (0, 50)
y_range = (-25, 25)
resolution = 0.1  # meters per pixel
target_size = (224, 224)  # H x W for EfficientNet

def project_to_bev(points, values, x_range, y_range, resolution):
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )
    points = points[mask]
    values = values[mask]

    points_adj = points.copy()
    points_adj[:, 0] -= x_range[0]
    points_adj[:, 1] -= y_range[0]

    x_size = int((x_range[1] - x_range[0]) / resolution)
    y_size = int((y_range[1] - y_range[0]) / resolution)

    x_bev = np.clip((points_adj[:, 0] / resolution).astype(np.int32), 0, x_size - 1)
    y_bev = np.clip((points_adj[:, 1] / resolution).astype(np.int32), 0, y_size - 1)
    y_bev = y_size - 1 - y_bev

    bev_map = np.zeros((x_size, y_size), dtype=np.float32)
    for x, y, v in zip(x_bev, y_bev, values):
        bev_map[x, y] = max(bev_map[x, y], v)

    return bev_map

def resize_to_224(bev_stack):
    tensor = torch.from_numpy(bev_stack).unsqueeze(0)  # [1, C, H, W]
    resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    return resized.squeeze(0)  # [C, 224, 224]

def convert_all_radar_bins(radar_root, output_root):
    total_processed = 0
    total_skipped = 0
    total_invalid = 0
    total_errors = 0

    all_bin_paths = []
    for dirpath, _, filenames in os.walk(radar_root):
        radar_files = [f for f in filenames if f.endswith('.bin')]
        for fname in radar_files:
            bin_path = os.path.join(dirpath, fname)
            relative_path = os.path.relpath(dirpath, radar_root)
            out_dir = os.path.join(output_root, relative_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname.replace('.bin', '.pt'))
            all_bin_paths.append((bin_path, out_path))

    print(f"📦 Found {len(all_bin_paths)} .bin radar files to process.")

    for bin_path, out_path in tqdm(all_bin_paths, desc="🚀 Converting Radar Files"):
        if os.path.exists(out_path):
            total_skipped += 1
            continue

        try:
            raw = np.fromfile(bin_path, dtype=np.float32)
            if raw.size % 5 != 0:
                total_invalid += 1
                continue

            points = raw.reshape(-1, 5)
            snr_bev = project_to_bev(points, points[:, 3], x_range, y_range, resolution)
            rcs_bev = project_to_bev(points, points[:, 4], x_range, y_range, resolution)
            stacked = np.stack([snr_bev, rcs_bev], axis=0)  # [2, H, W]
            tensor224 = resize_to_224(stacked)  # [2, 224, 224]
            torch.save(tensor224, out_path)
            total_processed += 1

        except Exception as e:
            total_errors += 1
            print(f" Error processing {bin_path}: {e}")

    print(f"\n Conversion Summary:")
    print(f"    Converted: {total_processed}")
    print(f"    Skipped: {total_skipped}")
    print(f"    Invalid shape: {total_invalid}")
    print(f"    Errors: {total_errors}")

# Run the script
if __name__ == '__main__':
    convert_all_radar_bins(radar_root, output_root)
