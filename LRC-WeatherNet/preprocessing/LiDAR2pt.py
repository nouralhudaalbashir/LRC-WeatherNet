import os
import numpy as np
import torch
from tqdm import tqdm

# ==== Configuration ====
input_root = "/nfs/home/noualb20/LiDAR_Radar_Fusion/BEV_LiDAR3"  # Path to class-named folders with .npy files
output_root = "/nfs/home/noualb20/1FinalLiDAR_PT"  # Path to save .pt files
os.makedirs(output_root, exist_ok=True)

skipped_log_path = os.path.join(output_root, "skipped_files.txt")
skipped_files = []

# ==== Process Each Class Folder ====
for class_name in sorted(os.listdir(input_root)):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue

    save_path = os.path.join(output_root, class_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing {class_name}:")
    for fname in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        if not fname.endswith(".npy"):
            continue

        fpath = os.path.join(class_path, fname)
        out_path = os.path.join(save_path, fname.replace(".npy", ".pt"))
        
        if os.path.exists(out_path):
            continue  # Already converted

        try:
            data = np.load(fpath)
            tensor = torch.from_numpy(data).float()
            torch.save(tensor, out_path)
        except Exception as e:
            skipped_files.append(f"{fpath} ({str(e)})")
            print(f" Skipping corrupted file: {fpath} ({e})")
            continue

# ==== Save Skipped Files Log ====
if skipped_files:
    with open(skipped_log_path, 'w') as f:
        f.write("\n".join(skipped_files))
    print(f"\n Skipped {len(skipped_files)} corrupted files. Log saved to: {skipped_log_path}")
else:
    print("\n All files processed successfully. No corrupted files found.")
