import os
import shutil
from pathlib import Path
from tqdm import tqdm

base_dir = Path("/nfs/home/noualb20/IMG_224")
splits = ['train', 'val', 'test']
split_ratios = [0.6, 0.2, 0.2]

# Get class folders, excluding existing split dirs
class_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name not in splits]

for class_folder in class_folders:
    print(f"\nProcessing: {class_folder.name}")
    files = sorted(class_folder.glob("*.jpg"))
    total = len(files)
    print(f"Total .jpg files: {total}")

    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])

    split_files = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, file_list in split_files.items():
        target_dir = base_dir / split / class_folder.name
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Moving {len(file_list)} files to '{split}/{class_folder.name}'")
        for file in tqdm(file_list, desc=f"    {split}", unit="file"):
            shutil.move(str(file), str(target_dir / file.name))
