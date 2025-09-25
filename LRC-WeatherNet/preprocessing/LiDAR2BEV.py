import open3d as o3d
import numpy as np
import os

# ==== CONFIG ====
pcd_root = '/nfs/home/DATASETS/MSU-4S_unzip'
output_root = '/nfs/home/noualb20/LiDAR_Radar_Fusion/BEV_LiDAR3'

x_range = (-50, 50)
y_range = (-50, 50)
resolution = 0.1  # meters per pixel

def lidar_to_bev(points, x_range, y_range, resolution):
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )
    points = points[mask]

    points_adj = points.copy()
    points_adj[:, 0] -= x_range[0]
    points_adj[:, 1] -= y_range[0]

    x_size = int((x_range[1] - x_range[0]) / resolution)
    y_size = int((y_range[1] - y_range[0]) / resolution)

    x_bev = np.clip((points_adj[:, 0] / resolution).astype(np.int32), 0, x_size - 1)
    y_bev = np.clip((points_adj[:, 1] / resolution).astype(np.int32), 0, y_size - 1)

    y_bev = y_size - 1 - y_bev

    intensities = points[:, 3] if points.shape[1] >= 4 else np.ones(points.shape[0])

    bev_map = np.zeros((x_size, y_size), dtype=np.float32)
    np.maximum.at(bev_map, (x_bev, y_bev), intensities)

    return bev_map

def convert_oust_pcds(pcd_root, output_root):
    total_count = 0
    total_skipped = 0

    for dirpath, _, filenames in os.walk(pcd_root):
        if os.path.basename(dirpath) != 'oust':
            continue

        segment_name = os.path.basename(os.path.dirname(dirpath))
        out_dir = os.path.join(output_root, segment_name)
        os.makedirs(out_dir, exist_ok=True)

        segment_count = 0
        segment_skipped = 0

        for fname in filenames:
            if fname.endswith('.pcd'):
                pcd_path = os.path.join(dirpath, fname)
                save_path = os.path.join(out_dir, fname.replace('.pcd', '.npy'))

                if os.path.exists(save_path):
                    segment_skipped += 1
                    continue

                try:
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    points = np.asarray(pcd.points)

                    if points.shape[1] < 4:
                        points = np.hstack([points, np.ones((points.shape[0], 1))])

                    bev = lidar_to_bev(points, x_range, y_range, resolution)
                    np.save(save_path, bev)
                    segment_count += 1
                except Exception as e:
                    print(f" Error processing {pcd_path}: {e}")

        total_count += segment_count
        total_skipped += segment_skipped

        print(f" Finished folder: {segment_name} | New: {segment_count} | Skipped: {segment_skipped}")

    print(f"\n All Done! Total new: {total_count}, total skipped: {total_skipped}")

if __name__ == '__main__':
    convert_oust_pcds(pcd_root, output_root)
