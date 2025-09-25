import yaml
import pandas as pd
import numpy as np
import os
from pandas import json_normalize
# Configuration
H, W = 64, 1024  # Range-view image dimensions
crop_width = 256
fov_up, fov_down = 17.5, -16.5  # Field of view in degrees
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Function to split dataset into train, val, and test sets
def split_dataset(files, train_ratio, val_ratio, test_ratio):
    total_files = len(files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    return train_files, val_files, test_files

radar_transformer = {'radar_1': [3.2766, -0.9652, 0.0, 0.0, 0.0, -0.5, 0.8660254], 
'radar_2': [3.3274, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
'radar_3': [3.2766, 0.9652, 0.0, 0.0, 0.0, 0.5, 0.8660254], 
'radar_4': [0.0, 0.9652, 0.762, 0.0, 0.0, 0.6427876, 0.7660444], 
'radar_6': [0.0, -0.9652, 0.762, 0.0, 0.0, -0.6427876, 0.7660444], 
'radar_5': [-0.6604, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]}

from scipy.spatial.transform import Rotation as h


# Transform into a global coordinate system
def transform_coordinates(row, quaternion, translation):
    

    radar_coords = np.array([row['x_radar'], row['y_radar'], row['z_radar']])
    r = h.from_quat(quaternion)

    # Calculate the inverse rotation matrix
    inverse_rotation = r.inv()
   
    original_point = inverse_rotation.apply(radar_coords)
    #print(original_point)
    rotation_angle = r.magnitude() * (180 / np.pi)  # rotation angle 
    #print(rotation_angle, "***********", original_point)
    global_coords = original_point + translation
    
    
    return pd.Series(global_coords, index=['x', 'y', 'z'])



def update_yaml_df(path_dir):
    
    with open(path_dir, 'r') as file:
        yaml_data = yaml.safe_load(file)  
        flat_data = json_normalize(yaml_data, sep='.')
        
    radars = ["radar_1", "radar_2", "radar_3"] #, "radar_4", "radar_5", "radar_6"]
    all_radar = pd.DataFrame()
        

    for radar in radars:
        
        s = "/sensors/radar/{}/objects.detections".format(radar)
        if s not in flat_data.columns:
            #print(f"Warning: {s} not found in YAML data. Skipping...")
            continue
            
        radar_df = pd.DataFrame(flat_data[s][0])
    
    
        if not radar_df.empty:
            
            translation = radar_transformer[radar][:3]
            quaternion = radar_transformer[radar][3:]
    
    
            radar_df["x_radar"] = radar_df["range"] * np.cos(radar_df["elevation"]) * np.cos(radar_df["azimuth"])
            radar_df["y_radar"] = radar_df["range"] * np.cos(radar_df["elevation"]) * np.sin(radar_df["azimuth"])
            radar_df["z_radar"] = radar_df["range"] * np.sin(radar_df["elevation"])


            points = radar_df.apply(transform_coordinates, axis=1, quaternion=quaternion, translation=translation)
            radar_df[['x', 'y', 'z']] = points

            all_radar = pd.concat([all_radar, radar_df], ignore_index=True)
            all_radar = all_radar[['x', 'y', 'z', 'snr', 'rcs']]
            
    return all_radar



def convert_yaml_to_bin(input_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
   
    for root, _, files in os.walk(input_dir):
        
        if root.endswith("misc"):
            
            yaml_files = [f for f in files if f.endswith('.yaml')]
            train_files, val_files, test_files = split_dataset(yaml_files, train_ratio, val_ratio, test_ratio)

       
            split_map = {
            'train': train_files,
            'val': val_files,
            'test': test_files }
        

            parent_dir = os.path.dirname(root)  # Removes "misc"
            folder_name = os.path.basename(parent_dir)  # Gets "2022_rain"
            


            print(root)
           
            for split, split_files in split_map.items():
                
                folder_split = os.path.join(output_dir, split)
                os.makedirs(folder_split, exist_ok=True)
                
                for file_name in split_files:
                    
                    save_dir = os.path.join(folder_split, folder_name)
                    os.makedirs(save_dir, exist_ok=True)

                    output_file_path = os.path.join(save_dir, file_name.replace('.yaml', '.bin'))
                    if os.path.exists(output_file_path):
                        continue
                        
                       #print(output_file_path)
                       #print(f"Skipping {file_name} as {output_file_path} already exists.")
                      
                    
                #___________________________________________________________________________
                #Fusing radar sensors and transform the coordinates
                #___________________________________________________________________________
                
                    radar_data = update_yaml_df(os.path.join(root,file_name))
                    if radar_data.empty:
                        
                        #print("The conversion resulted in empty file or {} was empty".format(file_name))
                        continue 
                
                
                #___________________________________________________________________________


                    np_data = radar_data.to_numpy(dtype=np.float32)  
                    np_data.tofile(os.path.join(save_dir, file_name.replace('.yaml', '.bin')))  # Saves as .bin
                
                            
                '''
 
                scan = LaserScan(project=True, H=H, W=W, fov_up=fov_up, fov_down=fov_down)
                scan.open_scan(file_path)
                scan.do_range_projection()
                '''

            

    




convert_yaml_to_bin(input_dir="/nfs/home/DATASETS/MSU-4S_unzip", output_dir="/nfs/home/noualb20/LiDAR_Radar_Fusion/3Radar_180")