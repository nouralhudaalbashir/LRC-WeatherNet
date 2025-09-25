# LRC-WeatherNet

LRC-WeatherNet is a deep learning pipeline for **multi-sensor weather classification** using LiDAR, Radar, and Camera data.  
It is part of a master's thesis project on **multi-modal fusion for autonomous vehicles**.

## Structure
- `models/` → Training scripts for different input modalities (LiDAR-only, Radar-only, RGB-only, Fusion, Gated Fusion).
- `preprocessing/` → Scripts to convert raw sensor data into BEV or `.pt` files.
- `notebooks/` → Jupyter notebooks for experiments and visualization.

## Requirements
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt

