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


This repository contains the code for the master’s thesis:

Multi-Sensor Fusion for Classifying Challenging Weather Conditions in Autonomous Driving
Albashir, Nour Alhuda; Hamoud, Danial
Halmstad University, School of Information Technology, 2025
Independent thesis, Advanced level (degree of Master – Two Years), 20 credits / 30 HE credits

 Abstract

This thesis explores multimodal deep learning approaches for weather-type classification in autonomous systems by fusing LiDAR, radar, and RGB sensor data. A range of fusion strategies were implemented and evaluated across two distinct frameworks: one based on the 3D point cloud encoder PointPillars, and the other on the lightweight image-based EfficientNet-B0 architecture. Both early and mid-level fusion combinations were tested to assess the complementary value of spatial and semantic features.

Experiments were conducted on the Michigan State University Four Seasons (MSU-4S) dataset, which provides synchronized data from all three modalities. Among several fusion configurations, the most effective results were achieved through early fusion of LiDAR and radar features, followed by gated mid-level fusion with RGB.

PointPillars-based model: 87.77% test accuracy, macro F1 = 0.870

EfficientNet-B0-based model: 86.77% test accuracy, macro F1 = 0.8452

Insight: Gated multimodal fusion balances spatial precision and computational efficiency, offering valuable potential for real-time weather classification.

Publication Details

Place, publisher, year, edition, pages: 2025, p. 110

National Category: Computer Vision and Learning Systems

Subject / Course: Computer Science and Engineering

Educational Program: Computer Science and Engineering, 300 credits

Supervisors: Eren Erdal Aksoy (Guest Professor)

Examiners: Carlos Silla (Professor)

Presentation: 2025-06-04, F506, Halmstad (English)

Identifiers

URN: urn:nbn:se:hh:diva-57333

OAI: oai:DiVA.org:hh-57333

DiVA, id: diva2:1997440

Keywords

Multi-sensor fusion · Early fusion · Gated fusion · EfficientNet · PointPillars · LiDAR · Radar · RGB · Bird’s Eye View · Autonomous vehicles · Weather classification
