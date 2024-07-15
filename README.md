# Local-aware Visual State Space for Small Defect Segmentation in Complex Component Images
This repository contains the PyTorch implementation and the proposed SD-Seg dataset.
## 1.Dataset Preparation
Download SD-Seg Dataset from 

## 2.Train
Train the model to get the checkpoints.
```
python train.py
```

## 3.Test
Calculate pixel-level and region-level metrics to evaluate the performance.
```
python cal_metrics.py
```

Output the segmentation masks for visualization.
```
python Test_defect.py
```
