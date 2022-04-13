# kitti3d_displayer

A Python tool for displaying KITTI3D dataset

## Requirements
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/kitti3d_displayer.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n pcdet.v0.5.0 python=3.6
   conda activate pcdet.v0.5.0
   cd kitti3d_displayer
   pip install -r requirements.txt
   ```
 - Install visualization tools
   ```
   pip install mayavi
   pip install pyqt5
   pip install open3d-python
   ```

## Usages
 - Display data in KITTI3D dataset
   ```
   python kitti3d_displayer.py
   ```
