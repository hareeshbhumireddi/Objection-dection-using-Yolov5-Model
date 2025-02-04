# Test script to check installed package versions

import torch
print("Torch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import numpy as np
print("NumPy version:", np.__version__)

import cv2
print("OpenCV version:", cv2.__version__)

import ultralytics
print("Ultralytics version:", ultralytics.__version__)
