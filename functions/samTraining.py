import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from skimage.morphology import remove_small_objects, disk
from scipy.ndimage import binary_dilation, binary_erosion
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage import measure

# 這裡是SAM2的訓練函數，用於把整個sam扔到數據集上進行訓練