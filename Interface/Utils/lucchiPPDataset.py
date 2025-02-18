import tempfile
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import gradio as gr

from skimage import measure

class LucchiPPDataset(Dataset):
    """
    Lucchi++数据集加载器
    数据集结构：
    dataset/Lucchi++/
        ├── Train_In/    
        ├── Train_Out/   
        ├── Test_In/     
        └── Test_Out/    
    """
    def __init__(self, data_dir, split='test', transform=None):
        """
        参数:
            data_dir (str): Lucchi++数据集的根目录
            split (str): 'train' 或 'test'
            transform: 可选的图像变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 设置图像和掩码目录
        if split == 'train':
            self.image_dir = os.path.join(data_dir, "Train_In")
            self.mask_dir = os.path.join(data_dir, "Train_Out")
        else:
            self.image_dir = os.path.join(data_dir, "Test_In")
            self.mask_dir = os.path.join(data_dir, "Test_Out")
            
        # 获取所有图像文件
        self.image_files = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像路径
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, f"{idx}.png")
        
        # 读取图像和掩码
        image = cv2.imread(image_path)[..., ::-1]  # BGR转RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"无法读取图像或掩码: {image_path}, {mask_path}")
            
        # 调整大小
        r = min(1024 / image.shape[1], 1024 / image.shape[0])
        image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                         interpolation=cv2.INTER_NEAREST)
        
        # 二值化掩码
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 获取点标注
        eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
        labels = measure.label(eroded_mask)
        regions = measure.regionprops(labels)
        
        points = []
        for region in regions:
            y, x = region.coords[np.random.randint(len(region.coords))]
            points.append([x, y])
            
        points = np.array(points)
        
        # 调整维度
        binary_mask = np.expand_dims(binary_mask, axis=0)  # (1, H, W)
        if len(points) > 0:
            points = np.expand_dims(points, axis=1)  # (N, 1, 2)
            
        num_masks = len(regions)
        
        return image, binary_mask, points, num_masks

def load_lucchi_dataset(data_dir="dataset/Lucchi++", split='test'):
    """
    加载Lucchi++数据集
    
    参数:
        data_dir (str): 数据集根目录
        split (str): 'train' 或 'test'
    
    返回:
        LucchiPPDataset对象
    """
    return LucchiPPDataset(data_dir, split)