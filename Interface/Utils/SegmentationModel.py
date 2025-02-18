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

# Path to the chest-ct-segmentation dataset folder
data_dir = "dataset\Lucchi++" 
train_images_dir = os.path.join(data_dir, "Train_In")
train_masks_dir = os.path.join(data_dir, "Train_Out")
test_images_dir = os.path.join(data_dir, "Test_In")
test_masks_dir = os.path.join(data_dir, "Test_Out")
PATCH_SIZE = 128

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            self.conv_block(3, 32, stride=2),
            self.conv_block(32, 64, stride=2),
            self.conv_block(64, 128, stride=2),
            self.conv_block(128, 256, stride=2)
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
            self.upconv_block(64, 32),
            self.upconv_block(32, 32)
        )
        
        # Final classification layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        
    def conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
        
        # Decoder
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(self.decoder) - 1:
                x = x + features[-i-2]  # Skip connection
        
        # Final classification
        x = self.final(x)
        return x
# Custom Dataset class
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, patch_size=128, stride=64, transform=None):
        self.data_list = data_list
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # 预处理所有图像的patches
        self.all_patches = []
        self.all_masks = []
        self.all_positions = []
        self.all_sizes = []
        self.all_paths = []
        
        for item in data_list:
            image = cv2.imread(item["image"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(item["annotation"], cv2.IMREAD_GRAYSCALE)
            
            patches, mask_patches, positions, original_size = self.process_image_to_patches(image, mask)
            
            for patch, mask_patch in zip(patches, mask_patches):
                if self.transform:
                    patch = self.transform(patch)
                
                patch = torch.FloatTensor(patch.transpose(2, 0, 1)) / 255.0
                mask_patch = torch.FloatTensor(mask_patch).unsqueeze(0) / 255.0
                
                self.all_patches.append(patch)
                self.all_masks.append(mask_patch)
                self.all_positions.append(positions)
                self.all_sizes.append(original_size)
                self.all_paths.append(item["image"])

    def process_image_to_patches(self, image, mask):
        """处理图像和掩码为patches"""
        h, w = image.shape[:2]
        patches = []
        mask_patches = []
        positions = []
        
        for y in range(0, h-self.patch_size+1, self.stride):
            for x in range(0, w-self.patch_size+1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                patches.append(patch)
                mask_patches.append(mask_patch)
                positions.append((y, x))
        
        # 处理边缘情况
        if h % self.stride != 0:
            y = h - self.patch_size
            for x in range(0, w-self.patch_size+1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                mask_patches.append(mask_patch)
                positions.append((y, x))
        
        if w % self.stride != 0:
            x = w - self.patch_size
            for y in range(0, h-self.patch_size+1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                mask_patches.append(mask_patch)
                positions.append((y, x))
        
        if h % self.stride != 0 and w % self.stride != 0:
            y = h - self.patch_size
            x = w - self.patch_size
            patch = image[y:y+self.patch_size, x:x+self.patch_size]
            mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
            patches.append(patch)
            mask_patches.append(mask_patch)
            positions.append((y, x))
        
        return patches, mask_patches, positions, (h, w)

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        return {
            'patches': self.all_patches[idx],
            'mask_patches': self.all_masks[idx],
            'positions': self.all_positions[idx],
            'original_size': self.all_sizes[idx],
            'image_path': self.all_paths[idx]
        } 