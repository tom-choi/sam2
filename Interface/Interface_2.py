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

# Inference function
def predict(model, image_path, device="cuda"):
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 768))
    
    # Parameters for overlapping patches
    patch_size = PATCH_SIZE
    overlap = patch_size // 2  # 50% overlap
    
    # Calculate steps with overlap
    h_steps = int(np.ceil((768 - patch_size) / (patch_size - overlap))) + 1
    w_steps = int(np.ceil((1024 - patch_size) / (patch_size - overlap))) + 1
    
    patches_list = []
    patch_positions = []  # Store positions for reconstruction
    
    # Extract overlapping patches
    for i in range(h_steps):
        for j in range(w_steps):
            # Calculate patch coordinates
            y_start = min(i * (patch_size - overlap), 768 - patch_size)
            x_start = min(j * (patch_size - overlap), 1024 - patch_size)
            
            # Extract patch
            patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # Normalize and convert channel order
            patch = patch / 255.0
            patch = patch.transpose(2, 0, 1)
            patches_list.append(patch)
            patch_positions.append((y_start, x_start))
    
    # Convert to tensor
    patches_array = np.stack(patches_list)
    patches_tensor = torch.from_numpy(patches_array).float().to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(patches_tensor)
        pred_masks = torch.sigmoid(outputs) > 0.5
    
    # Initialize weight and prediction maps
    weight_map = np.zeros((768, 1024))
    pred_map = np.zeros((768, 1024))
    
    # Create weight kernel for blending
    y, x = np.mgrid[0:patch_size, 0:patch_size]
    weight_kernel = np.exp(-((x - patch_size/2)**2 + (y - patch_size/2)**2) / (2*(patch_size/4)**2))
    
    # Reconstruct full mask with weighted averaging
    for idx, (y_start, x_start) in enumerate(patch_positions):
        mask_patch = pred_masks[idx, 0].cpu().numpy()
        
        # Apply weight kernel
        weighted_patch = mask_patch * weight_kernel
        
        # Add to prediction and weight maps
        y_end = min(y_start + patch_size, 768)
        x_end = min(x_start + patch_size, 1024)
        h, w = y_end - y_start, x_end - x_start
        
        pred_map[y_start:y_end, x_start:x_end] += weighted_patch[:h, :w]
        weight_map[y_start:y_end, x_start:x_end] += weight_kernel[:h, :w]
    
    # Normalize by weights
    full_mask = np.divide(pred_map, weight_map, where=weight_map > 0)
    full_mask = (full_mask > 0.5).astype(np.float32)
    
    return full_mask

# 加载预训练模型
def load_segmentation_model(model_type):
    model = SegmentationModel(num_classes=1)
    if model_type == "Best IoU":
        model.load_state_dict(torch.load("best_model_iou.pth", map_location=device))
    else:
        model.load_state_dict(torch.load("best_model_loss.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# 初始化模型
model_dict = {
    "Best IoU": load_segmentation_model("Best IoU"),
    "Best Loss": load_segmentation_model("Best Loss")
}

# 创建预测函数
def predict_interface(input_image, model_type):
    # 转换图像格式并保存临时文件
    with torch.no_grad():
        # 将Gradio的numpy数组转换为OpenCV格式
        if input_image.ndim == 3 and input_image.shape[2] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
            
            # 使用选定的模型进行预测
            prediction = predict(model_dict[model_type], f.name, device)
    
    # 将预测结果转换为可视化的掩模
    original_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    original_image = cv2.resize(original_image, (1024, 768))
    
    # 创建叠加可视化
    overlay = original_image.copy()
    overlay[prediction == 1] = [0, 0, 255]  # 用红色标记预测区域
    
    # 调整大小以便显示
    original_image = cv2.resize(original_image, (512, 384))
    mask = cv2.resize(prediction.astype(np.uint8)*255, (512, 384))
    overlay = cv2.resize(overlay, (512, 384))
    
    return original_image, mask, overlay

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chest CT Segmentation Demo 🏥")
    gr.Markdown("Upload a chest CT scan image to segment the anatomy.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input CT Scan", type="numpy")
            model_type = gr.Dropdown(
                choices=["Best IoU", "Best Loss"],
                value="Best IoU",
                label="Model Selection"
            )
            submit_btn = gr.Button("Run Segmentation 🚀")
        
        with gr.Column():
            output_original = gr.Image(label="Original Image", type="numpy")
            output_mask = gr.Image(label="Segmentation Mask", type="numpy")
            output_overlay = gr.Image(label="Overlay Result", type="numpy")
    
    submit_btn.click(
        fn=predict_interface,
        inputs=[input_image, model_type],
        outputs=[output_original, output_mask, output_overlay]
    )
    
    gr.Examples(
        examples=[os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)[:2]],
        inputs=input_image,
        label="Example CT Scans (Click to try)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)