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

from Utils.SegmentationModel import *

# Path to the chest-ct-segmentation dataset folder
data_dir = "dataset\Lucchi++" 
train_images_dir = os.path.join(data_dir, "Train_In")
train_masks_dir = os.path.join(data_dir, "Train_Out")
test_images_dir = os.path.join(data_dir, "Test_In")
test_masks_dir = os.path.join(data_dir, "Test_Out")
PATCH_SIZE = 128

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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