import gc
import os
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from functions.imagePreprocessing import ImagePreprocessor

# from functions.SegmentationSet import UNet
'''
这是一个用于图像分割的深度学习模型，采用了经典的U-Net架构设计。主要逻辑框架：
整体架构
    编码器（下采样部分）
    解码器（上采样部分）
    最终分类层
编码器部分（Encoder）
    输入图像经过4次连续的下采样处理
    每次下采样都会：
        将图像尺寸缩小一半
        增加特征通道数（3→32→64→128→256）
        提取更深层的特征信息
        
解码器部分（Decoder）
    将编码器压缩的信息逐步还原
    通过4次上采样操作：
        逐步增大图像尺寸
        减少特征通道数（256→128→64→32→32）
    使用跳跃连接（Skip Connection）
        将编码器对应层的特征与解码器特征相加
        帮助保留细节信息，改善分割效果
        
特殊设计
    每个卷积块都包含：
        卷积层
        批归一化
        ReLU激活函数
    使用跳跃连接来防止信息丢失
    最后通过一个卷积层得到每个像素的分类结果
'''
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
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

# A YNet with Attention Block and Spatial Dropout
class Att_YNet(nn.Module):
    class AttentionBlock(nn.Module):
        def __init__(self, F_g, F_l, F_int, batch_norm=False):
            super(Att_YNet.AttentionBlock, self).__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int) if batch_norm else nn.Identity()
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int) if batch_norm else nn.Identity()
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1) if batch_norm else nn.Identity(),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi
    
    def get_activation(activation_type):
        """Helper function to get activation layer"""
        if activation_type.lower() == 'elu':
            return nn.ELU()
        elif activation_type.lower() == 'relu':
            return nn.ReLU()
        elif activation_type.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_type.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation_type.lower() == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def __init__(self, image_shape, activation='elu', feature_maps=[16, 32, 64, 128, 256],
                 drop_values=[0.1, 0.1, 0.2, 0.2, 0.3], spatial_dropout=False, batch_norm=False,
                 n_classes=1):
        super(Att_YNet, self).__init__()

        self.depth = len(feature_maps) - 1
        self.activation = Att_YNet.get_activation(activation)  # 使用辅助函数获取激活层
        self.spatial_dropout = spatial_dropout
        self.batch_norm = batch_norm

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_channels = image_shape[0]  # 使用image_shape的第一个维度作为输入通道
        for i in range(self.depth):
            block = nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else feature_maps[i-1], feature_maps[i], 3, padding=1),
                nn.BatchNorm2d(feature_maps[i]) if batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout2d(drop_values[i]) if spatial_dropout else nn.Dropout(drop_values[i]),
                nn.Conv2d(feature_maps[i], feature_maps[i], 3, padding=1),
                nn.BatchNorm2d(feature_maps[i]) if batch_norm else nn.Identity(),
                self.activation
            )
            self.encoder_blocks.append(block)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_maps[-2], feature_maps[-1], 3, padding=1),
            nn.BatchNorm2d(feature_maps[-1]) if batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout2d(drop_values[-1]) if spatial_dropout else nn.Dropout(drop_values[-1]),
            nn.Conv2d(feature_maps[-1], feature_maps[-1], 3, padding=1),
            nn.BatchNorm2d(feature_maps[-1]) if batch_norm else nn.Identity(),
            self.activation
        )

        # Decoder (UNet)
        self.unet_decoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i in range(self.depth-1, -1, -1):
            self.attention_blocks.append(Att_YNet.AttentionBlock(feature_maps[i], feature_maps[i], feature_maps[i]//2, batch_norm))
            block = nn.Sequential(
                nn.Conv2d(feature_maps[i]*2, feature_maps[i], 3, padding=1),
                nn.BatchNorm2d(feature_maps[i]) if batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout2d(drop_values[i]) if spatial_dropout else nn.Dropout(drop_values[i]),
                nn.Conv2d(feature_maps[i], feature_maps[i], 3, padding=1),
                nn.BatchNorm2d(feature_maps[i]) if batch_norm else nn.Identity(),
                self.activation
            )
            self.unet_decoder_blocks.append(block)

        # Decoder (AutoEncoder)
        self.ae_decoder_blocks = nn.ModuleList()
        for i in range(self.depth-1, -1, -1):
            block = nn.Sequential(
                nn.Conv2d(feature_maps[i+1] if i != self.depth-1 else feature_maps[-1], feature_maps[i], 3, padding=1),
                nn.BatchNorm2d(feature_maps[i]) if batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout2d(drop_values[i]) if spatial_dropout else nn.Dropout(drop_values[i]),
                nn.Conv2d(feature_maps[i], feature_maps[i], 3, padding=1),
                nn.BatchNorm2d(feature_maps[i]) if batch_norm else nn.Identity(),
                self.activation
            )
            self.ae_decoder_blocks.append(block)

        self.final_conv_mask = nn.Conv2d(feature_maps[0], n_classes, 1)
        self.final_conv_img = nn.Conv2d(feature_maps[0], image_shape[0], 1)  # 输出通道数应与输入图像通道数相同

    def forward(self, x):
        # Encoder
        encoder_features = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_features.append(x)
            x = F.max_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # UNet Decoder
        unet_x = x
        for i, block in enumerate(self.unet_decoder_blocks):
            unet_x = F.interpolate(unet_x, scale_factor=2, mode='bilinear', align_corners=True)
            attn = self.attention_blocks[i](unet_x, encoder_features[-(i+1)])
            unet_x = torch.cat([unet_x, attn], dim=1)
            unet_x = block(unet_x)

        # AutoEncoder Decoder
        ae_x = x
        for block in self.ae_decoder_blocks:
            ae_x = F.interpolate(ae_x, scale_factor=2, mode='bilinear', align_corners=True)
            ae_x = block(ae_x)

        mask = torch.sigmoid(self.final_conv_mask(unet_x))
        img = self.final_conv_img(ae_x)

        return img, mask

# 加载保存的模型
def load_model(model_path, device="cuda", model_type = "UNET"):
    """
    加载保存的模型
    
    参数:
        model_path: 模型文件路径
        device: 使用的设备，默认为cuda
    
    返回:
        model: 加载的模型
    """
    num_classes = 1

    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        device = "cpu"
    
    # 初始化模型架构
    if model_type == "UNET":
        model = UNet(num_classes)
    elif model_type == "YNET":
        model = Att_YNet(num_classes)
    else:
        raise ValueError("Invalid model type. Please choose 'UNET' or 'YNET'.")
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 将模型转移到指定设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    print(f"模型已从 {model_path} 加载")
    
    return model

def load_model(model_path, device="cuda"):
    """
    加载保存的模型
    
    参数:
        model_path: 模型文件路径
        device: 使用的设备，默认为cuda
    
    返回:
        model: 加载的模型
    """
    # 检查设备
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        device = "cpu"
    
    # 初始化模型架构
    model = UNet(num_classes=1)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # 将模型转移到指定设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    print(f"模型已从 {model_path} 加载")
    
    return model


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    # Convert predictions to binary
    pred_mask = (pred_mask > threshold).float()

    # Calculate intersection and union
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection

    # Calculate IoU
    iou = (intersection + 1e-7) / (union + 1e-7)

    # Calculate Dice coefficient
    dice = (2. * intersection + 1e-7) / (pred_mask.sum() + true_mask.sum() + 1e-7)

    return iou.item(), dice.item()


def reconstruct_from_patches(patches, positions, original_size, patch_size, stride):
    """
    从patches重建完整图像
    
    参数:
        patches: 预测的patch列表
        positions: 每个patch的左上角坐标列表，格式为[(y1, x1), (y2, x2), ...]
        original_size: 原始图像尺寸，格式为(height, width)
        patch_size: patch的大小
        stride: patch滑动的步长

    返回:
        reconstructed: 重建后的完整图像
    """
    h, w = original_size
    reconstructed = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    for patch, (y, x) in zip(patches, positions):
        patch_h = min(patch_size, h - y)
        patch_w = min(patch_size, w - x)
        reconstructed[y:y+patch_h, x:x+patch_w] += patch[:patch_h, :patch_w]
        count[y:y+patch_h, x:x+patch_w] += 1
    
    # 处理重叠区域
    count[count == 0] = 1
    reconstructed /= count
    return reconstructed

def train_model(model, train_loader, val_loader, num_epochs=50, device="cuda"):
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_iou = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        
        for batch in tqdm(train_loader):
            patches = batch['patches'].to(device)
            mask_patches = batch['mask_patches'].to(device)
            
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, mask_patches)
            
            loss.backward()
            optimizer.step()
            
            pred_masks = torch.sigmoid(outputs) > 0.5
            iou, dice = calculate_metrics(pred_masks.float(), mask_patches)
            
            train_loss += loss.item()
            train_iou += iou
            train_dice += dice
        
        # 计算平均值
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch in val_loader:
                patches = batch['patches'].to(device)
                mask_patches = batch['mask_patches'].to(device)
                
                outputs = model(patches)
                loss = criterion(outputs, mask_patches)
                
                pred_masks = torch.sigmoid(outputs) > 0.5
                iou, dice = calculate_metrics(pred_masks.float(), mask_patches)
                
                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_model(model, 'best_model_iou.pth')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'best_model_loss.pth')

def optimized_train_model(model, train_loader, val_loader, num_epochs=50, device="cuda", 
                          batch_accumulation=1, pin_memory=True):
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_iou = 0
    best_val_loss = float('inf')
    
    # 设置异步数据加载
    train_loader.pin_memory = pin_memory
    val_loader.pin_memory = pin_memory
    
    # 初始清理内存
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        batch_count = 0
        
        optimizer.zero_grad()  # 外部清零梯度
        
        for batch_idx, batch in enumerate(train_loader):
            # 使用梯度累积减少GPU内存使用
            patches = batch['patches'].to(device, non_blocking=True)
            mask_patches = batch['mask_patches'].to(device, non_blocking=True)
            
            outputs = model(patches)
            loss = criterion(outputs, mask_patches) / batch_accumulation
            
            loss.backward()
            
            # 梯度累积：每累积batch_accumulation个批次更新一次权重
            if (batch_idx + 1) % batch_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            iou, dice = calculate_metrics(pred_masks, mask_patches)
            
            train_loss += loss.item() * batch_accumulation
            train_iou += iou
            train_dice += dice
            batch_count += 1
            
            # 在内部也进行内存清理
            del patches, mask_patches, outputs, pred_masks, loss
            
            # 定期清理GPU内存
            if batch_idx % 10 == 0 and device == "cuda":
                torch.cuda.empty_cache()
        
        # 处理剩余的梯度
        if batch_count % batch_accumulation != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 计算平均值
        train_loss /= batch_count
        train_iou /= batch_count
        train_dice /= batch_count
        
        # 训练完成后清理内存
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                patches = batch['patches'].to(device, non_blocking=True)
                mask_patches = batch['mask_patches'].to(device, non_blocking=True)
                
                outputs = model(patches)
                loss = criterion(outputs, mask_patches)
                
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                iou, dice = calculate_metrics(pred_masks, mask_patches)
                
                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
                val_count += 1
                
                del patches, mask_patches, outputs, pred_masks, loss
        
        val_loss /= val_count
        val_iou /= val_count
        val_dice /= val_count
        
        # 验证完成后清理内存
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 简洁输出格式
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}')
        print(f'Validation - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')
        print('------------------------------------------------------------')
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'models/UnetTrain/best_model_iou.pth')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/UnetTrain/best_model_loss.pth')   
   
def predict(model, image, mask=None, device="cuda", patch_size=256, stride=None):
    """
    核心预测函数：接收预处理后的图像，执行模型预测，返回预测结果和评估指标
    
    参数:
        model: 训练好的模型
        image: 预处理后的图像数组（RGB格式），已确保尺寸适合patch提取
        mask: 预处理后的真实掩码数组（可选，用于计算评估指标）
        device: 使用的设备，默认为cuda
        patch_size: 处理的patch大小
        stride: patch滑动的步长，None则默认为patch_size//2
    
    返回:
        pred_mask: 预测的分割掩码
        metrics: 评估指标（若提供真实掩码）
    """
    # 设置默认stride
    if stride is None:
        stride = patch_size // 2
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 执行预测
    model.eval()
    
    # 计算步数
    h_steps = max(1, (h - patch_size + stride) // stride)
    w_steps = max(1, (w - patch_size + stride) // stride)
    
    patches_list = []
    patch_positions = []
    
    # 提取重叠的patches
    for i in range(h_steps):
        for j in range(w_steps):
            # 计算patch坐标
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)
            
            # 提取patch
            patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # 处理可能的边界情况，确保patch尺寸正确
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                temp_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = temp_patch
            
            # 标准化和通道顺序转换
            normalized_patch = patch.astype(np.float32) / 255.0
            normalized_patch = normalized_patch.transpose(2, 0, 1)
            patches_list.append(normalized_patch)
            patch_positions.append((y_start, x_start))
    
    # 转换为tensor
    patches_array = np.stack(patches_list)
    patches_tensor = torch.from_numpy(patches_array).float().to(device)
    
    # 分批处理以避免内存不足
    batch_size = 16  # 可以根据可用GPU内存调整
    all_pred_patches = []
    
    for i in range(0, len(patches_tensor), batch_size):
        batch = patches_tensor[i:i+batch_size]
        with torch.no_grad():
            outputs = model(batch)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            all_pred_patches.extend([p[0].cpu().numpy() for p in pred_masks])
    
    # 重建完整的预测掩码
    prediction = reconstruct_from_patches(
        all_pred_patches, 
        patch_positions, 
        (h, w), 
        patch_size, 
        stride
    )
    
    # 二值化
    prediction = (prediction > 0.5).astype(np.float32)
    
    # 计算评估指标（如果提供了真实掩码）
    metrics = None
    if mask is not None:
        # 确保掩码格式正确
        # 假设255是前景，0和2是背景
        mask_float = (mask == 255).astype(np.float32)
            
        # 转换为tensor计算IoU和Dice
        pred_tensor = torch.from_numpy(prediction)
        true_tensor = torch.from_numpy((mask_float > 0.5).astype(np.float32))
        
        # 确保尺寸一致
        if pred_tensor.shape != true_tensor.shape:
            print(f"警告: 预测掩码 ({pred_tensor.shape}) 和真实掩码 ({true_tensor.shape}) 尺寸不一致")
            # 使用最近邻插值调整大小
            pred_tensor = torch.nn.functional.interpolate(
                pred_tensor.unsqueeze(0).unsqueeze(0), 
                size=true_tensor.shape, 
                mode='nearest'
            ).squeeze(0).squeeze(0)
        
        # 计算评估指标
        iou, dice = calculate_metrics(pred_tensor, true_tensor)
        metrics = {"IoU": iou, "Dice": dice}
        
        print(f"评估指标: IoU={iou:.4f}, Dice={dice:.4f}")
    
    return prediction, metrics

#### new testing part (2025.4.7)
def visualize_results(original_img, preprocessed_img, true_mask, pred_mask, metrics, file_name, save_dir, padding_info=None):
    """
    创建和保存可视化结果
    
    参数:
        original_img: 原始图像
        preprocessed_img: 预处理后的图像
        true_mask: 真实掩码
        pred_mask: 预测掩码
        metrics: 评估指标
        file_name: 文件名前缀
        save_dir: 保存目录
        padding_info: 填充信息字典
    """
    # 转换预处理图像为BGR（用于保存）
    if len(preprocessed_img.shape) == 3 and preprocessed_img.shape[2] == 3:
        preprocessed_img_bgr = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR)
    else:
        preprocessed_img_bgr = preprocessed_img
    
    # 创建轮廓图
    orig_with_contours = original_img.copy()
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(orig_with_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_with_contours.png"), orig_with_contours)
    
    # 创建叠加图
    pred_mask_color = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    pred_mask_color[pred_mask > 0] = [0, 255, 0]  # 绿色
    alpha_overlay = 0.3
    overlay = cv2.addWeighted(original_img, 1, pred_mask_color, alpha_overlay, 0)
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_overlay.png"), overlay)
    
    # 如果有真实掩码，创建对比图
    if true_mask is not None:
        # 创建彩色对比掩码
        color_mask = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
        
        # 调整大小以匹配
        if true_mask.shape != pred_mask.shape:
            resized_pred = cv2.resize(
                pred_mask, 
                (true_mask.shape[1], true_mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            resized_pred = pred_mask
        
        true_mask_bin = true_mask > 0
        pred_mask_bin = resized_pred > 0
        
        # 正确检测 (绿色)
        color_mask[np.logical_and(true_mask_bin, pred_mask_bin)] = [0, 255, 0]
        
        # 错误检测 (红色)
        color_mask[np.logical_and(np.logical_not(true_mask_bin), pred_mask_bin)] = [0, 0, 255]
        
        # 漏检 (蓝色)
        color_mask[np.logical_and(true_mask_bin, np.logical_not(pred_mask_bin))] = [255, 0, 0]
        
        cv2.imwrite(os.path.join(save_dir, f"{file_name}_color_mask.png"), color_mask)
        
        # 叠加到原始图像
        comparison_overlay = cv2.addWeighted(original_img, 1, color_mask, 0.5, 0)
        cv2.imwrite(os.path.join(save_dir, f"{file_name}_comparison.png"), comparison_overlay)
    
    # 创建matplotlib可视化
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 预处理图像（如果有填充信息，显示填充边界）
    plt.subplot(232)
    if len(preprocessed_img.shape) == 3:
        plt.imshow(preprocessed_img)
    else:
        plt.imshow(preprocessed_img, cmap='gray')
    
    # 如果有填充信息，在预处理图像上绘制填充边界
    if padding_info:
        h, w = padding_info["original_size"]
        top, left = padding_info["top"], padding_info["left"]
        
        # 绘制填充边界的矩形
        rect = plt.Rectangle(
            (left, top), 
            w, h, 
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.title(f'Preprocessed Image with Padding\n(red rectangle: original image boundaries)')
    else:
        plt.title('Preprocessed Image')
    plt.axis('off')
    
    # 预测掩码
    plt.subplot(233)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')
    
    # 如果有真实掩码
    if true_mask is not None:
        plt.subplot(234)
        plt.imshow(true_mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(comparison_overlay, cv2.COLOR_BGR2RGB))
        plt.title(f'Comparison\nIoU: {metrics["IoU"]:.4f}, Dice: {metrics["Dice"]:.4f}')
        plt.axis('off')
    
    plt.subplot(236)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Prediction Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建直方图比较
    plt.figure(figsize=(12, 6))
    
    # 原始图像直方图
    plt.subplot(121)
    if len(original_img.shape) == 3:
        orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original_img
    plt.hist(orig_gray.ravel(), 256, [0, 256])
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # 预处理后图像直方图
    plt.subplot(122)
    if len(preprocessed_img.shape) == 3:
        preproc_gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
    else:
        preproc_gray = preprocessed_img
    plt.hist(preproc_gray.ravel(), 256, [0, 256])
    plt.title('Preprocessed Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_name}_histograms.png"), dpi=300, bbox_inches='tight')
    plt.close()

def segmentation_pipeline(model_path, image_path, mask_path=None, save_dir="test/predictData", 
                          patch_size=256, stride=None, value=0, alpha=1.0, device=None):
    """
    完整的分割流程：加载模型、预处理、预测、后处理和可视化
    
    参数:
        model_path: 模型文件路径
        image_path: 图像路径
        mask_path: 掩码路径，可选
        save_dir: 保存结果的目录
        patch_size: 处理的patch大小
        stride: patch滑动的步长
        value: 亮度调整值
        alpha: 对比度调整系数
        device: 使用的设备，None则自动选择
    
    返回:
        post_processed_mask: 后处理后的掩码
        metrics: 评估指标（若提供真实掩码）
    """
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置默认stride
    if stride is None:
        stride = patch_size // 2
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取文件名（不含扩展名）作为保存前缀
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"处理图像: {file_name}")
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    
    # 1. 加载模型
    model = load_model(model_path, device)
    
    # 2. 读取图像和掩码
    # 读取原始图像
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_h, original_w = original_img.shape[:2]
    
    # 读取真实掩码（如果有）
    true_mask = None
    if mask_path and os.path.exists(mask_path):
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 保存引用到原始真实掩码，用于最终评估
        original_true_mask = true_mask.copy()
    
    # 保存原始图像
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_original.png"), original_img)
    
    # 如果有真实掩码，保存它
    if true_mask is not None:
        cv2.imwrite(os.path.join(save_dir, f"{file_name}_true_mask.png"), true_mask)
    
    # 3. 预处理
    # 确保图像尺寸能被patch_size整除，必要时进行填充
    # 计算需要填充的尺寸
    # 先计算padding_info
    padding_info = ImagePreprocessor.calculate_padding_info(original_h, original_w, patch_size)

    # 从padding_info中提取填充值
    top_pad = padding_info["top"]
    bottom_pad = padding_info["bottom"]
    left_pad = padding_info["left"]
    right_pad = padding_info["right"]
    
    
    # 初始化预处理器
    preprocessor = ImagePreprocessor()
    
    # 应用预处理（包含亮度对比度调整和填充）
    preprocessed_img, preprocessed_mask = preprocessor.preprocess(
        original_img_rgb, 
        true_mask, 
        patch_size=patch_size,
        padding=(top_pad, bottom_pad, left_pad, right_pad),  # 传入填充信息
        value=value, 
        alpha=alpha
    )
    
    # 保存预处理后的图像
    preprocessed_img_bgr = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_preprocessed.png"), preprocessed_img_bgr)
    
    # 如果有预处理后的掩码，保存它
    if preprocessed_mask is not None:
        cv2.imwrite(os.path.join(save_dir, f"{file_name}_preprocessed_mask.png"), 
                   (preprocessed_mask ).astype(np.uint8))
    
    # 输出预处理前后的尺寸变化
    print(f"原始图像尺寸: {original_h}x{original_w}")
    print(f"预处理后图像尺寸: {preprocessed_img.shape[0]}x{preprocessed_img.shape[1]}")
    print(f"填充信息: 上={top_pad}, 下={bottom_pad}, 左={left_pad}, 右={right_pad}")
    
    # 4. 执行模型预测
    pred_mask, raw_metrics = predict(
        model=model,
        image=preprocessed_img,
        mask=preprocessed_mask,
        device=device,
        patch_size=patch_size,
        stride=stride
    )
    
    # 保存原始预测结果
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_prediction_raw.png"), 
               (pred_mask * 255).astype(np.uint8))
    
    # 5. 裁剪回原始尺寸
    # 计算裁剪区域
    h_start = top_pad
    w_start = left_pad
    h_end = preprocessed_img.shape[0] - bottom_pad
    w_end = preprocessed_img.shape[1] - right_pad
    
    # 确保不会越界
    h_start = max(0, h_start)
    w_start = max(0, w_start)
    h_end = min(preprocessed_img.shape[0], h_end)
    w_end = min(preprocessed_img.shape[1], w_end)
    
    # 裁剪预测掩码回原始尺寸
    pred_mask_original_size = pred_mask[h_start:h_end, w_start:w_end]
    
    # 确保预测掩码与原始图像尺寸完全匹配
    if pred_mask_original_size.shape[:2] != (original_h, original_w):
        print(f"警告: 裁剪后的预测掩码 ({pred_mask_original_size.shape[:2]}) 与原始图像 ({original_h}, {original_w}) 尺寸不完全匹配，正在调整...")
        pred_mask_original_size = cv2.resize(
            pred_mask_original_size, 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        )
    
    # 保存裁剪后的预测结果
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_prediction_original_size.png"), 
               (pred_mask_original_size * 255).astype(np.uint8))
    
    # 6. 后处理:降噪
    post_processed_mask = ImagePreprocessor.advanced_denoise(
        input_image=pred_mask_original_size,  # 已经是0-1范围的浮点数
        binary_threshold=0.5,
        min_noise_size=30,
        max_hole_size=200,
        opening_radius=5,
        closing_radius=3
    )

    # 保存后处理的掩码（转换为0-255范围保存图像）
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_prediction_final.png"), 
            (post_processed_mask * 255).astype(np.uint8))
    
    # 7. 计算后处理后的评估指标
    final_metrics = None
    if true_mask is not None:
        # 确保掩码尺寸完全匹配
        if post_processed_mask.shape != original_true_mask.shape:
            print(f"警告: 后处理掩码 ({post_processed_mask.shape}) 与原始真实掩码 ({original_true_mask.shape}) 尺寸不一致，正在调整...")
            # 调整大小以匹配原始真实掩码
            post_processed_mask_resized = cv2.resize(
                post_processed_mask, 
                (original_true_mask.shape[1], original_true_mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            post_processed_mask_resized = post_processed_mask
        
        # 转换掩码格式为浮点型 (同predict函数中的处理方式)
        if original_true_mask.dtype != np.float32:
            if original_true_mask.dtype == np.uint8:
                # 关键修改：使用"== 255"而不是"> 0"进行二值化
                true_mask_float = (original_true_mask == 255).astype(np.float32)
            else:
                true_mask_float = original_true_mask.astype(np.float32)
        else:
            true_mask_float = original_true_mask
        
        # 将后处理掩码转换为浮点型
        post_processed_float = (post_processed_mask_resized > 0).astype(np.float32)
        
        # 转换为tensor计算IoU和Dice (与predict函数相同)
        pred_tensor = torch.from_numpy(post_processed_float)
        true_tensor = torch.from_numpy((true_mask_float > 0.5).astype(np.float32))
        
        # 计算评估指标
        iou, dice = calculate_metrics(pred_tensor, true_tensor)
        final_metrics = {"IoU": iou, "Dice": dice}
        
        # 将指标写入文件
        with open(os.path.join(save_dir, f"{file_name}_metrics.txt"), 'w') as f:
            if raw_metrics:
                f.write(f"原始预测: IoU={raw_metrics['IoU']:.4f}, Dice={raw_metrics['Dice']:.4f}\n")
            f.write(f"后处理后: IoU={iou:.4f}, Dice={dice:.4f}\n")
        
        # 打印原始和后处理的评估指标比较
        if raw_metrics:
            print(f"原始预测指标: IoU={raw_metrics['IoU']:.4f}, Dice={raw_metrics['Dice']:.4f}")
        print(f"后处理后指标: IoU={iou:.4f}, Dice={dice:.4f}")
        
        # 计算改善百分比
        if raw_metrics:
            iou_improvement = (iou - raw_metrics['IoU']) / raw_metrics['IoU'] * 100 if raw_metrics['IoU'] > 0 else float('inf')
            dice_improvement = (dice - raw_metrics['Dice']) / raw_metrics['Dice'] * 100 if raw_metrics['Dice'] > 0 else float('inf')
            print(f"指标改善: IoU: {iou_improvement:.2f}%, Dice: {dice_improvement:.2f}%")
            
    # 8. 创建可视化结果
    visualize_results(
        original_img=original_img,
        preprocessed_img=preprocessed_img,
        true_mask=original_true_mask if true_mask is not None else None,
        pred_mask=post_processed_mask,
        metrics=final_metrics,
        file_name=file_name,
        save_dir=save_dir,
        padding_info=padding_info
    )
    
    return post_processed_mask / 255.0, final_metrics

####


# 优化后预测函数 2025.3.5 --18:21(old version)
def optimized_predict(model, image_path, mask_path=None, device="cuda", save_dir="test/predictData", patch_size=256, stride=None, value=-30, alpha=1.3):
    """
    完整的预测流程：读取图像、预处理、预测、保存结果
    
    参数:
        model: 训练好的模型
        image_path: 图像路径
        mask_path: 掩码路径，可选
        device: 使用的设备，默认为cuda
        save_dir: 保存结果的目录
        patch_size: 处理的patch大小
        stride: patch滑动的步长，None则默认为patch_size//2
        value: 亮度调整值
        alpha: 对比度调整系数
    
    返回:
        pred_mask: 预测的分割掩码（已裁剪回原始尺寸）
    """
    # 设置默认stride
    if stride is None:
        stride = patch_size // 2  # 默认50%重叠
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取文件名（不含扩展名）作为保存前缀
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. 读取原始图像
    orig_image = cv2.imread(image_path)
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    # 保存原始图像尺寸
    orig_h, orig_w = orig_image.shape[:2]
    
    # 2. 读取原始mask（如果提供）
    orig_mask = None
    if mask_path and os.path.exists(mask_path):
        orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 3. 执行预处理，使用与训练时完全相同的预处理流程
    preprocessor = ImagePreprocessor()
    
    # 调用与训练时完全相同的预处理方法
    processed_image, processed_mask = preprocessor.preprocess(
        orig_image_rgb, 
        orig_mask,
        patch_size=patch_size,
        value=value,
        alpha=alpha
    )
    
    # 计算填充信息（比较预处理前后的尺寸变化）
    processed_h, processed_w = processed_image.shape[:2]
    
    # 计算每边的填充像素数
    h_diff = processed_h - orig_h
    w_diff = processed_w - orig_w
    
    # 对称填充情况下：
    top_pad = h_diff // 2
    bottom_pad = h_diff - top_pad
    left_pad = w_diff // 2
    right_pad = w_diff - left_pad
    
    padding_info = (top_pad, bottom_pad, left_pad, right_pad)
    
    # 4. 保存原始和预处理数据
    # 原始图像
    cv2.imwrite(os.path.join(save_dir, f"original.png"), orig_image)
    
    # 原始mask（如果存在）
    if orig_mask is not None:
        cv2.imwrite(os.path.join(save_dir, f"original_mask.png"), orig_mask)
    
    # 预处理后的图像
    processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f"processed.png"), processed_image_bgr)
    
    # 预处理后的mask（如果存在）
    if processed_mask is not None:
        # 将浮点mask转换为8位灰度图像
        if processed_mask.dtype != np.uint8:
            processed_mask_save = (processed_mask * 255).astype(np.uint8)
        else:
            processed_mask_save = processed_mask
        cv2.imwrite(os.path.join(save_dir, f"processed_mask.png"), processed_mask_save)
    
    # 5. 执行预测
    model.eval()
    
    # 获取当前处理图像的尺寸
    current_h, current_w = processed_image.shape[:2]
    
    # 计算步数
    h_steps = max(1, (current_h - patch_size + stride) // stride)
    w_steps = max(1, (current_w - patch_size + stride) // stride)
    
    patches_list = []
    patch_positions = []
    
    # 提取重叠的patches
    for i in range(h_steps):
        for j in range(w_steps):
            # 计算patch坐标
            y_start = min(i * stride, current_h - patch_size)
            x_start = min(j * stride, current_w - patch_size)
            
            # 确保不会超出图像边界
            y_end = min(y_start + patch_size, current_h)
            x_end = min(x_start + patch_size, current_w)
            
            # 如果patch尺寸不足，进行填充
            patch = processed_image[y_start:y_end, x_start:x_end]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                temp_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = temp_patch
            
            # 标准化和通道顺序转换
            normalized_patch = patch.astype(np.float32) / 255.0
            normalized_patch = normalized_patch.transpose(2, 0, 1)
            patches_list.append(normalized_patch)
            patch_positions.append((y_start, x_start))
    
    # 转换为tensor
    patches_array = np.stack(patches_list)
    patches_tensor = torch.from_numpy(patches_array).float().to(device)
    
    # 分批处理以避免内存不足
    batch_size = 16  # 可以根据可用GPU内存调整
    all_pred_patches = []
    
    for i in range(0, len(patches_tensor), batch_size):
        batch = patches_tensor[i:i+batch_size]
        with torch.no_grad():
            outputs = model(batch)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            all_pred_patches.extend([p[0].cpu().numpy() for p in pred_masks])
    
    # 重建完整的预测掩码
    processed_prediction = reconstruct_from_patches(
        all_pred_patches, 
        patch_positions, 
        (current_h, current_w), 
        patch_size, 
        stride
    )
    
    # 二值化
    processed_prediction = (processed_prediction > 0.5).astype(np.float32)
    
    # 6. 精确裁剪回原始尺寸
    # 使用padding_info来精确删除填充的像素
    if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
        # 计算裁剪区域
        h_start = top_pad
        w_start = left_pad
        h_end = current_h - bottom_pad
        w_end = current_w - right_pad
        
        # 确保不会发生越界
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        h_end = min(current_h, h_end)
        w_end = min(current_w, w_end)
        
        # 精确裁剪
        prediction = processed_prediction[h_start:h_end, w_start:w_end]
    else:
        # 如果没有填充，直接使用处理后的预测结果
        prediction = processed_prediction
    
    # 确保最终尺寸与原始图像一致
    if prediction.shape[0] != orig_h or prediction.shape[1] != orig_w:
        print(f"警告：裁剪后的掩码尺寸 ({prediction.shape[0]}x{prediction.shape[1]}) 与原始图像尺寸 ({orig_h}x{orig_w}) 不一致。")
        
        # 如果差异很小，可能是舍入误差，直接调整大小
        if abs(prediction.shape[0] - orig_h) <= 2 and abs(prediction.shape[1] - orig_w) <= 2:
            # 创建正确大小的空白掩码
            final_pred = np.zeros((orig_h, orig_w), dtype=np.float32)
            # 复制可用区域
            common_h = min(prediction.shape[0], orig_h)
            common_w = min(prediction.shape[1], orig_w)
            final_pred[:common_h, :common_w] = prediction[:common_h, :common_w]
            prediction = final_pred
            
            print(f"已调整掩码尺寸以匹配原始图像尺寸。")
        else:
            print(f"警告：裁剪结果与原始图像尺寸差异较大，可能存在预处理问题。")
    
    # 7. 保存预测结果
    # 保存预处理图像上的完整预测结果
    processed_pred_save = (processed_prediction * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"processed_prediction.png"), processed_pred_save)
    
    # 保存裁剪回原始尺寸的预测结果
    pred_save = (prediction * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"prediction.png"), pred_save)
    
    # 保存padding信息，便于验证
    padding_text = f"top_pad: {top_pad}, bottom_pad: {bottom_pad}, left_pad: {left_pad}, right_pad: {right_pad}"
    with open(os.path.join(save_dir, f"padding_info.txt"), 'w') as f:
        f.write(padding_text)
    
    print(f"所有结果已保存至 {save_dir} 目录")
    print(f"原始图像尺寸: {orig_h}x{orig_w}")
    print(f"预处理后图像尺寸: {current_h}x{current_w}")
    print(f"填充信息: {padding_text}")
    print(f"最终预测掩码尺寸: {prediction.shape[0]}x{prediction.shape[1]}")
    
    return prediction


def save_prediction_results(model, image_path, mask_path=None, save_dir="test/predictData", patch_size=256, 
                            stride=128, value=-30, alpha=1.3, device=None):
    """
    加载模型、执行预测并保存结果
    
    参数:
        model: 模型对象
        image_path: 图像路径
        mask_path: 掩码路径，可选
        save_dir: 保存结果的目录
        patch_size: 处理的patch大小
        stride: patch滑动的步长
        value: 亮度调整值
        alpha: 对比度调整系数
        device: 使用的设备，None则自动选择
    """
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"使用设备: {device}")
    # print(f"模型路径: {model_path}")
    print(f"图像路径: {image_path}")
    print(f"掩码路径: {mask_path}")
    print(f"保存目录: {save_dir}")
    print(f"Patch大小: {patch_size}")
    print(f"Stride: {stride}")
    print(f"亮度调整值: {value}")
    print(f"对比度系数: {alpha}")
    
    # 读取原始图像
    original_img = cv2.imread(image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 获取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 使用您的preprocess方法预处理图像
    # 创建一个空白mask作为placeholder，如果没有提供真实mask
    dummy_mask = np.zeros_like(original_img[:,:,0]) if mask_path is None else cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 初始化预处理器
    preprocessor = ImagePreprocessor()  # 假设您的类名是ImagePreprocessor
    
    # 应用预处理
    preprocessed_img, preprocessed_mask = preprocessor.preprocess(
        original_img, 
        dummy_mask, 
        patch_size=patch_size,
        value=value, 
        alpha=alpha
    )
    
    # 保存预处理后的图像
    if len(preprocessed_img.shape) == 3:
        preprocessed_img_display = preprocessed_img.copy()
    else:
        preprocessed_img_display = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(os.path.join(save_dir, f"preprocessed.png"), preprocessed_img_display)
    
    # 加载模型
    # model = load_model(model_path, device=device)
    
    # 执行预测，显式指定stride参数
    pred_mask = optimized_predict(
        model=model,
        image_path=image_path,  # 原始图像路径，predict内部会调用相同的预处理
        mask_path=mask_path,
        device=device,
        save_dir=save_dir,
        patch_size=patch_size,
        stride=stride,  # 确保与训练时一致
        value=value,
        alpha=alpha
    )
    
    # 计算评估指标（如果提供了真实掩码）
    metrics_result = {}
    
    # 真实掩码（如果提供）
    if mask_path and os.path.exists(mask_path):
        # 读取真实掩码
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 保存二值化的真实掩码
        _, true_mask_bin = cv2.threshold(true_mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(save_dir, f"true_mask_binary.png"), true_mask_bin)
        
        # 计算IoU和Dice系数
        # 首先将掩码转换为torch tensor
        pred_tensor = torch.from_numpy(pred_mask.astype(np.float32))
        true_tensor = torch.from_numpy((true_mask_bin > 0).astype(np.float32))
        
        # 确保尺寸一致
        if pred_tensor.shape != true_tensor.shape:
            print(f"警告: 预测掩码 ({pred_tensor.shape}) 和真实掩码 ({true_tensor.shape}) 尺寸不一致")
            # 调整预测掩码大小以匹配真实掩码
            pred_tensor = torch.nn.functional.interpolate(
                pred_tensor.unsqueeze(0).unsqueeze(0), 
                size=true_tensor.shape, 
                mode='nearest'
            ).squeeze(0).squeeze(0)
        
        # 计算评估指标
        iou, dice = calculate_metrics(pred_tensor, true_tensor)
        metrics_result = {"IoU": iou, "Dice": dice}
        
        print(f"IoU: {iou:.4f}")
        print(f"Dice: {dice:.4f}")
        
        # 将指标写入文件
        with open(os.path.join(save_dir, f"metrics.txt"), 'w') as f:
            f.write(f"IoU: {iou:.4f}\n")
            f.write(f"Dice: {dice:.4f}\n")
        
        # 生成对比图
        comparison = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
        
        # 预测掩码为白色
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
        comparison[pred_mask_bin > 0] = [255, 255, 255]  # 白色
        
        # 添加真实掩码的轮廓 (绿色)
        true_mask_edges = cv2.Canny(true_mask_bin, 100, 200)
        comparison[true_mask_edges > 0] = [0, 255, 0]  # 绿色
        
        # 保存对比图
        cv2.imwrite(os.path.join(save_dir, f"comparison.png"), comparison)
        
        # 创建彩色掩码图 (正确:绿色, 错误:红色, 漏检:蓝色)
        color_mask = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
        
        true_mask_bin = true_mask_bin > 0
        pred_mask_bin = pred_mask_bin > 0
        
        # 正确检测 (绿色) - 真实掩码和预测掩码都为1
        color_mask[np.logical_and(true_mask_bin, pred_mask_bin)] = [0, 255, 0]
        
        # 错误检测 (红色) - 真实掩码为0但预测掩码为1
        color_mask[np.logical_and(np.logical_not(true_mask_bin), pred_mask_bin)] = [0, 0, 255]
        
        # 漏检 (蓝色) - 真实掩码为1但预测掩码为0
        color_mask[np.logical_and(true_mask_bin, np.logical_not(pred_mask_bin))] = [255, 0, 0]
        
        cv2.imwrite(os.path.join(save_dir, f"color_mask.png"), color_mask)
        
        # 将彩色掩码叠加到原始图像上
        alpha_overlay = 0.5
        overlay = cv2.addWeighted(original_img, 1, color_mask, alpha_overlay, 0)
        cv2.imwrite(os.path.join(save_dir, f"overlay.png"), overlay)
    
    # 保存预测掩码（不同格式）
    # 二值化掩码
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(save_dir, f"prediction_binary.png"), pred_mask_bin)
    
    # 轮廓标记
    orig_with_contours = original_img.copy()
    contours, _ = cv2.findContours(pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(orig_with_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, f"with_contours.png"), orig_with_contours)
    
    # 半透明叠加
    pred_mask_color = cv2.cvtColor(pred_mask_bin, cv2.COLOR_GRAY2BGR)
    pred_mask_color[pred_mask_bin > 0] = [0, 255, 0]  # 将预测区域标记为绿色
    alpha_overlay = 0.3
    overlay = cv2.addWeighted(original_img, 1, pred_mask_color, alpha_overlay, 0)
    cv2.imwrite(os.path.join(save_dir, f"prediction_overlay.png"), overlay)
    
    # 可视化结果（使用matplotlib）- 添加预处理图像
    plt.figure(figsize=(15, 12))  # 增加画布大小以适应更多的子图
    
    # 原始图像
    plt.subplot(231)
    plt.imshow(original_img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # 预处理后的图像
    plt.subplot(232)
    if len(preprocessed_img.shape) == 3:
        plt.imshow(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(preprocessed_img, cmap='gray')
    plt.title(f'Preprocessed (value={value}, alpha={alpha})')
    plt.axis('off')
    
    # 预测掩码
    plt.subplot(233)
    plt.imshow(pred_mask, cmap='gray')
    plt.title(f'Prediction {pred_mask.shape}')
    plt.axis('off')
    
    # 真实掩码（如果提供）和指标信息
    if mask_path and os.path.exists(mask_path):
        plt.subplot(234)
        plt.imshow(true_mask, cmap='gray')
        plt.title(f'Ground Truth\nIoU: {metrics_result["IoU"]:.4f}, Dice: {metrics_result["Dice"]:.4f}')
        plt.axis('off')
        
        # 叠加显示（预测掩码轮廓叠加在真实掩码上）
        plt.subplot(235)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Overlay (Green: TP, Red: FP, Blue: FN)')
        plt.axis('off')
        
        # 对比图
        plt.subplot(236)
        plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
        plt.title('Comparison (White: Pred, Green: True edge)')
        plt.axis('off')
    else:
        # 如果没有真实掩码，则调整布局
        plt.subplot(234)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Prediction Overlay')
        plt.axis('off')
        
        # 预测掩码（二值化）
        plt.subplot(235)
        plt.imshow(pred_mask_bin, cmap='gray')
        plt.title('Binary Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存可视化结果
    plt.savefig(os.path.join(save_dir, f"visualization.png"), dpi=300, bbox_inches='tight')
    
    # 显示可视化结果
    plt.show()
    
    # 创建直方图比较
    plt.figure(figsize=(12, 6))
    
    # 原始图像直方图
    if len(original_img.shape) == 3:
        orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original_img
    
    plt.subplot(121)
    plt.hist(orig_gray.ravel(), 256, [0, 256])
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # 预处理后图像直方图
    preproc_gray = preprocessed_img
    if len(preprocessed_img.shape) == 3:
        preproc_gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
    
    plt.subplot(122)
    plt.hist(preproc_gray.ravel(), 256, [0, 256])
    plt.title('Preprocessed Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"histograms.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"所有结果已保存至 {save_dir} 目录")
    print("预测完成！")
    
    return pred_mask, metrics_result if mask_path and os.path.exists(mask_path) else None

