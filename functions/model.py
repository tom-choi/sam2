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

def load_model(model_path, num_classes, device, model_type = "UNET"):
    
    # Create an instance of your model
    if model_type == "UNET":
        model = UNet(num_classes)
    elif model_type == "YNET":
        model = Att_YNet(num_classes)
    else:
        raise ValueError("Invalid model type. Please choose 'UNET' or 'YNET'.")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict into your model
    model.load_state_dict(state_dict)
    
    # Move the model to the specified device
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
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
    """从patches重建完整图像"""
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

def predict(model, image_path, device="cuda", PATCH_SIZE = 64):
    model.eval()
    
    # 加载图像，保持原始尺寸
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取原始图像尺寸
    h, w = image.shape[:2]
    
    # Patch参数
    patch_size = PATCH_SIZE
    overlap = patch_size // 2  # 50% 重叠
    
    # 计算步数
    h_steps = int(np.ceil((h - patch_size) / (patch_size - overlap))) + 1
    w_steps = int(np.ceil((w - patch_size) / (patch_size - overlap))) + 1
    
    patches_list = []
    patch_positions = []
    
    # 提取重叠的patches
    for i in range(h_steps):
        for j in range(w_steps):
            # 计算patch坐标
            y_start = min(i * (patch_size - overlap), h - patch_size)
            x_start = min(j * (patch_size - overlap), w - patch_size)
            
            # 提取patch
            patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # 标准化和通道顺序转换
            patch = patch / 255.0
            patch = patch.transpose(2, 0, 1)
            patches_list.append(patch)
            patch_positions.append((y_start, x_start))
    
    # 转换为tensor
    patches_array = np.stack(patches_list)
    patches_tensor = torch.from_numpy(patches_array).float().to(device)
    
    # 运行推理
    with torch.no_grad():
        outputs = model(patches_tensor)
        pred_masks = torch.sigmoid(outputs) > 0.5
    
    # 初始化权重和预测图
    weight_map = np.zeros((h, w))
    pred_map = np.zeros((h, w))
    
    # 创建权重核用于混合
    y, x = np.mgrid[0:patch_size, 0:patch_size]
    weight_kernel = np.exp(-((x - patch_size/2)**2 + (y - patch_size/2)**2) / (2*(patch_size/4)**2))
    
    # 使用加权平均重建完整掩码
    for idx, (y_start, x_start) in enumerate(patch_positions):
        mask_patch = pred_masks[idx, 0].cpu().numpy()
        
        # 应用权重核
        weighted_patch = mask_patch * weight_kernel
        
        # 添加到预测和权重图
        y_end = min(y_start + patch_size, h)
        x_end = min(x_start + patch_size, w)
        patch_h = y_end - y_start
        patch_w = x_end - x_start
        
        pred_map[y_start:y_end, x_start:x_end] += weighted_patch[:patch_h, :patch_w]
        weight_map[y_start:y_end, x_start:x_end] += weight_kernel[:patch_h, :patch_w]
    
    # 权重归一化
    full_mask = np.divide(pred_map, weight_map, where=weight_map > 0)
    full_mask = (full_mask > 0.5).astype(np.float32)
    
    return full_mask