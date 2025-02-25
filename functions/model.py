import torch
import torch.nn as nn

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







