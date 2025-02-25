import torch
import torch.nn as nn
import torch.nn.functional as F

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






