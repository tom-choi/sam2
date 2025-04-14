# 这里是基于pytorch，添加的一些图像预处理的函数

'''
总结：
采用"先预处理后切patch"的策略
预处理步骤顺序参考random_preprocess
'''
import datetime
import json
import cv2
import numpy as np
import torch
import matplotlib   
import matplotlib.pyplot as plt
import os
from skimage import morphology # for denoise
import random

class ImagePreprocessor:

    # 图像size填充处理 以及 填充像素的颜色修改 
    # @staticmethod
    # def check_and_pad_image(image, mask, patch_size, padding=None):
    #     """
    #     检查图片尺寸是否能被patch_size整除，如果不能则根据指定的填充值进行填充
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         mask: 对应的mask (H, W)
    #         patch_size: patch大小
    #         padding: 可选的填充值元组 (top_pad, bottom_pad, left_pad, right_pad)
    #     Returns:
    #         处理后的image和mask，以及填充信息
    #     """
    #     original_h, original_w = image.shape[:2]
        
    #     # 计算需要填充的像素数
    #     if padding is not None:
    #         # 使用提供的填充值
    #         top_pad, bottom_pad, left_pad, right_pad = padding
    #     else:
    #         # 计算需要填充的像素数，使其能被patch_size整除
    #         pad_h = (patch_size - original_h % patch_size) % patch_size
    #         pad_w = (patch_size - original_w % patch_size) % patch_size
            
    #         # 将填充平均分布在两侧
    #         top_pad = pad_h // 2
    #         bottom_pad = pad_h - top_pad
    #         left_pad = pad_w // 2
    #         right_pad = pad_w - left_pad
            
    #         # 创建填充信息
    #         padding = (top_pad, bottom_pad, left_pad, right_pad)
        
    #     # 记录原始尺寸和填充信息（可以返回给调用者）
    #     padding_info = {
    #         "padding": padding,
    #         "original_size": (original_h, original_w)
    #     }
        
    #     # 检查是否需要填充
    #     if top_pad == 0 and bottom_pad == 0 and left_pad == 0 and right_pad == 0:
    #         return image, mask, padding_info
        
    #     # 对图像进行填充（黑色）
    #     image = cv2.copyMakeBorder(image, 
    #                             top_pad, bottom_pad, 
    #                             left_pad, right_pad, 
    #                             cv2.BORDER_CONSTANT, 
    #                             value=[0, 0, 0])
        
    #     # 对mask进行填充（使用背景颜色）
    #     background_value = 0
    #     mask = cv2.copyMakeBorder(mask, 
    #                             top_pad, bottom_pad, 
    #                             left_pad, right_pad, 
    #                             cv2.BORDER_CONSTANT, 
    #                             value=background_value)
        
    #     return image, mask
    
    ### 3.28新增完善填充处理的图片和背景pixel值设置
    @staticmethod
    def check_and_pad_image(image, mask, patch_size, padding=None):
        """
        检查图片尺寸是否能被patch_size整除，如果不能则根据指定的填充值进行填充
        Args:
            image: 原始图像 (H, W, C)
            mask: 对应的mask (H, W)，值为0、2、255，其中2是背景
            patch_size: patch大小
            padding: 可选的填充值元组 (top_pad, bottom_pad, left_pad, right_pad)
        Returns:
            处理后的image和mask，以及填充信息
        """
        original_h, original_w = image.shape[:2]
        
        # 计算需要填充的像素数
        if padding is not None:
            # 使用提供的填充值
            top_pad, bottom_pad, left_pad, right_pad = padding
        else:
            # 计算需要填充的像素数，使其能被patch_size整除
            pad_h = (patch_size - original_h % patch_size) % patch_size
            pad_w = (patch_size - original_w % patch_size) % patch_size
            
            # 将填充平均分布在两侧
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad
            
            # 创建填充信息
            padding = (top_pad, bottom_pad, left_pad, right_pad)
        
        # 记录原始尺寸和填充信息（可以返回给调用者）
        padding_info = {
            "padding": padding,
            "original_size": (original_h, original_w)
        }
        
        # 检查是否需要填充
        if top_pad == 0 and bottom_pad == 0 and left_pad == 0 and right_pad == 0:
            return image, mask, padding_info
        
        # 对图像进行填充（黑色）
        image = cv2.copyMakeBorder(image, 
                                top_pad, bottom_pad, 
                                left_pad, right_pad, 
                                cv2.BORDER_CONSTANT, 
                                value=[0, 0, 0])
        
        # 对mask进行填充（使用背景颜色值2）
        background_value = 2  # 将背景值设置为2
        mask = cv2.copyMakeBorder(mask, 
                                top_pad, bottom_pad, 
                                left_pad, right_pad, 
                                cv2.BORDER_CONSTANT, 
                                value=background_value)
        
        return image, mask, padding_info
    
    ###3.28新增计算padding_info的函数
    def calculate_padding_info(image_height, image_width, patch_size):
        """
        计算图像需要的填充信息，确保图像尺寸能被patch_size整除
        
        参数:
            image_height (int): 原始图像高度
            image_width (int): 原始图像宽度
            patch_size (int): 分块大小
            
        返回:
            dict: 包含填充信息的字典
        """
        # 计算需要填充的尺寸
        h_padding = (patch_size - image_height % patch_size) % patch_size
        w_padding = (patch_size - image_width % patch_size) % patch_size
        
        # 记录填充信息，便于后续裁剪回原始尺寸
        top_pad = h_padding // 2
        bottom_pad = h_padding - top_pad
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        
        padding_info = {
            "top": top_pad,
            "bottom": bottom_pad,
            "left": left_pad,
            "right": right_pad,
            "original_size": (image_height, image_width)
        }
        
        return padding_info
    
    
    # 3.26新增，通过迁移原图像到新的画布，或许后续可以快速删除添加的黑色像素区域
    @staticmethod
    def _handle_edges(image, mask, patch_size):
        """处理图像边缘，确保能被完整地分割成patch"""
        h, w = image.shape[:2]
        new_h = ((h - 1) // patch_size + 1) * patch_size
        new_w = ((w - 1) // patch_size + 1) * patch_size
        
        # 如果尺寸已经符合要求，直接返回
        if h == new_h and w == new_w:
            return image, mask
            
        # 创建新的画布
        new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # 将原始图像和掩码复制到新画布
        new_image[:h, :w] = image
        new_mask[:h, :w] = mask
        
        return new_image, new_mask
    
    #### 检测透明边缘的输出图片格式
    # @staticmethod
    # def handle_transparent_edges(image, mask):
    #     """
    #     处理PNG图像的透明边缘，将其转换为黑色背景
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         mask: 对应的mask (H, W)
    #     Returns:
    #         处理后的image和mask
    #     """
    #     # 处理图像的透明区域
    #     if image.shape[2] == 4:  # 如果有alpha通道
    #         # 分离alpha通道
    #         b, g, r, alpha = cv2.split(image)
            
    #         # 创建黑色背景
    #         black_background = np.zeros_like(image[..., :3])
            
    #         # 将图像合成到黑色背景上
    #         image = cv2.bitwise_and(image[..., :3], image[..., :3], mask=alpha)
            
    #         # 如果mask也是4通道（带alpha）
    #         if len(mask.shape) == 3 and mask.shape[2] == 4:
    #             _, _, _, mask_alpha = cv2.split(mask)
    #             # 先处理mask的透明区域
    #             mask = cv2.bitwise_and(mask[..., :3], mask[..., :3], mask=mask_alpha)
    #             # 转换为灰度图
    #             mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #             # 确保透明区域为0
    #             mask[mask_alpha == 0] = 0
    #         else:
    #             # 对单通道mask使用图像的alpha通道
    #             mask = cv2.bitwise_and(mask, mask, mask=alpha)
    #             # 确保透明区域为0
    #             mask[alpha == 0] = 0
        
    #     # 如果mask是4通道但图像不是
    #     elif len(mask.shape) == 3 and mask.shape[2] == 4:
    #         _, _, _, mask_alpha = cv2.split(mask)
    #         # 先处理mask的透明区域
    #         mask = cv2.bitwise_and(mask[..., :3], mask[..., :3], mask=mask_alpha)
    #         # 转换为灰度图
    #         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #         # 确保透明区域为0
    #         mask[mask_alpha == 0] = 0
        
    #     # 将mask严格二值化
    #     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
    #     return image, mask

    #  # 3.5新增内容，配合predict的部分使用，记录添加的像素位置
    
    ###3.28新增完善检测透明边缘的输出图片格式
    @staticmethod
    def handle_transparent_edges(image, mask):
        """
        处理PNG图像的透明边缘，将其转换为黑色背景
        Args:
            image: 原始图像 (H, W, C)
            mask: 对应的mask (H, W)，值为0、2、255，其中2是背景
        Returns:
            处理后的image和mask
        """
        # 处理图像的透明区域
        if image.shape[2] == 4:  # 如果有alpha通道
            # 分离alpha通道
            b, g, r, alpha = cv2.split(image)
            
            # 创建黑色背景
            black_background = np.zeros_like(image[..., :3])
            
            # 将图像合成到黑色背景上
            image = cv2.bitwise_and(image[..., :3], image[..., :3], mask=alpha)
            
            # 如果mask也是4通道（带alpha）
            if len(mask.shape) == 3 and mask.shape[2] == 4:
                _, _, _, mask_alpha = cv2.split(mask)
                # 先处理mask的透明区域
                mask = cv2.bitwise_and(mask[..., :3], mask[..., :3], mask=mask_alpha)
                # 转换为灰度图 - 使用BGR2GRAY而非RGB2GRAY
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # 确保透明区域为背景值2
                mask[mask_alpha == 0] = 2  # 使用背景值2而非0
            else:
                # 对单通道mask使用图像的alpha通道
                # 保存当前背景值
                background_areas = (mask == 2)
                mask = cv2.bitwise_and(mask, mask, mask=alpha)
                # 确保透明区域使用背景值2
                mask[alpha == 0] = 2  # 使用背景值2而非0
                # 确保原有背景区域保持为2
                mask[background_areas] = 2
        
        # 如果mask是4通道但图像不是
        elif len(mask.shape) == 3 and mask.shape[2] == 4:
            _, _, _, mask_alpha = cv2.split(mask)
            # 先处理mask的透明区域
            mask = cv2.bitwise_and(mask[..., :3], mask[..., :3], mask=mask_alpha)
            # 转换为灰度图 - 使用BGR2GRAY而非RGB2GRAY
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # 确保透明区域为背景值2
            mask[mask_alpha == 0] = 2  # 使用背景值2而非0
        
        # 不进行二值化处理，而是确保只有0、2、255三种值
        # 先保存当前的背景区域
        background_areas = (mask == 2)
        # 对非背景区域进行二值化处理
        foreground_areas = ~background_areas
        if np.any(foreground_areas):
            temp_mask = mask.copy()
            temp_mask[background_areas] = 0  # 临时将背景设为0以便二值化
            _, temp_mask = cv2.threshold(temp_mask, 127, 255, cv2.THRESH_BINARY)
            # 恢复处理后的前景区域
            mask[foreground_areas] = temp_mask[foreground_areas]
        # 确保背景值仍然是2
        mask[background_areas] = 2
        
        return image, mask
    
    @staticmethod
    def preprocess_with_padding_info(self, image_path, mask_path=None, patch_size=256, value=-30, alpha=1.3):
        """
        预处理图像并提供填充信息
        
        参数:
            image_path: 图像路径
            mask_path: 掩码路径，可选
            patch_size: patch大小
            value: 亮度调整值
            alpha: 对比度调整系数
        
        返回:
            processed_image: 预处理后的图像
            processed_mask: 预处理后的掩码（如果提供了mask_path）
            background_mask: 背景掩码
            padding_info: (top_pad, bottom_pad, left_pad, right_pad)
        """
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码（如果提供）
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 记录原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 创建背景掩码（初始全0）
        background_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # 调整亮度
        image = self.adjust_brightness(image, value)
        
        # 调整对比度
        image = self.adjust_contrast(image, alpha)
        
        # 处理透明边缘（如果需要）
        # 这部分取决于您的具体实现
        
        # 计算需要的填充量，使尺寸是patch_size的倍数
        h, w = image.shape[:2]
        new_h = ((h // patch_size) + (1 if h % patch_size != 0 else 0)) * patch_size
        new_w = ((w // patch_size) + (1 if w % patch_size != 0 else 0)) * patch_size
        
        # 计算每边需要的填充量
        top_pad = (new_h - h) // 2
        bottom_pad = new_h - h - top_pad
        left_pad = (new_w - w) // 2
        right_pad = new_w - w - left_pad
        
        # 填充图像
        padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # 如果有掩码，也进行相同的填充
        padded_mask = None
        if mask is not None:
            padded_mask = cv2.copyMakeBorder(mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        
        # 填充背景掩码
        padded_background = cv2.copyMakeBorder(background_mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        
        return padded_image, padded_mask, padded_background, (top_pad, bottom_pad, left_pad, right_pad)

    ################### 以下是详细的图像预处理的函数(没说明那就是对单张大图进行处理) ##########

# 图像归一化(两种方法)
    ''' 主要用途是把图片统一设置亮度、对比度、饱和度，
        使得图片的颜色分布更加均匀，从而提高模型的泛化能力。
        归一化的操作统一是对整张图片做的而不是对patche做的。
    
    '''
    ##########
    # 改进尝试，3.3版本，保留更多细节的归一化，建议使用
    @staticmethod  # 此方法需要结合前后景识别生成的mask去使用
    def normalize_image(image, mask=None, low_percentile=1, high_percentile=99):
        """
        改进的归一化处理，考虑有效区域 
        Args:
            image: 原始图像 (H, W, C)
            mask: 有效区域掩码 (H, W)，None表示全图有效
            low_percentile: 低百分位数剪裁点（默认1%）
            high_percentile: 高百分位数剪裁点（默认99%）
        Returns:
            归一化后的图像
        """
        # 1. 输入验证
        if not (0 <= low_percentile < high_percentile <= 100):
            raise ValueError("百分位数范围无效")
        
        # 2. 处理单通道图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # 3. 使用float64中间计算提高精度
        image = image.astype(np.float64)
        normalized = np.zeros_like(image)
        
        if mask is None:
            for c in range(image.shape[2]):
                channel = image[..., c]
                low_val, high_val = np.percentile(channel, [low_percentile, high_percentile])
                
                # 4. 添加微小值避免除以0
                scale = max(high_val - low_val, 1e-6)
                normalized[..., c] = np.clip(255 * (channel - low_val) / scale, 0, 255)
        else:
            # 5. 确保mask正确
            if mask.shape != image.shape[:2]:
                raise ValueError("mask尺寸与图像不匹配")
            mask = mask.astype(bool)
            
            for c in range(image.shape[2]):
                channel = image[..., c]
                valid_pixels = channel[mask]
                
                if len(valid_pixels) > 0:
                    low_val, high_val = np.percentile(valid_pixels, [low_percentile, high_percentile])
                    scale = max(high_val - low_val, 1e-6)
                    normalized[..., c] = np.clip(255 * (channel - low_val) / scale, 0, 255)
                else:
                    normalized[..., c] = channel
        
        return normalized.astype(np.uint8)
    
    ###########
    
    ###########
    # 旧版本3.3之前的归一化方法
    
    # @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    # def normalize_image(image, mask=None):
    #     """
    #     改进的归一化处理，考虑有效区域
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         mask: 有效区域掩码 (H, W)，None表示全图有效
    #     Returns:
    #         归一化后的图像
    #     """
    #     image = image.astype(np.float32)
        
    #     if mask is None:
    #         # 如果没有提供mask，使用全图
    #         mean = np.mean(image, axis=(0, 1))
    #         std = np.std(image, axis=(0, 1))
    #     else:
    #         # 只计算有效区域的统计量
    #         mask = mask.astype(bool)
    #         mean = []
    #         std = []
    #         for c in range(image.shape[2]):
    #             channel = image[..., c]
    #             mean.append(np.mean(channel[mask]))
    #             std.append(np.std(channel[mask]))
    #         mean = np.array(mean)
    #         std = np.array(std)
        
    #     # 归一化处理
    #     image = (image - mean) / (std + 1e-7)
    #     image = np.clip(image * 255, 0, 255)
        
    #     return image.astype(np.uint8)

    ###########


# 直方图均衡化(两种方法)
    
    ###########    
    # 改进尝试，3.3版本，用到CLAHE算法
    
    # @staticmethod  
    # def histogram_equalization(image, mask=None, clip_limit=2.0, tile_grid_size=(8, 8)):
    #     """
    #     改进的直方图均衡化，使用CLAHE算法并考虑有效区域
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         mask: 有效区域掩码 (H, W)，None表示全图有效
    #         clip_limit: CLAHE算法的对比度限制参数
    #         tile_grid_size: CLAHE算法的网格大小
    #     Returns:
    #         均衡化后的图像
    #     """
    #     # 转换到LAB色彩空间（比HSV更适合处理医学图像）
    #     lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #     l_channel = lab[:,:,0]
        
    #     # 创建CLAHE对象
    #     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    #     if mask is None:
    #         # 如果没有提供mask，对整个亮度通道应用CLAHE
    #         l_channel = clahe.apply(l_channel)
    #     else:
    #         # 只对有效区域进行CLAHE处理
    #         mask = mask.astype(bool)
            
    #         # 为了只处理掩码区域，我们需要创建副本
    #         l_output = l_channel.copy()
            
    #         # 在掩码区域应用CLAHE
    #         # 注意：这里我们仍然对整个通道应用CLAHE，但只复制掩码区域的结果
    #         l_equalized = clahe.apply(l_channel)
    #         l_output[mask] = l_equalized[mask]
            
    #         # 更新亮度通道
    #         l_channel = l_output
        
    #     # 更新LAB图像的亮度通道
    #     lab[:,:,0] = l_channel
        
    #     # 将LAB转回RGB
    #     result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
    #     # 轻微锐化以增强细胞边界（可选，如果效果不好可以注释掉）
    #     kernel = np.array([[-0.5,-0.5,-0.5], 
    #                     [-0.5, 5,-0.5],
    #                     [-0.5,-0.5,-0.5]])
    #     result = cv2.filter2D(result, -1, kernel)
        
    #     return result.astype(np.uint8)

    ###########   
    
    # 3.3前的旧版本
    # @staticmethod # 此方法需要结合前后景识别生成的mask去使用,旧版本
    # def histogram_equalization(image, mask=None):
    #     """
    #     改进的直方图均衡化，考虑有效区域
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         mask: 有效区域掩码 (H, W)，None表示全图有效
    #     Returns:
    #         均衡化后的图像
    #     """
    #     # 1. 添加输入验证
    #     if image.dtype != np.uint8:
    #         raise ValueError("输入图像应为uint8类型")
        
    #     # 2. 处理单通道图像的情况
    #     if len(image.shape) == 2:
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     v_channel = hsv[..., 2].copy()  # 3. 使用copy避免修改原数组
        
    #     if mask is None:
    #         v_channel = cv2.equalizeHist(v_channel)
    #     else:
    #         # 4. 确保mask是二值且与图像尺寸匹配
    #         if mask.shape != image.shape[:2]:
    #             raise ValueError("mask尺寸与图像不匹配")
    #         mask = mask.astype(bool)
            
    #         valid_v = v_channel[mask]
    #         if len(valid_v) == 0:
    #             return image  # 5. 无有效区域时直接返回
            
    #         # 6. 使用更精确的直方图计算方法
    #         hist = np.histogram(valid_v, bins=256, range=(0, 256))[0]
    #         cdf = hist.cumsum()
    #         cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
            
    #         # 7. 使用查找表提高效率
    #         lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
    #         v_channel[mask] = lut[valid_v]
        
    #     hsv[..., 2] = v_channel
    #     return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
 ##### 3.28矫正图片格式的histogram，对3.3的修订版(4.11使用，后续发现有问题就尝试再修订)
    @staticmethod
    def histogram_equalization(image, mask=None):
        """
        直方图均衡化，考虑有效区域，并保持输入输出格式一致
        Args:
            image: 原始图像 (H, W, C)，BGR或RGB格式，3通道
            mask: 有效区域掩码 (H, W)，灰度格式，值为0、2、255
        Returns:
            均衡化后的图像，与输入格式相同
        """
        # 检查输入类型
        if image.dtype != np.uint8:
            raise ValueError("输入图像应为uint8类型")
        
        # 保存原始图像形状信息
        original_shape = image.shape
        channels = 1 if len(original_shape) == 2 else original_shape[2]
        
        # 对RGB/BGR图像处理 - 转HSV处理亮度通道
        if channels == 3:
            # 转HSV以处理亮度通道
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 假设是BGR格式，如实际是RGB则需调整
            v_channel = hsv[..., 2].copy()
            
            if mask is None:
                # 整体均衡化
                v_channel = cv2.equalizeHist(v_channel)
            else:
                # 确保mask是二值且与图像尺寸匹配
                if mask.shape != image.shape[:2]:
                    raise ValueError("mask尺寸与图像不匹配")
                
                # 创建有效区域mask - 通常我们将非背景区域视为有效区域
                # 根据您的数据，值为2的是背景，所以mask != 2的是有效区域
                valid_mask = (mask != 2)
                
                valid_v = v_channel[valid_mask]
                if len(valid_v) == 0:
                    return image  # 无有效区域时直接返回
                
                # 直方图均衡化
                hist = np.histogram(valid_v, bins=256, range=(0, 256))[0]
                cdf = hist.cumsum()
                # 避免除零错误
                if cdf[-1] == 0:
                    return image
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
                
                # 使用查找表
                lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
                v_channel[valid_mask] = lut[valid_v]
            
            # 更新亮度通道
            hsv[..., 2] = v_channel
            
            # 转回原始颜色空间
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 与转换时使用相同的变换对
        
        # 对灰度图直接处理    
        else:
            if mask is None:
                result = cv2.equalizeHist(image)
            else:
                if mask.shape != image.shape:
                    raise ValueError("mask尺寸与图像不匹配")
                
                练的时候使用valid_mask = (mask != 2)
                v_channel = image.copy()
                valid_v = v_channel[valid_mask]
                
                if len(valid_v) == 0:
                    return image
                
                hist = np.histogram(valid_v, bins=256, range=(0, 256))[0]
                cdf = hist.cumsum()
                if cdf[-1] == 0:
                    return image
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
                
                lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
                v_channel[valid_mask] = lut[valid_v]
                
                result = v_channel
        
        return result.astype(np.uint8)
    
    # 4.13修订版
    # @staticmethod
    # def histogram_equalization(image, mask=None):
    #     """
    #     直方图均衡化，考虑有效区域，并保持输入输出格式一致
    #     Args:
    #         image: 原始图像 (H, W, C)，BGR或RGB格式，3通道
    #         mask: 有效区域掩码 (H, W)，灰度格式，值为0表示背景，非0表示前景
    #     Returns:
    #         均衡化后的图像，与输入格式相同
    #     """
    #     # 检查输入类型
    #     if image.dtype != np.uint8:
    #         raise ValueError("输入图像应为uint8类型")
        
    #     # 保存原始图像形状信息
    #     original_shape = image.shape
    #     channels = 1 if len(original_shape) == 2 else original_shape[2]
        
    #     # 确保掩码是单通道
    #     if mask is not None and len(mask.shape) == 3:
    #         if mask.shape[2] == 4:  # RGBA掩码
    #             # 使用Alpha通道作为掩码
    #             mask = mask[..., 3].copy()
    #             mask = (mask > 0).astype(np.uint8) * 255
    #         elif mask.shape[2] == 1:
    #             mask = mask[..., 0].copy()
        
    #     # 对RGB/BGR图像处理 - 转HSV处理亮度通道
    #     if channels == 3:
    #         # 转HSV以处理亮度通道
    #         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 假设是BGR格式，如实际是RGB则需调整
    #         v_channel = hsv[..., 2].copy()
            
    #         if mask is None:
    #             # 整体均衡化
    #             v_channel = cv2.equalizeHist(v_channel)
    #         else:
    #             # 确保mask是二值且与图像尺寸匹配
    #             if mask.shape != image.shape[:2]:
    #                 raise ValueError("mask尺寸与图像不匹配")
                
    #             # 创建有效区域mask - 前景区域(非零)视为有效区域
    #             valid_mask = (mask > 0)
                
    #             valid_v = v_channel[valid_mask]
    #             if len(valid_v) == 0:
    #                 return image  # 无有效区域时直接返回
                
    #             # 直方图均衡化
    #             hist = np.histogram(valid_v, bins=256, range=(0, 256))[0]
    #             cdf = hist.cumsum()
    #             # 避免除零错误
    #             if cdf[-1] == 0:
    #                 return image
    #             cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
                
    #             # 使用查找表
    #             lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
    #             v_channel[valid_mask] = lut[valid_v]
            
    #         # 更新亮度通道
    #         hsv[..., 2] = v_channel
            
    #         # 转回原始颜色空间
    #         result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 与转换时使用相同的变换对
        
    #     # 对灰度图直接处理    
    #     else:
    #         if mask is None:
    #             result = cv2.equalizeHist(image)
    #         else:
    #             if mask.shape != image.shape:
    #                 raise ValueError("mask尺寸与图像不匹配")
                
    #             # 使用非零区域作为有效区域
    #             valid_mask = (mask > 0)
    #             v_channel = image.copy()
    #             valid_v = v_channel[valid_mask]
                
    #             if len(valid_v) == 0:
    #                 return image
                
    #             hist = np.histogram(valid_v, bins=256, range=(0, 256))[0]
    #             cdf = hist.cumsum()
    #             if cdf[-1] == 0:
    #                 return image
    #             cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
                
    #             lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
    #             v_channel[valid_mask] = lut[valid_v]
                
    #             result = v_channel
        
    #     return result.astype(np.uint8)
    
    # 3.26新版直方图
    def enhance_image(self, image, mask=None):
        """图像增强处理，提高细节和纹理可见性"""
        # 将图像转换为LAB色彩空间处理
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        # 使用CLAHE (对比度受限自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        if mask is None:
            cl = clahe.apply(l_channel)
        else:
            # 只对前景应用CLAHE
            cl = l_channel.copy()
            foreground = mask > 0
            cl[foreground] = clahe.apply(l_channel[foreground].reshape(-1)).reshape(-1)
        
        # 更新L通道
        lab[:,:,0] = cl
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    ###########
   
    
    #3.26新增，此方法暂时不确定是否可行，需要验证
    @staticmethod
    def standardize_image(self, image):
        """标准化图像到均值0，方差1，提高模型训练稳定性"""
        image = image.astype(np.float32)
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1)) + 1e-6  # 避免除以0
        return (image - mean) / std
    
    # #3.26版,噪声移除
    # def denoise_image(self, image, mask=None):
    #     """非局部均值去噪，保留图像细节的同时去除噪声"""
    #     if mask is None:
    #         return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    #     else:
    #         result = image.copy()
    #         foreground = mask > 0
    #         # 只对前景区域去噪
    #         if np.any(foreground):
    #             # 创建ROI
    #             x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    #             roi = image[y:y+h, x:x+w]
    #             # 对ROI应用去噪
    #             denoised_roi = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
    #             # 使用掩码将去噪结果合并回原图
    #             mask_roi = foreground[y:y+h, x:x+w]
    #             result[y:y+h, x:x+w][mask_roi] = denoised_roi[mask_roi]
    #         return result



    # 图像细胞比例矫正
    ''' 可能不同的图片发来的细胞比例不同，需要进行矫正，
        比如同样一个细胞，近距离拍摄和远距离拍的大小就不同，
        需要将其要调整到一样大小,有一个想法就是取一个数据集某几个图片得出大致的细胞比例，
        然后对所有导入的数据图片进行矫正，这样就可以保证所有图片的细胞比例都一样了。
        主要算法就是根据最初计算的基本比例，计算出当前图片的比例，然后进行等比例的缩放，
        这样就可以保证细胞的大小基本一致。
    '''
    

    # 图像旋转
    '''
    这个函数专注于图像旋转，同时处理图像及其对应的掩码。
    旋转是以图像中心为旋转中心，旋转后维持图像的原始尺寸。
    旋转角度通过参数angle从外部获取。
    '''
    # @staticmethod
    # def rotate(img, mask, angle):
    #     # 获取图像尺寸
    #     h, w = img.shape[:2]
    #     # 设置旋转中心为图像中心
    #     center = (w/2, h/2)
    #     # 计算旋转矩阵
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     # 对图像进行旋转
    #     rotated_img = cv2.warpAffine(img, M, (w, h))
    #     # 对掩码进行相同的旋转
    #     rotated_mask = cv2.warpAffine(mask, M, (w, h))
        
    #     return rotated_img, rotated_mask

    @staticmethod
    def rotate(img, mask, angle):
        """
        旋转图像和mask，并自动填充成矩形保持原比例
        Args:
            img: 输入图像 (H,W,C)
            mask: 输入mask (H,W) 背景值为2
            angle: 旋转角度(度)
        Returns:
            旋转后的图像和mask
        """
        # 获取原始尺寸
        h, w = img.shape[:2]
        center = (w/2, h/2)
        
        # 执行旋转（不自动调整尺寸）
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=2)
        
        # 计算旋转后的有效区域（避免角上的黑色填充区）
        # 创建临时全白图像用于计算有效区域
        temp = np.ones((h, w), dtype=np.uint8) * 255
        rotated_temp = cv2.warpAffine(temp, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # 找到有效区域的边界
        coords = cv2.findNonZero(rotated_temp)
        if coords is not None:
            x, y, w_rect, h_rect = cv2.boundingRect(coords)
            
            # 计算需要的填充量（保持原始宽高比）
            pad_top = y
            pad_bottom = h - (y + h_rect)
            pad_left = x
            pad_right = w - (x + w_rect)
            
            # ===== 新增：填充量限制检查 =====
            max_pad_ratio = 0.7  # 最大允许填充原图70%的面积
            if (pad_top + pad_bottom) > h*max_pad_ratio or \
            (pad_left + pad_right) > w*max_pad_ratio:
                # 回退到90度倍数的标准旋转
                fallback_angle = random.choice([0, 90, 180, 270])
                if angle == fallback_angle:  # 避免无限递归
                    return rotated_img, rotated_mask
                return ImagePreprocessor.rotate(img, mask, fallback_angle)
            # ===== 结束新增 =====
            
            # 应用填充（使用check_and_pad_image逻辑）
            rotated_img, rotated_mask, _ = ImagePreprocessor.check_and_pad_image(
                rotated_img[y:y+h_rect, x:x+w_rect],
                rotated_mask[y:y+h_rect, x:x+w_rect],
                patch_size=1,  # 禁用patch_size整除检查
                padding=(pad_top, pad_bottom, pad_left, pad_right)
            )
        
        return rotated_img, rotated_mask

    # 图像缩放
    '''
    这个函数专注于图像缩放，同时处理图像及其对应的掩码。
    缩放是按照比例进行的，直接返回缩放后的图像和掩码。
    缩放比例通过参数scale从外部获取。
    '''
    @staticmethod
    def scale(img, mask, scale, patch_size=256):
        """
        对图像和掩码进行缩放，并确保缩放后的尺寸是patch_size的整数倍
        
        Args:
            img: 输入图像
            mask: 输入掩码
            scale: 缩放因子
            patch_size: 补丁大小，默认为256
            
        Returns:
            缩放后的图像和掩码
        """
        # 获取原始尺寸
        h, w = img.shape[:2]
        
        # 计算缩放后的尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 调整尺寸确保是patch_size的整数倍
        remainder_h = new_h % patch_size
        remainder_w = new_w % patch_size
        
        if remainder_h != 0:
            # 调整高度到最近的patch_size的整数倍
            new_h = new_h + (patch_size - remainder_h) if remainder_h > patch_size / 2 else new_h - remainder_h
        
        if remainder_w != 0:
            # 调整宽度到最近的patch_size的整数倍
            new_w = new_w + (patch_size - remainder_w) if remainder_w > patch_size / 2 else new_w - remainder_w
        
        # 确保尺寸至少为patch_size
        new_h = max(patch_size, new_h)
        new_w = max(patch_size, new_w)
        
        # 计算实际使用的缩放因子
        actual_scale_h = new_h / h
        actual_scale_w = new_w / w
        
        # 对图像进行缩放
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 对掩码进行相同的缩放
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 如果想记录实际使用的缩放因子，可以打印或返回
        # print(f"原始缩放因子: {scale}, 实际缩放因子: 高度={actual_scale_h}, 宽度={actual_scale_w}")
        
        return scaled_img, scaled_mask




###### 旧版3.28之前的亮度和对比度调整

    # # 亮度调整
    # @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    # def adjust_brightness(image, value=0, mask=None):
    #     """
    #     改进的亮度调整，只处理前景区域
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         value: 亮度调整值，正数增加亮度，负数减少亮度
    #         mask: 前景mask (H, W)，None表示全图处理
    #     Returns:
    #         调整亮度后的图像
    #     """
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     h, s, v = cv2.split(hsv)
        
    #     if mask is None:
    #         # 如果没有mask，处理全图
    #         lim = 255 - value
    #         v[v > lim] = 255
    #         v[v <= lim] = np.clip(v[v <= lim] + value, 0, 255)
    #     else:
    #         # 只处理前景区域
    #         foreground = mask > 0
    #         v_foreground = v[foreground]
            
    #         lim = 255 - value
    #         v_foreground[v_foreground > lim] = 255
    #         v_foreground[v_foreground <= lim] = np.clip(v_foreground[v_foreground <= lim] + value, 0, 255)
            
    #         v[foreground] = v_foreground
        
    #     final_hsv = cv2.merge((h, s, v))
    #     image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    #     return image.astype(np.uint8)

    # # 对比度调整
    # @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    # def adjust_contrast(image, alpha=1.0, mask=None):
    #     """
    #     改进的对比度调整，只处理前景区域
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         alpha: 对比度调整系数，大于1增加对比度，小于1减少对比度
    #         mask: 前景mask (H, W)，None表示全图处理
    #     Returns:
    #         调整对比度后的图像
    #     """
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
    #     if mask is None:
    #         # 如果没有mask，处理全图
    #         hsv[..., 2] = cv2.convertScaleAbs(hsv[..., 2], alpha=alpha, beta=0)
    #     else:
    #         # 只处理前景区域
    #         foreground = mask > 0
    #         v_channel = hsv[..., 2]
            
    #         # 获取前景区域的V通道值
    #         v_foreground = v_channel[foreground]
            
    #         # 调整对比度
    #         v_foreground = cv2.convertScaleAbs(v_foreground, alpha=alpha, beta=0)
            
    #         # 将调整后的值重新赋值回原数组
    #         v_channel[foreground] = v_foreground.reshape(-1)  # 确保是一维数组
    #         hsv[..., 2] = v_channel
        
    #     return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#####
    
    
########下面是3.28修订格式版的亮度和对比度调整

    # 亮度调整
    @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    def adjust_brightness(image, value=0, mask=None):
        """
        改进的亮度调整，只处理前景区域
        Args:
            image: 原始图像 (H, W, C)
            value: 亮度调整值，正数增加亮度，负数减少亮度
            mask: 前景mask (H, W)，None表示全图处理
        Returns:
            调整亮度后的图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 改为BGR2HSV
        h, s, v = cv2.split(hsv)
        
        if mask is None:
            # 如果没有mask，处理全图
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] = np.clip(v[v <= lim] + value, 0, 255)
        else:
            # 只处理前景区域
            foreground = mask > 0
            v_foreground = v[foreground]
            
            lim = 255 - value
            v_foreground[v_foreground > lim] = 255
            v_foreground[v_foreground <= lim] = np.clip(v_foreground[v_foreground <= lim] + value, 0, 255)
            
            v[foreground] = v_foreground
        
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)  # 改为HSV2BGR
        return image.astype(np.uint8)

    # 对比度调整
    @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    def adjust_contrast(image, alpha=1.0, mask=None):
        """
        改进的对比度调整，只处理前景区域
        Args:
            image: 原始图像 (H, W, C)
            alpha: 对比度调整系数，大于1增加对比度，小于1减少对比度
            mask: 前景mask (H, W)，None表示全图处理
        Returns:
            调整对比度后的图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 改为BGR2HSV
        
        if mask is None:
            # 如果没有mask，处理全图
            hsv[..., 2] = cv2.convertScaleAbs(hsv[..., 2], alpha=alpha, beta=0)
        else:
            # 只处理前景区域
            foreground = mask > 0
            v_channel = hsv[..., 2]
            
            # 获取前景区域的V通道值
            v_foreground = v_channel[foreground]
            
            # 调整对比度
            v_foreground = cv2.convertScaleAbs(v_foreground, alpha=alpha, beta=0)
            
            # 将调整后的值重新赋值回原数组
            v_channel[foreground] = v_foreground.reshape(-1)  # 确保是一维数组
            hsv[..., 2] = v_channel
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 改为HSV2BGR
    
    # 3.28新增mask处理，直接选mask区域值为2的作为背景(数据集mask标签做好就可以用这个)
    @staticmethod
    def generate_mask(mask, background_value=2):
        """
        处理细胞掩码图像，将指定值(例如2)的区域设为背景，生成前后景掩码
        
        Args:
            mask: 输入的掩码图像 (H, W)，可能包含多个值
            background_value: 要识别为背景的像素值，默认为2
            
        Returns:
            处理后的掩码 (H, W)，前景保持原值(如255)，背景为0
        """
        # 创建新的掩码
        # print(type(mask))
        processed_mask = mask.copy()
        
        # 将背景值设为0，其他值保持不变
        processed_mask[processed_mask == background_value] = 0
        
        return processed_mask
    
    
    ## 4.13尝试修改mask，RRBA掩码的设定
    # @staticmethod
    # def generate_mask(mask, background_value=2):
    #     """处理RGBA掩码，根据前三个通道判断背景"""
    #     if len(mask.shape) == 3 and mask.shape[2] == 4:
    #         # 仅使用RGB通道判断背景
    #         rgb_mask = mask[..., :3]
    #         # 如果所有RGB通道都是background_value，则视为背景
    #         is_background = np.all(rgb_mask == background_value, axis=2)
    #         processed_mask = np.ones(mask.shape[:2], dtype=np.uint8) * 255
    #         processed_mask[is_background] = 0
    #         return processed_mask
    #     # 处理单通道掩码
    #     elif len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[2] == 1):
    #         processed_mask = mask.copy()
    #         if len(processed_mask.shape) == 3:
    #             processed_mask = processed_mask[..., 0]
    #         processed_mask[processed_mask == background_value] = 0
    #         return processed_mask
    #     else:
    #         raise ValueError(f"Unsupported mask shape: {mask.shape}")
    
    
    @staticmethod
    def show_image( image, title="Image"):
        """
        显示图像
        Args:
            image: 要显示的图像
            title: 图像的标题
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
 
    
################### Post-Processing part  



    #####  新的高级形态学去噪函数（基于连通域分析）  #######

    # @staticmethod
    # def advanced_denoise(input_image, binary_threshold=128, min_noise_size=50, max_hole_size=100,
    #                     opening_radius=2, closing_radius=None, invert_input=False, invert_output=False):
    #     """
    #     高级形态学去噪函数（基于连通域分析）
        
    #     参数说明：
    #     1. input_image: 输入图像变量（灰度或彩色图像）
    #     2. binary_threshold: 二值化阈值（高于此值为白色，低于为黑色）
    #     3. min_noise_size: 最小噪声面积（像素数），小于此值的孤立白点会被移除
    #     4. max_hole_size: 最大孔洞面积（像素数），小于此值的黑色孔洞会被填充
    #     5. opening_radius: 开操作结构元的圆盘半径（消除毛刺）
    #     6. closing_radius: 闭操作结构元的圆盘半径（可选，连接断裂区域）
    #     7. invert_input: 输入图像是否黑白反转（True表示黑底白字）
    #     8. invert_output: 输出图像是否黑白反转
        
    #     返回：
    #     处理后的二值图像
    #     """
    #     # 确保输入图像为灰度图
    #     if len(input_image.shape) > 2:
    #         img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #     else:
    #         img = input_image.copy()
        
    #     # 二值化
    #     _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)
        
    #     # 可选：反转图像（例如黑底白字的情况）
    #     if invert_input:
    #         binary = 255 - binary
        
    #     # 转换为布尔矩阵（True=白色，False=黑色）
    #     binary_bool = binary.astype(bool)
        
    #     # 第二步：填充小孔洞（黑色噪点）
    #     cleaned = morphology.remove_small_holes(binary_bool, area_threshold=max_hole_size)
        
    #     # 第一步：移除小面积噪声（白色噪点）
    #     filled = morphology.remove_small_objects(cleaned, min_size=min_noise_size)
        
    #     # 第三步：开操作（平滑边缘）
    #     if opening_radius > 0:
    #         smoothed = morphology.opening(filled, morphology.disk(opening_radius))
    #     else:
    #         smoothed = filled
        
    #     # 可选：闭操作（连接断裂区域）
    #     if closing_radius and closing_radius > 0:
    #         smoothed = morphology.closing(smoothed, morphology.disk(closing_radius))
        
    #     # 恢复为0-255图像
    #     result = smoothed.astype(np.uint8) * 255
        
    #     # 可选：反转输出
    #     if invert_output:
    #         result = 255 - result
        
    #     # 返回结果图像
    #     return result
    
    @staticmethod
    def advanced_denoise(input_image, binary_threshold=0.5, min_noise_size=50, max_hole_size=100,
                        opening_radius=2, closing_radius=None, invert_input=False, invert_output=False):
        """
        高级形态学去噪函数（基于连通域分析）
        
        参数说明：
        1. input_image: 输入图像变量（0-1浮点或0-255整数）
        2. binary_threshold: 二值化阈值（对应输入图像范围）
        3. min_noise_size: 最小噪声面积（像素数），小于此值的孤立白点会被移除
        4. max_hole_size: 最大孔洞面积（像素数），小于此值的黑色孔洞会被填充
        5. opening_radius: 开操作结构元的圆盘半径（消除毛刺）
        6. closing_radius: 闭操作结构元的圆盘半径（可选，连接断裂区域）
        7. invert_input: 输入图像是否黑白反转（True表示黑底白字）
        8. invert_output: 输出图像是否黑白反转
        
        返回：
        处理后的二值图像（与输入图像相同的数据类型和范围）
        """
        # 判断输入图像类型和范围
        is_float = input_image.dtype == np.float32 or input_image.dtype == np.float64
        
        # 确保输入图像为灰度图
        if len(input_image.shape) > 2:
            img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            img = input_image.copy()
        
        # 转换为整数范围进行处理
        if is_float:
            # 假设输入为0-1范围，转换为0-255
            img = (img * 255).astype(np.uint8)
            # 调整阈值
            threshold_value = int(binary_threshold * 255)
        else:
            threshold_value = binary_threshold
        
        # 二值化
        _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 调试信息
        print(f"二值化前的图像范围: {np.min(img)}-{np.max(img)}")
        print(f"二值化阈值: {threshold_value}")
        print(f"二值化后白色像素数: {np.sum(binary > 0)}")
        
        # 可选：反转图像（例如黑底白字的情况）
        if invert_input:
            binary = 255 - binary
        
        # 转换为布尔矩阵（True=白色，False=黑色）
        binary_bool = binary.astype(bool)
        
        # 第二步：填充小孔洞（黑色噪点）
        cleaned = morphology.remove_small_holes(binary_bool, area_threshold=max_hole_size)
        
        # 第一步：移除小面积噪声（白色噪点）
        filled = morphology.remove_small_objects(cleaned, min_size=min_noise_size)
        
        # 第三步：开操作（平滑边缘）
        if opening_radius > 0:
            smoothed = morphology.opening(filled, morphology.disk(opening_radius))
        else:
            smoothed = filled
        
        # 可选：闭操作（连接断裂区域）
        if closing_radius and closing_radius > 0:
            smoothed = morphology.closing(smoothed, morphology.disk(closing_radius))
        
        # 可选：反转输出
        if invert_output:
            smoothed = ~smoothed
        
        # 返回结果图像（与输入格式相同）
        if is_float:
            result = smoothed.astype(np.float32)
            return result
        else:
            result = smoothed.astype(np.uint8) * 255
            return result
    
    
    # 示例调用
    # img = cv2.imread('input.jpg')
    # result = advanced_denoise(
    #     input_image=image1,
    #     binary_threshold=128,    # 根据图像调整
    #     min_noise_size=50,       # 小于50像素的噪点被移除
    #     max_hole_size=100,       # 小于100像素的孔洞被填充
    #     opening_radius=2,        # 边缘平滑强度
    #     closing_radius=3         # 连接断裂区域（设为None则禁用）
    #     closing_radius=None,          # 闭操作结构元半径（可选，用于连接断裂区域）
    #     invert_input=False,           # 是否反转输入图像（黑/白反转）
    #     invert_output=False          # 是否反转输出图像
    # )  
    # cv2.imwrite('output.jpg', result)

##########################    
 

#######################  随机参数生成  ###############

    #生成随机参数
    def generate_random_params(self, param_ranges=None, record_seed=True):
        """
        生成随机参数
        Args:
            param_ranges: 参数范围字典，格式为 {'参数名': (最小值, 最大值)}
            record_seed: 是否记录随机种子
        Returns:
            随机参数字典
        """
        # 默认参数范围
        default_ranges = {
            'value': (-40, 40),               # 亮度调整
            'alpha': (0.8, 1.2),              # 对比度调整
            'low_percentile': (0, 10),         # 低百分位
            'high_percentile': (90, 100),     # 高百分位
            'scale': (0.7, 1.5),               # 缩放比例范围
            'rotation': [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330],# 旋转角度选项
            
            # 可以添加更多参数
        }
        
        # 使用用户提供的参数范围覆盖默认值
        if param_ranges:
            for param, range_val in param_ranges.items():
                default_ranges[param] = range_val
        
        # 设置随机种子
        if record_seed:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
        else:
            seed = None    
        
        # 生成随机参数
        random_params = {'seed': seed}
        for param, range_val in default_ranges.items():
            if param == 'rotation' :
                # 特殊处理旋转角度 - 从列表中随机选择
                random_params[param] = random.choice(range_val)
            elif isinstance(range_val, tuple):
                min_val, max_val = range_val
                if isinstance(min_val, int) and isinstance(max_val, int):
                    random_params[param] = random.randint(min_val, max_val)
                else:
                    random_params[param] = random.uniform(min_val, max_val)
        
        return random_params
    
    # 保存随机种子
    def save_params(self, params, folder='preprocessedSeed'):
        """
        保存参数到文件
        Args:
            params: 参数字典
            folder: 保存文件夹路径
        Returns:
            保存的文件路径
        """
        # 创建保存文件夹
        os.makedirs(folder, exist_ok=True)
        
        # 生成文件名，包含时间戳和种子值
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        seed = params.get('seed', 'unknown')
        filename = f"params_{timestamp}_seed{seed}.json"
        filepath = os.path.join(folder, filename)
        
        # 保存参数到JSON文件
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        
        return filepath
    # 加载随机种子
    def load_params(self, filepath):
        """
        从文件加载参数
        Args:
            filepath: 参数文件路径
        Returns:
            参数字典
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        return params

## 下面是调用示例
  
# # 生成新的随机参数进行预处理
# processed_image, processed_mask, params = processor.random_preprocess(
#     image, mask, patch_size=256, padding=0)

# # 使用已有参数文件进行预处理
# processed_image, processed_mask, params  = processor.random_preprocess(
#     image, mask, patch_size=256, padding=0, 
#     seed_file='preprocessedSeed/params_20230501_120000_seed123456789.json')

# # 自定义参数范围
# custom_ranges = {
#     'value': (-50, 50),
#     'alpha': (0.7, 1.5),
#     'rotation': (0, 90)  # 限制旋转角度在0-90度范围 [[1]](https://poe.com/citation?message_id=370359726898&citation=1)
# }
# processed_image, processed_mask, params, param_file = processor.random_preprocess(
#     image, mask, patch_size=256, padding=0, param_ranges=custom_ranges)
    
#######################


    
    #  3.26前旧版总流程(不含random)
    def preprocess(self, image, mask, patch_size ,padding=None, value=0 , alpha=1,low_percentile=1,high_percentile=99):
        """
        完整的预处理流程
        Args:
            image: 原始图像
            mask: 对应的mask
            value: 亮度调整值，正数增加亮度，负数减少亮度
            alpha: 对比度调整系数，大于1增加对比度，小于1减少对比度
        Returns:
            处理后的image和mask
        """
        
        mask_background = self.generate_mask(mask)
        
        # 1. 归一化处理
        image = self.normalize_image(image, mask_background, 
                                    low_percentile=low_percentile, 
                                    high_percentile=high_percentile)
        
        # 2. 直方图均衡化
        image = self.histogram_equalization(image, mask_background)
        
        # 3. 亮度调整
        image = self.adjust_brightness(image, value, mask_background)
        
        # 4. 对比度调整
        image = self.adjust_contrast(image, alpha, mask_background)
        
        # 处理透明边缘
        image, mask = self.handle_transparent_edges(image, mask)
        
        # 图像填充
        image, mask, _ = self.check_and_pad_image(image, mask, patch_size, padding)
        

     
        
        return image, mask
    
    
    ## 4.2单一process步骤
    def preprocess_random(self, image, mask, patch_size, padding, value,alpha,low_percentile,high_percentile,rotation,scale):
                # 创建mask_background
        mask_background = self.generate_mask(mask)
        
        # 1. 归一化处理，使用随机百分位数
        image = self.normalize_image(image, mask_background, 
                                    low_percentile=low_percentile, 
                                    high_percentile=high_percentile)
        
        # 2. 直方图均衡化
        # if enable_histo_equalization == 1:
        #      image = self.histogram_equalization(image, mask_background)

        image = self.histogram_equalization(image, mask_background)
        
        # 3. 亮度调整
        image = self.adjust_brightness(image, value, mask_background)
        
        # 4. 对比度调整
        image = self.adjust_contrast(image, alpha, mask_background)
        
        # 5. 旋转
        image, mask = self.rotate(image, mask, rotation)
        
        # 6. 缩放
        image, mask = self.scale(image, mask, scale,patch_size)
        
        # 处理透明边缘
        image, mask = self.handle_transparent_edges(image, mask)
        
        # 图像填充
        image, mask, _ = self.check_and_pad_image(image, mask, patch_size, padding)
        

        
        return image, mask
    
    ## 4.2总流程 random_preprocess
    def random_preprocess(self, image, mask, patch_size, padding=None, param_ranges=None, 
                     seed_folder='preprocessedSeed', seed_file=None):
        """
        随机参数预处理或基于已有参数文件的预处理
        
        Args:
            image: 原始图像
            mask: 对应的mask
            patch_size: patch大小
            padding: 填充值
            param_ranges: 参数范围字典，仅在需要生成新参数时使用
            seed_folder: 保存/加载参数的文件夹
            seed_file: 要使用的已有参数文件的路径，如果为None则生成新参数
            
        Returns:
            处理后的image和mask，以及参数字典和参数文件路径
        """
        # 创建参数保存文件夹
        os.makedirs(seed_folder, exist_ok=True)
        
        # 判断是使用已有参数还是生成新参数
        if seed_file and os.path.exists(seed_file):
            # 从文件加载参数
            params = self.load_params(seed_file)
            param_file = seed_file
            
            # 设置随机种子以保证一致性
            if 'seed' in params and params['seed'] is not None:
                random.seed(params['seed'])
                np.random.seed(params['seed'])
        else:
            # 生成随机参数
            params = self.generate_random_params(param_ranges, record_seed=True)
            
            # 保存参数到文件
            param_file = self.save_params(params, folder=seed_folder)
        
        # 从参数中提取值
        value = params.get('value', 0)
        alpha = params.get('alpha', 1)
        low_percentile = params.get('low_percentile', 1)
        high_percentile = params.get('high_percentile', 99)
        scale = params.get('scale', 1.0)
        rotation = params.get('rotation', 0)
        
        # 调用preprocess_random函数进行预处理，采用"先预处理后切patch"的策略 
        processed_image, processed_mask = self.preprocess_random(
            image, mask, patch_size, padding, 
            value, alpha, low_percentile, high_percentile,rotation,scale
        )
        
        return processed_image, processed_mask, params
    
    ## 自测函数
    def preprocess_show(self, image_path, mask_path, patch_size , padding=None,value=0 , alpha=1.0):
        """
        完整的预处理流程
        Args:
            image_path (str): 原始图像路径
            mask_path (str): 对应的mask路径
            patch_size (int): 图像填充的patch大小
        Returns:
            numpy.ndarray: 处理后的图像
            numpy.ndarray: 处理后的mask
            
        
        1. 归一化处理：
        基于图像的均值和标准差进行归一化
        保留图像的相对亮度信息
        防止极端值影响后续处理
        2. 直方图均衡化：
        在HSV空间进行，只调整亮度通道
        增强图像对比度，同时保留颜色信息
        特别适合处理低对比度图像
        3. 处理顺序优化：
        先进行归一化，为后续处理提供一致的基础
        然后进行直方图均衡化，增强图像细节
        最后调整亮度和对比度，达到目标效果
        """
        # 读取图像和mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #这种颜色空间转换不会导致训练带来偏差，只是改变了通道的顺序（从BGR变为RGB），没有修改像素值本身
        
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    
        # 显示原始图像和mask
        self.show_image(image, title="Original Image")
        self.show_image(mask, title="Original Mask")
        mask_background = self.generate_mask(mask)
        self.show_image(mask_background, title="Background Mask")
        

        
        # 1. 归一化处理 (屏蔽黑色背景)
        image = self.normalize_image(image, mask_background)
        
        # 2. 直方图均衡化 (屏蔽黑色背景) 
        image = self.histogram_equalization(image, mask_background)
        # 3. 亮度调整
        image = self.adjust_brightness(image, value, mask_background)
        # 4. 对比度调整
        image = self.adjust_contrast(image, alpha, mask_background)
        # 处理透明边缘
        image, mask = self.handle_transparent_edges(image, mask)
        # 图像填充
        image, mask, _ = self.check_and_pad_image(image, mask, patch_size, padding)
        

        # 显示处理后的图像和mask
        self.show_image(image, title="Processed Image")
        self.show_image(mask, title="Processed Mask")

        return image, mask, mask_background  

