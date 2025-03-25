# 这里是基于pytorch，我添加的一些图像预处理的函数，包括：
# 1. 图像归一化
# 2. 图像裁剪
# 3. 图像旋转
# 4. 图像缩放




'''
总结建议：
采用"先预处理后切patch"的策略
预处理步骤顺序：
对比度矫正（整图）
细胞占比矫正（整图）
旋转增强（整图）
Patch切分
'''
import cv2
import numpy as np
import torch
import matplotlib   
import matplotlib.pyplot as plt
import os
from skimage import morphology # for denoise

class ImagePreprocessor:

    # 图像size填充处理 以及 填充像素的颜色修改 
    @staticmethod
    def check_and_pad_image(image, mask, patch_size):
        """
        检查图片尺寸是否能被patch_size整除，如果不能则用黑色填充
        Args:
            image: 原始图像 (H, W, C)
            mask: 对应的mask (H, W)
            patch_size: patch大小
        Returns:
            处理后的image和mask
        """
        h, w = image.shape[:2]
        
        # 计算需要填充的像素数
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        if pad_h == 0 and pad_w == 0:
            return image, mask
        
        # 对图像进行填充（黑色）
        image = cv2.copyMakeBorder(image, 
                                0, pad_h, 
                                0, pad_w, 
                                cv2.BORDER_CONSTANT, 
                                value=[0, 0, 0])
        

        # 对mask进行填充（使用背景颜色）
        # 获取mask的背景颜色（通常是0）
        background_value = 0
        mask = cv2.copyMakeBorder(mask, 
                                0, pad_h, 
                                0, pad_w, 
                                cv2.BORDER_CONSTANT, 
                                value=background_value)
        
        return image, mask
    
    @staticmethod
    def handle_transparent_edges(image, mask):
        """
        处理PNG图像的透明边缘，将其转换为黑色背景
        Args:
            image: 原始图像 (H, W, C)
            mask: 对应的mask (H, W)
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
                # 转换为灰度图
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # 确保透明区域为0
                mask[mask_alpha == 0] = 0
            else:
                # 对单通道mask使用图像的alpha通道
                mask = cv2.bitwise_and(mask, mask, mask=alpha)
                # 确保透明区域为0
                mask[alpha == 0] = 0
        
        # 如果mask是4通道但图像不是
        elif len(mask.shape) == 3 and mask.shape[2] == 4:
            _, _, _, mask_alpha = cv2.split(mask)
            # 先处理mask的透明区域
            mask = cv2.bitwise_and(mask[..., :3], mask[..., :3], mask=mask_alpha)
            # 转换为灰度图
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # 确保透明区域为0
            mask[mask_alpha == 0] = 0
        
        # 将mask严格二值化
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return image, mask

     # 3.5新增内容，配合predict的部分使用，记录添加的像素位置
    
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
    @staticmethod # 此方法需要结合前后景识别生成的mask去使用,旧版本
    def histogram_equalization(image, mask=None):
        """
        改进的直方图均衡化，考虑有效区域
        Args:
            image: 原始图像 (H, W, C)
            mask: 有效区域掩码 (H, W)，None表示全图有效
        Returns:
            均衡化后的图像
        """
        # 1. 添加输入验证
        if image.dtype != np.uint8:
            raise ValueError("输入图像应为uint8类型")
        
        # 2. 处理单通道图像的情况
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[..., 2].copy()  # 3. 使用copy避免修改原数组
        
        if mask is None:
            v_channel = cv2.equalizeHist(v_channel)
        else:
            # 4. 确保mask是二值且与图像尺寸匹配
            if mask.shape != image.shape[:2]:
                raise ValueError("mask尺寸与图像不匹配")
            mask = mask.astype(bool)
            
            valid_v = v_channel[mask]
            if len(valid_v) == 0:
                return image  # 5. 无有效区域时直接返回
            
            # 6. 使用更精确的直方图计算方法
            hist = np.histogram(valid_v, bins=256, range=(0, 256))[0]
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-6)
            
            # 7. 使用查找表提高效率
            lut = np.interp(np.arange(256), np.arange(256), cdf_normalized).astype(np.uint8)
            v_channel[mask] = lut[valid_v]
        
        hsv[..., 2] = v_channel
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ###########
   
   



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
    这里的旋转是为了训练的时候增强数据集，让模型更加鲁棒，视图都是2D的，
    训练的时候可以随机旋转图像，增强数据集。
    这里的旋转是以图像中心为旋转中心，可以指定旋转中心，并且旋转后维持图像的比例，
    然后根据边缘的缺少像素来执行填补像素的function。
    这里的旋转角度可以设定，也可以随机设定。
    '''
    @staticmethod
    def rotate(img, angle, center=None, scale=1.0):
        h, w = img.shape[:2]
        if center is None:
            center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, (w, h))
        return img






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
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
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
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
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
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
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
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    





   
   
    @staticmethod # 生成前后景mask,此版本是基于轮廓检测的
    def generate_mask(image, background_threshold=10, min_contour_area=100):
        """
        改进的mask生成，基于轮廓检测
        Args:
            image: 原始图像 (H, W, C)
            background_threshold: 背景判断阈值，默认10
            min_contour_area: 最小轮廓面积，小于该值的区域将被忽略
        Returns:
            前景mask (H, W)，前景为255，背景为0
        """
        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建mask
        mask = np.zeros_like(gray)
        
        # 绘制轮廓
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask
    
    
    # 图像缩放
    ''' 主要用途是配合图像旋转，还有细胞比例的矫正，用于调用
        这里缩放同时也会强化图像的分辨率，增强模型的鲁棒性。
        图像比例需要自定，但一般由外部传入的ratio决定。
    '''
    @staticmethod
    def resize(img, size):
        img = cv2.resize(img, size)
        return img
    
####### Post-Processing part  后处理部分


    #####  新的高级形态学去噪函数（基于连通域分析）  #######

    @staticmethod
    def advanced_denoise(
        input_image,                    # 输入图像变量
        binary_threshold=128,           # 二值化阈值（0-255）
        min_noise_size=50,              # 最小噪声面积（小于此值的白色区域会被移除）
        max_hole_size=100,              # 最大孔洞面积（小于此值的黑色孔洞会被填充）
        opening_radius=2,               # 开操作结构元半径（控制边缘平滑度）
        closing_radius=None,            # 闭操作结构元半径（可选，用于连接断裂区域）
        invert_input=False,             # 是否反转输入图像（黑/白反转）
        invert_output=False             # 是否反转输出图像
    ):
        """
        高级形态学去噪函数（基于连通域分析）
        
        参数说明：
        1. input_image: 输入图像变量（灰度或彩色图像）
        2. binary_threshold: 二值化阈值（高于此值为白色，低于为黑色）
        3. min_noise_size: 最小噪声面积（像素数），小于此值的孤立白点会被移除
        4. max_hole_size: 最大孔洞面积（像素数），小于此值的黑色孔洞会被填充
        5. opening_radius: 开操作结构元的圆盘半径（消除毛刺）
        6. closing_radius: 闭操作结构元的圆盘半径（可选，连接断裂区域）
        7. invert_input: 输入图像是否黑白反转（True表示黑底白字）
        8. invert_output: 输出图像是否黑白反转
        
        返回：
        处理后的二值图像
        """
        # 确保输入图像为灰度图
        if len(input_image.shape) > 2:
            img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            img = input_image.copy()
        
        # 二值化
        _, binary = cv2.threshold(img, binary_threshold, 255, cv2.THRESH_BINARY)
        
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
        
        # 恢复为0-255图像
        result = smoothed.astype(np.uint8) * 255
        
        # 可选：反转输出
        if invert_output:
            result = 255 - result
        
        # 返回结果图像
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
    
    
#######################
    
    def show_image(self, image, title="Image"):
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
    
    # 总流程
    def preprocess(self, image, mask, patch_size , value=0 , alpha=1):
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
        
        mask_background = self.generate_mask(image, background_threshold=10)
        
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
        image, mask = self.check_and_pad_image(image, mask, patch_size)

     
        
        return image, mask
    






    # 测试单张图片的流程，用于自行查看process的效果
    def preprocess_show(self, image_path, mask_path, patch_size , value=0 , alpha=1.0):
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
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    
        # 显示原始图像和mask
        self.show_image(image, title="Original Image")
        self.show_image(mask, title="Original Mask")
        
        mask_background = self.generate_mask(image, background_threshold=10)
        self.show_image(mask_background, title="Background Mask")
        
        
        # 1. 归一化处理 (屏蔽黑色背景)
        image = self.normalize_image(image, mask_background)
        print("normalize_image done")
        # 2. 直方图均衡化 (屏蔽黑色背景) 
        image = self.histogram_equalization(image, mask_background)
        print("histogram_equalization done")
        # 3. 亮度调整
        image = self.adjust_brightness(image, value, mask_background)
        print("adjust_brightness done")
        # 4. 对比度调整
        image = self.adjust_contrast(image, alpha, mask_background)
        print("adjust_contrast done")
        # 处理透明边缘
        image, mask = self.handle_transparent_edges(image, mask)
        print("handle_transparent_edges done")  
        # 图像填充
        image, mask = self.check_and_pad_image(image, mask, patch_size)
        print("check_and_pad_image done")





        # 显示处理后的图像和mask
        self.show_image(image, title="Processed Image")
        self.show_image(mask, title="Processed Mask")

        return image, mask, mask_background