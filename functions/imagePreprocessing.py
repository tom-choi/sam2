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

    ################### 以下是详细的图像预处理的函数(没说明那就是对单张大图进行处理) ##########

    # 图像归一化
    ''' 主要用途是把图片统一设置亮度、对比度、饱和度，
        使得图片的颜色分布更加均匀，从而提高模型的泛化能力。
        归一化的操作统一是对整张图片做的而不是对patche做的。
    
    '''


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


#################################################################
#下面的是没有前后景，直接进行整张图片的处理，可能带来偏差

    # @staticmethod
    # def adjust_brightness(image, value=0):
    #     """
    #     调整图像的亮度
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         value: 亮度调整值，正数增加亮度，负数减少亮度
    #     Returns:
    #         调整亮度后的图像
    #     """
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     h, s, v = cv2.split(hsv)

    #     # 调整亮度
    #     lim = 255 - value
    #     v[v > lim] = 255
    #     v[v <= lim] = np.clip(v[v <= lim] + value, 0, 255)

    #     final_hsv = cv2.merge((h, s, v))
    #     image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    #     # 确保图像数据类型为 uint8
    #     image = image.astype(np.uint8)

    #     return image

    # @staticmethod
    # def adjust_contrast(image, alpha=1.0):
    #     """
    #     调整图像的对比度
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         alpha: 对比度调整系数，大于1增加对比度，小于1减少对比度
    #     Returns:
    #         调整对比度后的图像
    #     """
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     image[..., 2] = cv2.convertScaleAbs(image[..., 2], alpha=alpha, beta=0)
    #     image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    #     return image

    # @staticmethod
    # def histogram_equalization(image):
    #     """
    #     直方图均衡化
    #     Args:
    #         image: 原始图像 (H, W, C)
    #     Returns:
    #         均衡化后的图像
    #     """
    #     # 将图像转换为HSV空间
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
    #     # 对V通道进行直方图均衡化
    #     hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
        
    #     # 转换回RGB空间
    #     image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #     return image
    
    
    # @staticmethod
    # def normalize_image(image):
    #     """
    #     归一化图像
    #     Args:
    #         image: 原始图像 (H, W, C)
    #     Returns:
    #         归一化后的图像
    #     """
    #     # 将图像转换为float32类型
    #     image = image.astype(np.float32)
        
    #     # 计算均值和标准差
    #     mean = np.mean(image, axis=(0, 1))
    #     std = np.std(image, axis=(0, 1))
        
    #     # 归一化处理
    #     image = (image - mean) / (std + 1e-7)
        
    #     # 将值映射到[0, 255]
    #     image = np.clip(image * 255, 0, 255)
        
    #     return image.astype(np.uint8)

###################################################################
# 下面的是加入前后景判断的所有处理效果

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
    
    
    @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    def normalize_image(image, mask=None):
        """
        改进的归一化处理，考虑有效区域
        Args:
            image: 原始图像 (H, W, C)
            mask: 有效区域掩码 (H, W)，None表示全图有效
        Returns:
            归一化后的图像
        """
        image = image.astype(np.float32)
        
        if mask is None:
            # 如果没有提供mask，使用全图
            mean = np.mean(image, axis=(0, 1))
            std = np.std(image, axis=(0, 1))
        else:
            # 只计算有效区域的统计量
            mask = mask.astype(bool)
            mean = []
            std = []
            for c in range(image.shape[2]):
                channel = image[..., c]
                mean.append(np.mean(channel[mask]))
                std.append(np.std(channel[mask]))
            mean = np.array(mean)
            std = np.array(std)
        
        # 归一化处理
        image = (image - mean) / (std + 1e-7)
        image = np.clip(image * 255, 0, 255)
        
        return image.astype(np.uint8)
    
 
    
    # @staticmethod # 此方法需要结合前后景识别生成的mask去使用
    # def histogram_equalization(image, mask=None):
    #     """
    #     改进的直方图均衡化，考虑有效区域
    #     Args:
    #         image: 原始图像 (H, W, C)
    #         mask: 有效区域掩码 (H, W)，None表示全图有效
    #     Returns:
    #         均衡化后的图像
    #     """
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
    #     if mask is None:
    #         # 如果没有提供mask，使用全图
    #         hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    #     else:
    #         # 只对有效区域进行均衡化
    #         v_channel = hsv[..., 2]
    #         valid_v = v_channel[mask]
            
    #         # 计算直方图
    #         hist = cv2.calcHist([valid_v], [0], None, [256], [0, 256])
            
    #         # 计算累积分布函数
    #         cdf = hist.cumsum()
    #         cdf_normalized = cdf * 255 / cdf[-1]
            
    #         # 应用均衡化
    #         v_channel[mask] = np.interp(valid_v, np.arange(256), cdf_normalized)
    #         hsv[..., 2] = v_channel
        
    #     return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    @staticmethod # 改进尝试
    def histogram_equalization(image, mask=None):
        """
        改进的直方图均衡化，考虑有效区域
        Args:
            image: 原始图像 (H, W, C)
            mask: 有效区域掩码 (H, W)，None表示全图有效
        Returns:
            均衡化后的图像
        """
        # 只转换一次颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v_channel = hsv[..., 2]
        
        if mask is None:
            # 如果没有提供mask，使用全图
            v_channel = cv2.equalizeHist(v_channel)
        else:
            # 只对有效区域进行均衡化
            mask = mask.astype(bool)
            
            # 保存原始有效区域的V通道值
            valid_v = v_channel[mask]
            
            # 计算直方图
            hist = cv2.calcHist([valid_v], [0], None, [256], [0, 256])
            
            # 计算累积分布函数
            cdf = hist.cumsum()
            
            # 避免除以0的情况，确保cdfs[-1]不为0
            cdf_normalized = cdf * 255 / (cdf[-1] if cdf[-1] != 0 else 1)
            
            # 应用均衡化，直接在原数组上进行操作
            v_channel[mask] = np.interp(valid_v, np.arange(256), cdf_normalized)
        
        # 最后一次颜色空间转换
        hsv[..., 2] = v_channel
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image.astype(np.uint8)

    
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
    def preprocess(self, image, mask,patch_size):
        """
        完整的预处理流程
        Args:
            image: 原始图像
            mask: 对应的mask
        Returns:
            处理后的image和mask
        """
        
        # 先调整亮度和对比度
        
        # 图像旋转 和 缩放
        
        # 处理透明边缘
        image, mask = self.handle_transparent_edges(image, mask)
        
        # 图像填充
        image, mask = self.check_and_pad_image(image, mask,patch_size)
        

     
        
        return image, mask
    
    
    # 测试流程，用于自行查看process的效果,这里没有加入前后景限定区域
    # def preprocess_show(self, image_path, mask_path, patch_size , value=100 , alpha=0.9):
    #     """
    #     完整的预处理流程
    #     Args:
    #         image_path (str): 原始图像路径
    #         mask_path (str): 对应的mask路径
    #         patch_size (int): 图像填充的patch大小
    #     Returns:
    #         numpy.ndarray: 处理后的图像
    #         numpy.ndarray: 处理后的mask
            
            
    #     1. 归一化处理：
    #     基于图像的均值和标准差进行归一化
    #     保留图像的相对亮度信息
    #     防止极端值影响后续处理
    #     2. 直方图均衡化：
    #     在HSV空间进行，只调整亮度通道
    #     增强图像对比度，同时保留颜色信息
    #     特别适合处理低对比度图像
    #     3. 处理顺序优化：
    #     先进行归一化，为后续处理提供一致的基础
    #     然后进行直方图均衡化，增强图像细节
    #     最后调整亮度和对比度，达到目标效果
    #     """
    #     # 读取图像和mask
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    
    #     # 显示原始图像和mask
    #     self.show_image(image, title="Original Image")
    #     self.show_image(mask, title="Original Mask")
        
    #     # 1. 归一化处理
    #     image = self.normalize_image(image)
        
    #     # 2. 直方图均衡化
    #     image = self.histogram_equalization(image)
        
    #     # 3. 亮度调整
    #     image = self.adjust_brightness(image, value)
        
    #     # 4. 对比度调整
    #     image = self.adjust_contrast(image, alpha)
        
    #     # 处理透明边缘
    #     image, mask = self.handle_transparent_edges(image, mask)

    #     # 图像填充
    #     image, mask = self.check_and_pad_image(image, mask, patch_size)






    #     # 显示处理后的图像和mask
    #     self.show_image(image, title="Processed Image")
    #     self.show_image(mask, title="Processed Mask")

    #     return image, mask

    # 加入前后景设置的流程
    def preprocess_show(self, image_path, mask_path, patch_size , value=100 , alpha=0.9):
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
        # 2. 直方图均衡化 (屏蔽黑色背景) --- 这个花费的时间很多，其它步骤都非常快
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


