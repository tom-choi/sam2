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

class ImagePreprocessor:

    # 图像size填充处理 以及 填充像素的颜色修改 

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

#################### 以下是详细的图像预处理的函数(没说明那就是对单张大图进行处理) ##########

    # 图像归一化
    ''' 主要用途是把图片统一设置亮度、对比度、饱和度，
        使得图片的颜色分布更加均匀，从而提高模型的泛化能力。
        归一化的操作统一是对整张图片做的而不是对patche做的。
    
    '''
    def normalize(img): # 这个函数是对图像进行归一化，将像素值映射到[-1,1]之间，像素值越大，代表颜色越深，反之越浅
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img

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
    def rotate(img, angle, center=None, scale=1.0):
        h, w = img.shape[:2]
        if center is None:
            center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, (w, h))
        return img


    # 图像缩放
    ''' 主要用途是配合图像旋转，还有细胞比例的矫正，用于调用
        这里缩放同时也会强化图像的分辨率，增强模型的鲁棒性。
        图像比例需要自定，但一般由外部传入的ratio决定。
    '''
    def resize(img, size):
        img = cv2.resize(img, size)
        return img
    
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
        # 处理透明边缘
        image, mask = self.handle_transparent_edges(image, mask)
        
        # 图像填充
        image, mask = self.check_and_pad_image(image, mask,patch_size)
        
        # 图像归一化
        image = self.normalize(image)
        
        return image, mask

