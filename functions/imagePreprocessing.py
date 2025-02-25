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

# 图像归一化
def normalize(img): # 这个函数是对图像进行归一化，将像素值映射到[-1,1]之间，像素值越大，代表颜色越深，反之越浅
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img

# 图像裁剪



# 图像旋转
def rotate(img, angle, center=None, scale=1.0):
    h, w = img.shape[:2]
    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(img, M, (w, h))
    return img


# 图像缩放
def resize(img, size):
    img = cv2.resize(img, size)
    return img

