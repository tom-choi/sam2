from concurrent.futures import ThreadPoolExecutor
import os
import torch
import torch.utils.data
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random

from tqdm import tqdm # 3.5 下午新增

from functions.imagePreprocessing import ImagePreprocessor
#使用 function：from functions.data import prepare_dataset



# 加载数据集: 
#  train_data, test_data = prepare_dataset("Kasthuri++")

# def prepare_dataset(dataset_name):
#     """
#     准备指定的数据集,返回训练数据和测试数据。
    
#     参数:
#     dataset_name (str): 数据集的名称,如"Kasthuri++"。
    
#     返回:
#     train_data (list): 训练数据列表。
#     test_data (list): 测试数据列表。
#     """
#     # 数据集所在的根目录路径
#     data_dir = os.path.join("dataset", dataset_name)
    
#     # 检查数据集目录是否存在
#     if not os.path.exists(data_dir):
#         print(f"Error: 未找到 {dataset_name} 数据集目录。")
#         return [], []
    
#     # 准备训练数据
#     train_data = []
#     train_images_dir = os.path.join(data_dir, "Train_In")
#     train_masks_dir = os.path.join(data_dir, "Train_Out")
    
#     if os.path.exists(train_images_dir) and os.path.exists(train_masks_dir):
#         # 获取文件列表并确保排序一致
#         train_images = sorted(os.listdir(train_images_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
#         train_masks = sorted(os.listdir(train_masks_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        
#         # 检查文件数量是否一致
#         if len(train_images) != len(train_masks):
#             print(f"Error: Train_In 和 Train_Out 文件数量不一致。")
#             return [], []
        
#         # 按顺序配对
#         for image_file, mask_file in zip(train_images, train_masks):
#             image_path = os.path.join(train_images_dir, image_file)
#             mask_path = os.path.join(train_masks_dir, mask_file)
#             train_data.append({
#                 "image": image_path,
#                 "annotation": mask_path,
#                 "index": len(train_data)  # 添加索引信息
#             })
#     else:
#         print(f"Error: 未找到 {dataset_name} 训练数据。")
    
#     # 准备测试数据（类似修改）
#     test_data = []
#     test_images_dir = os.path.join(data_dir, "Test_In")
#     test_masks_dir = os.path.join(data_dir, "Test_Out")
    
#     if os.path.exists(test_images_dir) and os.path.exists(test_masks_dir):
#         test_images = sorted(os.listdir(test_images_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
#         test_masks = sorted(os.listdir(test_masks_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        
#         if len(test_images) != len(test_masks):
#             print(f"Error: Test_In 和 Test_Out 文件数量不一致。")
#             return [], []
        
#         for image_file, mask_file in zip(test_images, test_masks):
#             image_path = os.path.join(test_images_dir, image_file)
#             mask_path = os.path.join(test_masks_dir, mask_file)
#             test_data.append({
#                 "image": image_path,
#                 "annotation": mask_path,
#                 "index": len(test_data)
#             })
#     else:
#         print(f"Error: 未找到 {dataset_name} 测试数据。")
    
#     # 打印训练数据和测试数据
#     print(f"Train Data ({dataset_name}):", train_data)
#     print(f"Test Data ({dataset_name}):", test_data)
#     print(f"Train Data ({dataset_name}):", len(train_data))
#     print(f"Test Data ({dataset_name}):", len(test_data))
#     return train_data, test_data

def prepare_dataset(dataset_name: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    准备指定的数据集,返回训练数据和测试数据。
    
    参数:
    dataset_name (str): 数据集的名称,如"Kasthuri++"。
    
    返回:
    train_data (list): 训练数据列表。
    test_data (list): 测试数据列表。
    """
    # 数据集所在的根目录路径
    data_dir = os.path.join("dataset", dataset_name)
    
    # 检查数据集目录是否存在
    if not os.path.exists(data_dir):
        print(f"Error: 未找到 {dataset_name} 数据集目录。")
        return [], []
    
    # 准备训练数据
    train_data = []
    train_images_dir = os.path.join(data_dir, "Train_In")
    train_masks_dir = os.path.join(data_dir, "Train_Out")
    
    if os.path.exists(train_images_dir) and os.path.exists(train_masks_dir):
        # 获取文件列表并确保排序一致
        train_images = sorted(os.listdir(train_images_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        train_masks = sorted(os.listdir(train_masks_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # 检查文件数量是否一致
        if len(train_images) != len(train_masks):
            print(f"Error: Train_In 和 Train_Out 文件数量不一致。")
            return [], []
        
        # 按顺序配对
        for image_file, mask_file in zip(train_images, train_masks):
            image_path = os.path.join(train_images_dir, image_file)
            mask_path = os.path.join(train_masks_dir, mask_file)
            train_data.append({
                "image": image_path,
                "annotation": mask_path,
                "index": len(train_data)  # 添加索引信息
            })
    else:
        print(f"Error: 未找到 {dataset_name} 训练数据。")
    
    # 准备测试数据
    test_data = []
    test_images_dir = os.path.join(data_dir, "Test_In")
    test_masks_dir = os.path.join(data_dir, "Test_Out")
    
    if os.path.exists(test_images_dir) and os.path.exists(test_masks_dir):
        test_images = sorted(os.listdir(test_images_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        test_masks = sorted(os.listdir(test_masks_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        if len(test_images) != len(test_masks):
            print(f"Error: Test_In 和 Test_Out 文件数量不一致。")
            return [], []
        
        for image_file, mask_file in zip(test_images, test_masks):
            image_path = os.path.join(test_images_dir, image_file)
            mask_path = os.path.join(test_masks_dir, mask_file)
            test_data.append({
                "image": image_path,
                "annotation": mask_path,
                "index": len(test_data)
            })
    else:
        print(f"Error: 未找到 {dataset_name} 测试数据。")
    
    # 打印训练数据和测试数据
    print(f"Train Data ({dataset_name}):", train_data)
    print(f"Test Data ({dataset_name}):", test_data)
    print(f"训练样本数量: {len(train_data)}")
    print(f"测试样本数量: {len(test_data)}")
    return train_data, test_data

# 定义数据集类: 
    #  创建数据集
    # train_dataset = SegmentationDataset(train_data)
    # test_dataset = SegmentationDataset(test_data)
class SegmentationDataset(torch.utils.data.Dataset):    
    def __init__(self, data_list, patch_size=128, stride=64, preProcess=True, transform=None):
        """
        初始化数据集
        Args:
            data_list: 包含图像和标注路径的列表
            patch_size: 切片大小
            stride: 滑动窗口步长
            preProcess: 是否进行预处理
            transform: 数据增强转换
        """
        self.data_list = data_list
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.preProcess = preProcess
        self.preprocessor = ImagePreprocessor()
        
        # 预处理所有图像的patches
        self.all_patches = []
        self.all_masks = []
        self.all_positions = []
        self.all_original_sizes = []
        self.all_image_paths = []
        self.all_mask_paths = []
        self.image_indices = []  # 新增: 记录每个patch对应的原始图像索引
        
        # 预处理模块初始化
        print("开始数据集预处理...")
        
        # 处理每张图片
        for idx, item in enumerate( tqdm(data_list, desc="处理图像")):
            # 读取图像
            image = cv2.imread(item["image"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(item["annotation"], cv2.IMREAD_GRAYSCALE)
            
            # 前置处理
            if preProcess:
                image, mask = self.preprocessor.preprocess(image, mask, patch_size)
            
            # 获取图像尺寸
            h, w = image.shape[:2]
            original_size = (h, w)
            
            # 计算所有有效的patch位置
            positions = self._calculate_positions(h, w)
            
            # 处理每个位置的patch
            for y, x in positions:
                # 提取patch
                patch = image[y:y+self.patch_size, x:x+self.patch_size].copy()
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size].copy()
                
                # 数据增强
                if self.transform:
                    patch = self.transform(patch)
                
                # 转换为tensor
                patch = torch.FloatTensor(patch.transpose(2, 0, 1)) / 255.0
                mask_patch = torch.FloatTensor(mask_patch).unsqueeze(0) / 255.0
                
                # 存储结果
                self.all_patches.append(patch)
                self.all_masks.append(mask_patch)
                self.all_positions.append((y, x))
                self.all_original_sizes.append(original_size)
                self.all_image_paths.append(item["image"])
                self.all_mask_paths.append(item["annotation"])
                self.image_indices.append(idx)  # 记录原始图像索引
        
        print(f"数据集预处理完成，共生成 {len(self.all_patches)} 个patch")

    def _calculate_positions(self, h, w):
        """计算图像中所有有效的patch位置"""
        positions = []
        
        # 横向和纵向的滑动位置
        h_idx = np.arange(0, h-self.patch_size+1, self.stride)
        w_idx = np.arange(0, w-self.patch_size+1, self.stride)
        
        # 确保处理到边缘
        if h % self.stride != 0 and h > self.patch_size:
            h_idx = np.append(h_idx, h-self.patch_size)
        if w % self.stride != 0 and w > self.patch_size:
            w_idx = np.append(w_idx, w-self.patch_size)
        
        # 生成所有位置组合
        for y in h_idx:
            for x in w_idx:
                positions.append((int(y), int(x)))
                
        return positions
    
    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        return {
            'patches': self.all_patches[idx],
            'mask_patches': self.all_masks[idx],
            'positions': self.all_positions[idx],
            'original_size': self.all_original_sizes[idx],
            'image_path': self.all_image_paths[idx],
            'mask_path': self.all_mask_paths[idx],
            'image_idx': self.image_indices[idx]  # 新增: 返回原始图像索引
        }
        
    def get_stats(self):
        """返回数据集的统计信息"""
        unique_images = len(set(self.image_indices))
        patches_per_image = len(self.all_patches) / unique_images if unique_images > 0 else 0
        
        return {
            'total_patches': len(self.all_patches),
            'unique_images': unique_images,
            'patches_per_image': patches_per_image,
            'patch_size': self.patch_size,
            'stride': self.stride
        }
        
    def filter_patches(self, min_content=0.05):
        """过滤掉内容很少的patch（可选）"""
        filtered_indices = []
        
        for i, mask in enumerate(self.all_masks):
            # 计算mask中前景像素比例
            foreground_ratio = mask.mean().item()
            if foreground_ratio >= min_content:
                filtered_indices.append(i)
        
        # 创建新的过滤后的数据列表
        self.all_patches = [self.all_patches[i] for i in filtered_indices]
        self.all_masks = [self.all_masks[i] for i in filtered_indices]
        self.all_positions = [self.all_positions[i] for i in filtered_indices]
        self.all_original_sizes = [self.all_original_sizes[i] for i in filtered_indices]
        self.all_image_paths = [self.all_image_paths[i] for i in filtered_indices]
        self.all_mask_paths = [self.all_mask_paths[i] for i in filtered_indices]
        self.image_indices = [self.image_indices[i] for i in filtered_indices]
        
        print(f"过滤后保留了 {len(self.all_patches)} 个patch（原有 {len(filtered_indices)} 个）")
    
    def read_image(self, idx):  # read and resize image and mask
        return self.all_image_paths[idx], self.all_mask_paths[idx]

    @staticmethod
    def verify_data_list(data_list: List[Dict[str, str]]) -> bool:
        """验证数据列表格式是否正确"""
        for item in data_list:
            if not isinstance(item, dict):
                return False
            if "image" not in item or "annotation" not in item:
                return False
            if not os.path.exists(item["image"]) or not os.path.exists(item["annotation"]):
                return False
        return True

    @staticmethod
    def create_data_list(image_dir: str, mask_dir: str) -> List[Dict[str, str]]:
        """
        创建数据列表的辅助函数
        Args:
            image_dir: 图像目录
            mask_dir: 掩码目录
        Returns:
            数据列表
        """
        data_list = []
        for img_name in os.listdir(image_dir):
            mask_name = img_name  # 假设掩码和图像同名
            data_list.append({
                "image": os.path.join(image_dir, img_name),
                "annotation": os.path.join(mask_dir, mask_name)
            })
        return data_list
    
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
        
        # 对mask进行填充（黑色）
        mask = cv2.copyMakeBorder(mask, 
                               0, pad_h, 
                               0, pad_w, 
                               cv2.BORDER_CONSTANT, 
                               value=0)
        
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
        # 如果图像没有alpha通道，直接返回
        if image.shape[2] != 4:
            return image, mask
            
        # 分离alpha通道
        b, g, r, alpha = cv2.split(image)
        
        # 创建黑色背景
        black_background = np.zeros_like(image[..., :3])
        
        # 将图像合成到黑色背景上
        image = cv2.bitwise_and(image[..., :3], image[..., :3], mask=alpha)
        
        # 对mask进行相同处理
        mask = cv2.bitwise_and(mask, mask, mask=alpha)
        
        return image, mask

    def process_image_to_patches(self, image, mask):
        image, mask = self.handle_transparent_edges(image, mask)
        h, w = image.shape[:2]
        
        h_idx = np.arange(0, h-self.patch_size+1, self.stride)
        w_idx = np.arange(0, w-self.patch_size+1, self.stride)
        
        if h % self.stride != 0:
            h_idx = np.append(h_idx, h-self.patch_size)
        if w % self.stride != 0:
            w_idx = np.append(w_idx, w-self.patch_size)
        
        y_coords, x_coords = np.meshgrid(h_idx, w_idx, indexing='ij')
        positions = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
        
        def extract_patch(pos):
            y, x = pos
            return (image[y:y+self.patch_size, x:x+self.patch_size],
                    mask[y:y+self.patch_size, x:x+self.patch_size])
        
        # 並行處理提取patches
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(extract_patch, positions))
        
        patches, mask_patches = zip(*results)
        return np.array(patches), np.array(mask_patches), positions.tolist(), (h, w)
    
    # def process_image_to_patches(self, image, mask):
        
    #     # 先处理透明边缘
    #     image, mask = self.handle_transparent_edges(image, mask)
    #     """处理图像和掩码为patches"""
    #     h, w = image.shape[:2]
    #     patches = []
    #     mask_patches = []
    #     positions = []
        
    #     for y in range(0, h-self.patch_size+1, self.stride):
    #         for x in range(0, w-self.patch_size+1, self.stride):
    #             patch = image[y:y+self.patch_size, x:x+self.patch_size]
    #             mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
    #             patches.append(patch)
    #             mask_patches.append(mask_patch)
    #             positions.append([y, x])  # 列表
        
    #     # 处理边缘情况
    #     if h % self.stride != 0:
    #         y = h - self.patch_size
    #         for x in range(0, w-self.patch_size+1, self.stride):
    #             patch = image[y:y+self.patch_size, x:x+self.patch_size]
    #             mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
    #             patches.append(patch)
    #             mask_patches.append(mask_patch)
    #             positions.append([y, x])
        
    #     if w % self.stride != 0:
    #         x = w - self.patch_size
    #         for y in range(0, h-self.patch_size+1, self.stride):
    #             patch = image[y:y+self.patch_size, x:x+self.patch_size]
    #             mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
    #             patches.append(patch)
    #             mask_patches.append(mask_patch)
    #             positions.append([y, x])
        
    #     if h % self.stride != 0 and w % self.stride != 0:
    #         y = h - self.patch_size
    #         x = w - self.patch_size
    #         patch = image[y:y+self.patch_size, x:x+self.patch_size]
    #         mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
    #         patches.append(patch)
    #         mask_patches.append(mask_patch)
    #         positions.append([y, x])
        
    #     return patches, mask_patches, positions, (h, w)
    
    def visualize_image_patches(self, image_path, num_patches=5):
        """可视化指定图片的patches分布"""
        try:
            # 找到指定图片在Train_In中的索引位置
            train_in_paths = [item["image"] for item in self.data_list]
            train_in_idx = train_in_paths.index(image_path)
            
            # 使用相同的索引获取Train_Out中的mask
            mask_path = self.data_list[train_in_idx]["annotation"]
            
            print(f"原图路径: {image_path}")
            print(f"对应的Mask路径: {mask_path}")
            print(f"图片索引: {train_in_idx}")  # 打印索引以便验证
            
            # 找到指定图片的所有patches的索引
            image_indices = [i for i, path in enumerate(self.all_image_paths) if path == image_path]
            
            if not image_indices:
                print(f"未找到图片的patches: {image_path}")
                return
            
            # 读取原始图片和对应的mask
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"无法读取图片: {image_path}")
                return
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if original_mask is None:
                print(f"无法读取mask: {mask_path}")
                return
                
            # 确保mask是二值图像
            _, original_mask = cv2.threshold(original_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 随机选择要显示的patches
            if len(image_indices) <= num_patches:
                selected_indices = image_indices  # 如果patches数量不足，选择所有
            else:
                selected_indices = random.sample(image_indices, num_patches)  # 随机选择num_patches个patch
            num_selected = len(selected_indices)
            
            # 创建图像网格
            plt.figure(figsize=(20, 8))
            
            # 显示原始图像和patches的位置
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Original Image with Patch Locations")
            
            # 显示二值mask和patches的位置
            plt.subplot(1, 2, 2)
            plt.imshow(original_mask, cmap='gray')  # 使用gray颜色映射
            plt.title("Gray Mask with Patch Locations (White: Cell)")
            
            # 在原图和mask上都标注patch位置
            colors = plt.cm.rainbow(np.linspace(0, 1, num_selected))
            for subplot_idx in [1, 2]:
                plt.subplot(1, 2, subplot_idx)
                
                for idx, patch_idx in enumerate(selected_indices):
                    pos = self.all_positions[patch_idx]
                    
                    if isinstance(pos[0], (tuple, list)):
                        y, x = pos[0]
                    else:
                        y, x = pos
                        
                    rect = plt.Rectangle(
                        xy=(int(x), int(y)),
                        width=self.patch_size,
                        height=self.patch_size,
                        fill=False,
                        color=colors[idx],
                        linewidth=2
                    )
                    plt.gca().add_patch(rect)
                    
                    plt.text(
                        int(x), int(y),
                        str(idx+1), 
                        color=colors[idx], 
                        fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # 显示每个patch的详细信息
            for idx, patch_idx in enumerate(selected_indices):
                item = self.__getitem__(patch_idx)
                
                # 转换数据格式
                patch = item['patches'].numpy()
                mask = item['mask_patches'].numpy()
                
                patch = np.transpose(patch, (1,2,0))
                mask = np.squeeze(mask)
                
                # 还原归一化
                patch = (patch * 255).astype(np.uint8)
                mask = (mask * 255).astype(np.uint8)
                
                # 创建子图
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                fig.suptitle(f'Patch {idx+1}')
                
                # 显示原始patch
                axes[0].imshow(patch)
                axes[0].set_title('Image Patch')
                axes[0].axis('off')
                
                # 显示二值mask
                axes[1].imshow(mask, cmap='gray')  # 使用gray颜色映射
                axes[1].set_title('Gray Mask (White: Cell)')
                axes[1].axis('off')
                
                # 显示叠加效果（红色表示细胞区域）
                overlay = patch.copy()
                overlay[mask > 0] = [255, 0, 0]  # 用红色标注细胞区域
                axes[2].imshow(overlay)
                axes[2].set_title('Overlay (Red: Cell)')
                axes[2].axis('off')
                
                plt.show()
                
        except Exception as e:
            print(f"可视化过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def visualize_random_images(self, num_images=3, patches_per_image=5):
        """
        随机选择几张图片并显示它们的patches
        
        Args:
            num_images (int): 要显示的图片数量
            patches_per_image (int): 每张图片显示的patch数量
        """
        # 获取唯一的图片路径
        unique_images = list(set(self.all_image_paths))
        
        # 随机选择图片
        selected_images = random.sample(unique_images, min(num_images, len(unique_images)))
        
        # 显示每张图片的patches
        for image_path in selected_images:
            print(f"\n显示图片的patches: {os.path.basename(image_path)}")
            self.visualize_image_patches(image_path, patches_per_image)
        
    def visualize_item(self, idx, patches_per_image=5):
        """
        可视化单个样本
        Args:
            idx: 样本的索引
            patches_per_image: 每个图像要显示的patches数量，默认为5
        """
        # 使用cv2加载图像
        image_path = self.all_image_paths[idx]
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"无法读取图片: {image_path}")
            return
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 显示图像
        cv2.imshow(f"Original Image {os.path.basename(image_path)}", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 显示图片的patches
        print(f"\n显示图片的patches: {os.path.basename(image_path)}")
        self.visualize_image_patches(image_path, patches_per_image)



    def visualize_batch(self, start_idx=0, num_samples=5):
        """
        可视化一批数据
        Args:
            start_idx: 起始索引
            num_samples: 要显示的样本数量
        """
        for i in range(start_idx, min(start_idx + num_samples, len(self))):
            print(f"\nSample {i}:")
            print(f"Image path: {self.all_image_paths[i]}")
            self.visualize_item(i)

    def print_dataset_info(self):
        """
        打印数据集基本信息
        """
        print("数据集信息:")
        print(f"总样本数: {len(self)}")
        print(f"图像块大小: {self.patch_size}x{self.patch_size}")
        print(f"滑动步长: {self.stride}")
        print(f"是否使用数据增强: {'是' if self.transform else '否'}")
        print("\n数据形状:")
        print(f"图像块形状: {self.all_patches[0].shape}")
        print(f"掩码块形状: {self.all_masks[0].shape}")




