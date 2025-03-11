import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from skimage.morphology import remove_small_objects, disk
from scipy.ndimage import binary_dilation, binary_erosion
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage import measure

i = 0

def read_batch(data, visualize_data=False):
    # 選擇隨機條目
    ent = data[np.random.randint(len(data))]

    # 讀取圖像
    Img = cv2.imread(ent["image"])[..., ::-1]  # 轉換BGR為RGB
    ann_map = cv2.imread(ent["remove_small_objectsannotation"], cv2.IMREAD_GRAYSCALE)  # 以灰度圖讀取註釋

    if Img is None or ann_map is None:
        print(f"錯誤：無法從路徑 {ent['image']} 或 {ent['annotation']} 讀取圖像或遮罩")
        return None, None, None, 0

    # 調整圖像和遮罩大小
    r = min(1024 / Img.shape[1], 1024 / Img.shape[0])  # 縮放因子
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 初始化二值遮罩
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)

    # 獲取二值遮罩並合併為單一遮罩
    inds = np.unique(ann_map)[1:]  # 跳過背景（索引0）
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # 為每個唯一索引創建二值遮罩
        binary_mask = np.maximum(binary_mask, mask)  # 與現有二值遮罩合併

    # 腐蝕合併的二值遮罩以避免邊界點
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    # 使用連通區域分析來找到所有獨立的白色區域
    labels = measure.label(eroded_mask)
    regions = measure.regionprops(labels)

    points = []
    for region in regions:
        # 為每個區域選擇一個隨機點
        y, x = region.coords[np.random.randint(len(region.coords))]
        points.append([x, y])  # 注意：我們存儲為 [x, y] 以與原始代碼保持一致

    points = np.array(points)

    if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(Img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')  # Corrected to plot y, x order

        # plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # 現在形狀是 (1024, 1024, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    # 返回圖像、二值化遮罩、點和遮罩數量
    return Img, binary_mask, points, len(inds)

# Visualize the data
# Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)

# 可修改的參數
PATCH_SIZE = 100  # Patch大小,影響分割精度和速度。更大的patch可能提高精度,但會增加計算時間。
MIN_SIZE = 25  # 移除小於此尺寸的白色區域。更大的值會移除更多的小區域。
SELEM_RADIUS = 4  # 形態學操作中圓形結構元素的半徑。更大的半徑會導致更多的平滑。
OVERLAP_THRESHOLD = 0.15  # 用於判斷是否保留重疊區域的閾值。更高的值會保留更多的重疊區域。

def read_batch_speical(Img, ann_map, visualize_data=False):
    # 調整圖像和遮罩大小
    r = min(1334 / Img.shape[1], 1334 / Img.shape[0])  # 縮放因子
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 初始化二值遮罩
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)

    # 獲取二值遮罩並合併為單一遮罩
    inds = np.unique(ann_map)[1:]  # 跳過背景（索引0）
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask)

    # 先进行后处理
    # 1. 移除小物体
    cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=MIN_SIZE, connectivity=2)
    
    # 2. 创建圆形结构元素并进行闭运算
    selem = disk(SELEM_RADIUS)
    processed_mask = binary_erosion(binary_dilation(cleaned_mask, selem), selem)
    
    # 3. 腐蝕處理過的遮罩以避免邊界點
    eroded_mask = cv2.erode(processed_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

    # 使用連通區域分析來找到所有獨立的白色區域
    labels = measure.label(eroded_mask)
    regions = measure.regionprops(labels)

    points = []
    for region in regions:
        # 使用区域质心作为点位置
        y, x = region.centroid
        # 将质心四舍五入到最近的整数坐标
        x, y = int(round(x)), int(round(y))
        
        # 确保点在区域内部
        if eroded_mask[y, x] == 0:
            # 如果质心不在区域内，找到最近的区域内点
            coords = region.coords
            distances = np.sqrt((coords[:, 0] - y)**2 + (coords[:, 1] - x)**2)
            nearest_idx = np.argmin(distances)
            y, x = coords[nearest_idx]
        
        points.append([x, y])

    points = np.array(points)

    if visualize_data:
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask after Morphological Operations
        plt.subplot(1, 3, 2)
        plt.title('Mask after Morphological Operations')
        plt.imshow(processed_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points
        plt.subplot(1, 3, 3)
        plt.title('Mask with Points')
        plt.imshow(processed_mask, cmap='gray')
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # 使用处理过的mask作为返回值
    binary_mask = processed_mask.astype(np.uint8)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    return Img, binary_mask, points, len(inds)

def read_image(image_path, mask_path):  # read and resize image and mask
   img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
   mask = cv2.imread(mask_path, 0)
   r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
   img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
   mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
   return img, mask

def get_points(mask, num_points):  # Sample points inside the input mask
   points = []
   coords = np.argwhere(mask > 0)
   for i in range(num_points):
       yx = np.array(coords[np.random.randint(len(coords))])
       points.append([[yx[1], yx[0]]])
   return np.array(points)
# image 為整個圖像
#
# 把获取预测的mask，扔到整個sam預測加強流程裡面
# 获取预测的mask
# predicted_mask = get_prediction_mask(model, image)

def build_sam2_model(
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt", 
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device="cuda"
    ):
    print("Creating SAM2 segmentation Model...")
    model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    print("SAM2 segmentation Model Created!")
    return model

def main_prediction_process(
        sam2_model, image, predicted_mask
    ):

    i = 1 # ON9

    predictor2 = SAM2ImagePredictor(sam2_model)


    # Generate random points for the input
    _, _, input_points, _ = read_batch_speical(image, predicted_mask, True) # predicted_mask ---> org_mask
    
    if (len(input_points) <= 1):
        return None, None, None, None

    # Perform inference and predict masks
    # 用box(整個圖像)的形式輸入prompt
    with torch.no_grad():
        predictor2.set_image(image)
        masks, scores, logits = predictor2.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Process the predicted masks and sort by scores
    np_masks_2 = np.array(masks[:, 0])
    np_scores_2 = scores[:, 0]
    sorted_masks_2 = np_masks_2[np.argsort(np_scores_2)][::-1]

    # Initialize segmentation map and occupancy mask
    occupancy_mask_2 = np.zeros_like(sorted_masks_2[0], dtype=bool)

    # Combine masks to create the final segmentation map
    for i in range(sorted_masks_2.shape[0]):
        mask_2 = sorted_masks_2[i]
        if (mask_2 * occupancy_mask_2).sum() / mask_2.sum() > OVERLAP_THRESHOLD:
            continue

        mask_bool_2 = mask_2.astype(bool)
        mask_bool_2[occupancy_mask_2] = False  # Set overlapping areas to False in the mask
        occupancy_mask_2[mask_bool_2] = True  # Update occupancy_mask

    # 移除小的白色區域
    seg_map2_cleaned = remove_small_objects(occupancy_mask_2, min_size=MIN_SIZE, connectivity=2)

    # 創建圓形結構元素
    selem = disk(SELEM_RADIUS)

    # 執行閉運算 (先膨脹後腐蝕)
    seg_map2_closed = binary_erosion(binary_dilation(seg_map2_cleaned, selem), selem)

    # 將 seg_map2_closed 轉換回原始數據類型
    seg_map2_final = np.zeros_like(seg_map2_closed, dtype=np.uint8)

    # 給mask上色
    for i in range(sorted_masks_2.shape[0]):
        mask_2 = sorted_masks_2[i]
        mask_bool_2 = mask_2.astype(bool)
        mask_bool_2 = mask_bool_2 & seg_map2_closed  # 只保留在 seg_map2_closed 中的白色區域
        seg_map2_final[mask_bool_2] = i + 1

    

    # 设置图像显示大小
    plt.figure(figsize=(15, 5))

    # 显示原始图像
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # 显示二值化mask
    plt.subplot(132)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Binarized Mask')
    plt.axis('off')

    # 显示带颜色标记的最终分割结果
    plt.subplot(133)
    plt.imshow(seg_map2_final, cmap='tab20')  # 使用tab20颜色图来显示不同的标签
    plt.title('Binarized Mask with Points')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(15, 5))

    # 显示清理小物体后的结果
    plt.subplot(131)
    plt.imshow(seg_map2_cleaned, cmap='gray')
    plt.title('After Remove Small Objects')
    plt.axis('off')

    # 显示膨胀操作后的结果
    plt.subplot(132)
    plt.imshow(binary_dilation(seg_map2_cleaned, selem), cmap='gray')
    plt.title('After Dilation')
    plt.axis('off')

    # 显示最终的闭运算结果
    plt.subplot(133)
    plt.imshow(seg_map2_closed, cmap='gray')
    plt.title('After Closing Operation')
    plt.axis('off')     

    plt.tight_layout()
    plt.show()

    # save as final_segmentation.jpg
    cv2.imwrite('final_segmentation.jpg', seg_map2_final.astype(np.uint8) * 255)

    



