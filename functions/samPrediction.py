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
MIN_SIZE = 50  # 移除小於此尺寸的白色區域。更大的值會移除更多的小區域。
SELEM_RADIUS = 4  # 形態學操作中圓形結構元素的半徑。更大的半徑會導致更多的平滑。
OVERLAP_THRESHOLD = 0.00  # 用於判斷是否保留重疊區域的閾值。更高的值會保留更多的重疊區域。

SCORE_THRESHOLD = 0.05  # 預測分數閾值

MIN_POINT_DISTANCE = 20  # 點之間的最小距離閾值

# 新增参数配置
N_NEGATIVE_POINTS = 25  # 负点采样数量
MIN_NEGATIVE_DISTANCE = 5  # 负点与正点的最小距离

def read_batch_speical(Img, ann_map, visualize_data=False):
    # 調整圖像和遮罩大小
    r = min(1024 / Img.shape[1], 1024 / Img.shape[0])  # 縮放因子
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
    # 生成正点（前景点）
    labels = measure.label(eroded_mask)
    regions = measure.regionprops(labels)
    
    # 优化后的正点采样逻辑
    selected_positive_points = []
    for region in regions:
        # 动态距离阈值（基于区域面积）
        dynamic_distance = max(15, int(0.08 * np.sqrt(region.area)))
        
        # 获取区域代表点
        y, x = region.centroid
        x, y = int(round(x)), int(round(y))
        if eroded_mask[y, x] == 0:
            coords = region.coords
            distances = np.sqrt((coords[:, 0] - y)**2 + (coords[:, 1] - x)**2)
            nearest_idx = np.argmin(distances)
            y, x = coords[nearest_idx]
            x, y = int(x), int(y)
        
        current_point = np.array([x, y])
        
        # 距离检查
        too_close = any(np.linalg.norm(current_point - p) < dynamic_distance for p in selected_positive_points)
        if not too_close:
            selected_positive_points.append(current_point)
    
    positive_points = np.array(selected_positive_points)

    # 修改后的负点采样逻辑（替换原有负点生成部分）
    # 創建距離圖，用於快速篩選合適的負點
    distance_map = np.zeros_like(eroded_mask, dtype=np.float32)
    for point in selected_positive_points:
        y, x = point[1], point[0]
        y_coords, x_coords = np.ogrid[:eroded_mask.shape[0], :eroded_mask.shape[1]]
        distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        distance_map = np.maximum(distance_map, distances)

    # 找出所有符合最小距離要求的背景點
    valid_background_mask = (distance_map >= MIN_NEGATIVE_DISTANCE) & (eroded_mask == 0)
    valid_coords = np.argwhere(valid_background_mask)

    # 如果有足夠的有效點，隨機選擇
    if len(valid_coords) >= N_NEGATIVE_POINTS:
        selected_indices = np.random.choice(len(valid_coords), N_NEGATIVE_POINTS, replace=False)
        negative_points = valid_coords[selected_indices]
        # 交換x,y坐標順序
        negative_points = np.fliplr(negative_points)
    else:
        # 如果有效點不足，使用所有有效點
        negative_points = np.fliplr(valid_coords)
        # 需要額外的點時，放寬距離限制
        remaining = N_NEGATIVE_POINTS - len(negative_points)
        if remaining > 0:
            background_mask = (eroded_mask == 0)
            remaining_coords = np.argwhere(background_mask)
            remaining_indices = np.random.choice(len(remaining_coords), remaining, replace=False)
            additional_points = np.fliplr(remaining_coords[remaining_indices])
            negative_points = np.vstack([negative_points, additional_points])
    
    negative_points = np.array(negative_points)
    
    # 组合正负点并创建标签
    # 组合所有提示点并创建标签数组
    all_points = np.vstack([positive_points, negative_points])
    point_labels = np.array([1]*len(positive_points) + [0]*len(negative_points)).reshape(-1, 1)

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
        for i, point in enumerate(all_points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

        
        plt.figure(figsize=(10, 5))
        plt.imshow(eroded_mask, cmap='gray')
        
        # 绘制正点（绿色）和负点（红色）
        if len(positive_points) > 0:
            plt.scatter(positive_points[:,0], positive_points[:,1], c='lime', s=50, 
                       edgecolors='white', linewidths=1, label='Positive Points')
        if len(negative_points) > 0:
            plt.scatter(negative_points[:,0], negative_points[:,1], c='red', s=30,
                       marker='x', linewidths=1, label='Negative Points')
        
        plt.legend()
        plt.title('Positive/Negative Points Visualization')
        plt.axis('off')
        plt.show()


    # 使用处理过的mask作为返回值
    binary_mask = processed_mask.astype(np.uint8)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = binary_mask.transpose((2, 0, 1))

    # 调整维度以匹配SAM输入格式
    all_points = np.expand_dims(all_points, axis=1)  # Shape: (N,1,2)

    return Img, binary_mask, all_points, point_labels, len(inds)

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


def calculate_iou(pred_mask, gt_mask):
    """
    Calculate IoU between prediction and ground truth masks
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0
    return intersection / union

def main_prediction_process(
        sam2_model, image, predicted_mask, ground_truth_mask=None
    ):
    import time
    
    # Record start time
    start_time = time.time()

    i = 1 

    # Calculate IoU for original prediction
    original_iou = None
    if ground_truth_mask is not None:
        original_iou = calculate_iou(predicted_mask > 0, ground_truth_mask > 0)
        print(f"Original Prediction IoU: {original_iou:.4f}")

    predictor2 = SAM2ImagePredictor(sam2_model)

    print("Generating random points for the input......")
    # Generate random points for the input
    _, _, input_points, input_labels, _ = read_batch_speical(image, predicted_mask, True)

    print(f"Generated {input_points.shape[0]} points for the input")
    
    if (len(input_points) <= 1):
        return None, None, None, None

    print(f"输入点坐标维度: {input_points.shape}")
    print(f"输入标签维度: {input_labels.shape}")
    print(f"输入标签维度: {np.ones([input_points.shape[0], 1]).shape}")

    print(f"Sam Model predicting......")  

    # (TODO)
    # image = ground_truth_mask
    # Ensure image is RGB (3 channels)
    # if len(image.shape) == 2:  # If grayscale
    #     image = np.stack([image] * 3, axis=-1)  # Convert to RGB
    # elif len(image.shape) == 3 and image.shape[2] == 1:  # If grayscale with channel dimension
    #     image = np.repeat(image, 3, axis=2)  # Convert to RGB
    
    plt.imshow(image)
    
    print(f"input_labels: {input_labels}")
    print(f"input_points: {input_points}")

    # SAM prediction
    with torch.no_grad():
        predictor2.set_image(image)
        masks, scores, logits = predictor2.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

    print(f"Sam Model predict done!")

    # 處理預測結果
    np_masks_2 = np.array(masks[:, 0])
    np_scores_2 = scores[:, 0]
    print(np_scores_2)
    sorted_masks_2 = np_masks_2[np.argsort(np_scores_2)][::-1]

    occupancy_mask_2 = np.zeros_like(sorted_masks_2[0], dtype=bool)

    for i in range(sorted_masks_2.shape[0]):
        mask_2 = sorted_masks_2[i]
        if (mask_2 * occupancy_mask_2).sum() / mask_2.sum() > OVERLAP_THRESHOLD:
            continue
        mask_bool_2 = mask_2.astype(bool)
        mask_bool_2[occupancy_mask_2] = False
        occupancy_mask_2[mask_bool_2] = True

    seg_map2_cleaned = remove_small_objects(occupancy_mask_2, min_size=MIN_SIZE, connectivity=2)
    selem = disk(SELEM_RADIUS)
    seg_map2_closed = binary_erosion(binary_dilation(seg_map2_cleaned, selem), selem)
    seg_map2_final = np.zeros_like(seg_map2_closed, dtype=np.uint8)

    for i in range(sorted_masks_2.shape[0]):
        mask_2 = sorted_masks_2[i]
        mask_bool_2 = mask_2.astype(bool)
        mask_bool_2 = mask_bool_2 & seg_map2_closed
        seg_map2_final[mask_bool_2] = i + 1

    # Calculate enhanced IoU and processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    enhanced_iou = None
    if ground_truth_mask is not None:
        enhanced_iou = calculate_iou(seg_map2_closed, ground_truth_mask > 0)
        iou_improvement = ((enhanced_iou - original_iou) / original_iou) * 100 if original_iou > 0 else 0
        
        print("\nPerformance Metrics:")
        print(f"Original IoU: {original_iou:.4f}")
        print(f"Enhanced IoU: {enhanced_iou:.4f}")
        print(f"IoU Improvement: {iou_improvement:.2f}%")
        print(f"Processing Time: {processing_time:.2f} seconds")

    # Visualization code remains the same...
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Binarized Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(seg_map2_final, cmap='tab20')
    plt.title('Binarized Mask with Points')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(seg_map2_cleaned, cmap='gray')
    plt.title('After Remove Small Objects')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(binary_dilation(seg_map2_cleaned, selem), cmap='gray')
    plt.title('After Dilation')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(seg_map2_closed, cmap='gray')
    plt.title('After Closing Operation')
    plt.axis('off')     

    plt.tight_layout()
    plt.show()

    cv2.imwrite('final_segmentation.jpg', seg_map2_final.astype(np.uint8) * 255)

    return seg_map2_final, original_iou, enhanced_iou, processing_time

    



