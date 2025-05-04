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
        # plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # 現在形狀是 (1024, 1024, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    # 返回圖像、二值化遮罩、點和遮罩數量
    return Img, binary_mask, points, len(inds)

# Visualize the data
# Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)

# 可修改的參數
PATCH_SIZE = 100  # Patch大小,影響分割精度和速度。更大的patch可能提高精度,但會增加計算時間。
MIN_SIZE = 5  # 移除小於此尺寸的白色區域。更大的值會移除更多的小區域。
SELEM_RADIUS = 4  # 形態學操作中圓形結構元素的半徑。更大的半徑會導致更多的平滑。
OVERLAP_THRESHOLD = 0.00  # 用於判斷是否保留重疊區域的閾值。更高的值會保留更多的重疊區域。

SCORE_THRESHOLD = 0.05  # 預測分數閾值

MIN_POINT_DISTANCE = 20  # 點之間的最小距離閾值

# 新增参数配置
N_NEGATIVE_POINTS = 45  # 负点采样数量
MIN_NEGATIVE_DISTANCE = 5  # 负点与正点的最小距离

def read_batch_speical(Img, ann_map, visualize_data=False, target_size=None):
    """
    Process image and annotation mask with flexible resizing options.

    Args:
        Img: Input image
        ann_map: Annotation mask
        visualize_ visualize the data
        target_size: Optional target size (height, width) for resizing, or None to use original size
    """
    # If target_size is provided, use it for resizing
    if target_size is not None:
        target_height, target_width = target_size
        # Calculate scale factors to maintain aspect ratio
        r_height = target_height / Img.shape[0]
        r_width = target_width / Img.shape[1]
        # Use smaller scaling factor to ensure the image fits within target dimensions
        r = min(r_height, r_width)
    else:
        # Keep original size (no scaling)
        r = 1.0

    # Apply resize only if scaling is needed (r != 1.0)
    if r != 1.0:
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
                           interpolation=cv2.INTER_NEAREST)

    # Rest of the function remains the same...
    # Initialize binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)

    # Get binary mask and merge into a single mask
    inds = np.unique(ann_map)[1:]  # Skip background (index 0)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)
        binary_mask = np.maximum(binary_mask, mask)

    # Post-processing
    # 1. Remove small objects
    cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=MIN_SIZE, connectivity=2)

    # 2. Create disk-shaped structuring element and perform closing operation
    selem = disk(SELEM_RADIUS)
    processed_mask = binary_erosion(binary_dilation(cleaned_mask, selem), selem)

    # 3. Erode the processed mask to avoid boundary points
    eroded_mask = cv2.erode(processed_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

    # Generate positive points (foreground points)
    labels = measure.label(eroded_mask)
    regions = measure.regionprops(labels)

    # Optimized positive point sampling logic
    selected_positive_points = []
    for region in regions:
        # Dynamic distance threshold (based on region area)
        dynamic_distance = max(15, int(0.08 * np.sqrt(region.area)))

        # Get region representative point
        y, x = region.centroid
        x, y = int(round(x)), int(round(y))
        if eroded_mask[y, x] == 0:
            coords = region.coords
            distances = np.sqrt((coords[:, 0] - y)**2 + (coords[:, 1] - x)**2)
            nearest_idx = np.argmin(distances)
            y, x = coords[nearest_idx]
            x, y = int(x), int(y)

        current_point = np.array([x, y])

        # Distance check
        too_close = any(np.linalg.norm(current_point - p) < dynamic_distance for p in selected_positive_points)
        if not too_close:
            selected_positive_points.append(current_point)

    positive_points = np.array(selected_positive_points)

    # Modified negative point sampling logic
    distance_map = np.zeros_like(eroded_mask, dtype=np.float32)
    for point in selected_positive_points:
        y, x = point[1], point[0]
        y_coords, x_coords = np.ogrid[:eroded_mask.shape[0], :eroded_mask.shape[1]]
        distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        distance_map = np.maximum(distance_map, distances)

    valid_background_mask = (distance_map >= MIN_NEGATIVE_DISTANCE) & (eroded_mask == 0)
    valid_coords = np.argwhere(valid_background_mask)

    if len(valid_coords) >= N_NEGATIVE_POINTS:
        selected_indices = np.random.choice(len(valid_coords), N_NEGATIVE_POINTS, replace=False)
        negative_points = valid_coords[selected_indices]
        negative_points = np.fliplr(negative_points)
    else:
        negative_points = np.fliplr(valid_coords)
        remaining = N_NEGATIVE_POINTS - len(negative_points)
        if remaining > 0:
            background_mask = (eroded_mask == 0)
            remaining_coords = np.argwhere(background_mask)
            remaining_indices = np.random.choice(len(remaining_coords), remaining, replace=False)
            additional_points = np.fliplr(remaining_coords[remaining_indices])
            negative_points = np.vstack([negative_points, additional_points])

    negative_points = np.array(negative_points)

    # Combine positive and negative points and create labels
    all_points = np.vstack([positive_points, negative_points]) if len(positive_points) > 0 and len(negative_points) > 0 else (
        positive_points if len(positive_points) > 0 else negative_points
    )
    point_labels = np.array([1]*len(positive_points) + [0]*len(negative_points)).reshape(-1, 1)

    # Visualization code (unchanged)
    if visualize_data:
        # ... visualization code remains the same ...
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
        # plt.show()

        plt.figure(figsize=(10, 5))
        plt.imshow(eroded_mask, cmap='gray')

        # Plot positive points (green) and negative points (red)
        if len(positive_points) > 0:
            plt.scatter(positive_points[:,0], positive_points[:,1], c='lime', s=50,
                       edgecolors='white', linewidths=1, label='Positive Points')
        if len(negative_points) > 0:
            plt.scatter(negative_points[:,0], negative_points[:,1], c='red', s=30,
                       marker='x', linewidths=1, label='Negative Points')

        plt.legend()
        plt.title('Positive/Negative Points Visualization')
        plt.axis('off')
        # plt.show()

    # Use processed mask as return value
    binary_mask = processed_mask.astype(np.uint8)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    binary_mask = binary_mask.transpose((2, 0, 1))

    # Adjust dimensions to match SAM input format
    all_points = np.expand_dims(all_points, axis=1) if len(all_points) > 0 else np.zeros((0, 1, 2))

    return Img, binary_mask, all_points, point_labels, len(inds)

def read_image(image_path, mask_path, target_size = None):  # read and resize image and mask
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    if target_size is not None:
        target_height, target_width = target_size
        # Calculate scale factors to maintain aspect ratio
        r_height = target_height / img.shape[0]
        r_width = target_width / img.shape[1]
        # Use smaller scaling factor to ensure the image fits within target dimensions
        r = min(r_height, r_width)
    else:
        # Keep original size (no scaling)
        r = 1.0

    # Apply resize only if scaling is needed (r != 1.0)
    if r != 1.0:
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
    # Ensure both masks have the same dimensions
    if pred_mask.shape != gt_mask.shape:
        # Resize one of the masks to match the other
        # Option 1: Resize ground truth to match prediction
        gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8),
                                    (pred_mask.shape[1], pred_mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        gt_mask = gt_mask_resized > 0

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0
    return intersection / union

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    # Convert predictions to binary
    pred_mask = (pred_mask > threshold).float()

    # Calculate intersection and union
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection

    # Calculate IoU
    iou = (intersection + 1e-7) / (union + 1e-7)

    # Calculate Dice coefficient
    dice = (2. * intersection + 1e-7) / (pred_mask.sum() + true_mask.sum() + 1e-7)

    return iou.item(), dice.item()


def main_prediction_process(
        sam2_model, image, predicted_mask, ground_truth_mask=None, save_dir = "test/predictData", visualize_data = True
    ):
    import time

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Record start time
    start_time = time.time()

    # Ensure predicted_mask is properly binarized - fix for transparent areas
    binary_predicted_mask = np.where(predicted_mask > 0, 1, 0).astype(np.uint8)

    # Calculate IoU for original prediction
    original_iou = None
    dice = None
    if ground_truth_mask is not None:
        # pred_tensor = torch.sigmoid(torch.from_numpy(binary_predicted_mask)) > 0.5
        # pred_masks = (torch.sigmoid(binary_predicted_mask) > 0.5).float()

        true_mask_float = (ground_truth_mask == 255).astype(np.float32)
        post_processed_float = (binary_predicted_mask > 0.5).astype(np.float32)

        original_iou, dice = calculate_metrics(
            torch.from_numpy(post_processed_float),
            torch.from_numpy(true_mask_float)
        )
        # original_iou, dice = calculate_metrics(pred_tensor.float() > 0, true_tensor)
        print(f"Original Prediction IoU: {original_iou:.4f}, Dice: {dice:.4f}")

    predictor2 = SAM2ImagePredictor(sam2_model)

    print("Generating random points for the input......")

    # Generate random points for the input using the binary mask
    _, _, input_points, input_labels, _ = read_batch_speical(image, binary_predicted_mask, True)

    print(f"Generated {input_points.shape[0]} points for the input")

    if (len(input_points) <= 1):
        return None, None, None, None

    print(f"输入点坐标维度: {input_points.shape}")
    print(f"输入标签维度: {input_labels.shape}")

    print(f"Sam Model predicting......")

    # SAM prediction
    with torch.no_grad():
        image = np.ascontiguousarray(image)
        predictor2.set_image(image)
        masks, scores, logits = predictor2.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

    print(f"Sam Model predict done!")

    # Process prediction results
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
    enhanced_dice = None
    if ground_truth_mask is not None:
        true_mask_float = (ground_truth_mask == 255).astype(np.float32)
        pred_masks_float = (seg_map2_closed > 0.5).astype(np.float32)

        enhanced_iou, enhanced_dice = calculate_metrics(
            torch.from_numpy(pred_masks_float),
            torch.from_numpy(true_mask_float)
        )

        print(original_iou)

        iou_improvement = ((enhanced_iou - original_iou) / original_iou) * 100 if original_iou > 0 else 0

        print("\nPerformance Metrics:")
        print(f"Original IoU: {original_iou:.4f}, Dice: {dice:.4f}")
        print(f"Enhanced IoU: {enhanced_iou:.4f}, Dice: {enhanced_dice:.4f}")
        print(f"IoU Improvement: {iou_improvement:.2f}%")
        print(f"Processing Time: {processing_time:.2f} seconds")

        # Create comparison visualization (similar to visualize_results)
        # Create color mask to show TP, FP, FN regions
        color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Convert masks to binary
        gt_binary = ground_truth_mask > 0
        pred_binary = seg_map2_closed > 0

        # Adjust sizes if needed
        if gt_binary.shape != pred_binary.shape:
            gt_binary = cv2.resize(
                gt_binary.astype(np.uint8),
                (pred_binary.shape[1], pred_binary.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ) > 0

        # Create valid area mask (non-transparent)
        valid_area = np.ones_like(gt_binary, dtype=bool)
        if len(image.shape) > 2 and image.shape[2] == 4:  # RGBA image
            valid_area = (image[:,:,3] > 0)

        # True Positive (green): correctly detected regions
        color_mask[np.logical_and(gt_binary, pred_binary)] = [0, 255, 0]

        # False Positive (red): incorrectly detected regions
        color_mask[np.logical_and(np.logical_not(gt_binary), pred_binary)] = [255, 0, 0]

        # False Negative (blue): missed regions
        color_mask[np.logical_and(np.logical_and(gt_binary, np.logical_not(pred_binary)), valid_area)] = [0, 0, 255]

        # Create overlay on original image
        comparison_overlay = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

        # Save the comparison visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(comparison_overlay)
        plt.title(f'Ground Truth vs Prediction\nIoU: {enhanced_iou:.4f}, Dice: {enhanced_dice:.4f}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sam_gt_pred_comparison.png'), dpi=300)
        plt.close()

    # Save the main comparison visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Initial Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(seg_map2_closed, cmap='gray')
    plt.title('Enhanced Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sam_comparison.png'), dpi=300)
    plt.close()

    # Save the processing steps visualization
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
    plt.savefig(os.path.join(save_dir, 'sam_processing_steps.png'), dpi=300)
    plt.close()

    # Save instance segmentation visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(seg_map2_final, cmap='tab20')
    plt.title('Instance Segmentation')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sam_segmentation_final.png'), dpi=300)
    plt.close()

    # Also save the final segmentation as raw array
    cv2.imwrite(os.path.join(save_dir, 'sam_final_segmentation.jpg'), seg_map2_closed.astype(np.uint8) * 255)

    # Create evaluation visualization if ground truth is available
    if ground_truth_mask is not None:
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(ground_truth_mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(seg_map2_closed, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(comparison_overlay)
        plt.title(f'ComparisonIoU: {enhanced_iou:.4f}, Dice: {enhanced_dice:.4f}')
        plt.axis('off')
        

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sam_evaluation.png'), dpi=300)
        # Save the comparison visualization without title and borders
        plt.figure(figsize=(8, 8))
        plt.imshow(comparison_overlay)
        plt.axis('off')  # Turn off the axis
        plt.tight_layout(pad=0)  # Adjust layout to minimize padding
        plt.savefig(os.path.join(save_dir, 'DEBUG_comparison_overlay.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    metrics = {
        "Original_IoU": original_iou,
        "Enhanced_IoU": enhanced_iou,
        "Original_Dice": dice,
        "Enhanced_Dice": enhanced_dice,
        "Processing_Time": processing_time
    }

    return seg_map2_closed.astype(np.uint8), metrics, comparison_overlay if ground_truth_mask is not None else None, processing_time

def main_prediction_process_batch(
        sam2_model, input_dir="batch_predictData", save_dir=None,
        device="cuda"):
    """
    Batch processing pipeline for SAM2 segmentation enhancement
    Args:
        sam2_model: The loaded SAM2 model
        input_dir: Directory containing subfolders with images and masks
        save_dir: Directory to save results (defaults to input_dir if None)
        device: Device to run the model on ("cuda" or "cpu")
    Returns:
        Dictionary containing results and statistics
    """
    import os
    import time
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # 设置matplotlib为非交互模式，防止图像弹出
    plt.ioff()  # 关闭交互模式

    # If save_dir is not specified, use input_dir
    if save_dir is None:
        save_dir = input_dir

    # Get all subdirectories in the input directory
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    subdirs.sort()  # Sort to process in order

    if not subdirs:
        print(f"Error: No subdirectories found in {input_dir}")
        return None

    print(f"\n开始处理 {len(subdirs)} 个样本...")

    # 准备收集结果
    results = []
    all_metrics = []
    best_iou = {'value': -1, 'index': -1, 'folder': '', 'metrics': None}
    worst_iou = {'value': float('inf'), 'index': -1, 'folder': '', 'metrics': None}
    total_processing_time = 0

    for idx, subdir in enumerate(subdirs):
        folder_path = os.path.join(input_dir, subdir)
        sample_name = subdir

        # Define file paths
        original_img_path = os.path.join(folder_path, f"{sample_name}_original.png")
        predicted_mask_path = os.path.join(folder_path, f"{sample_name}_prediction_final.png")
        true_mask_path = os.path.join(folder_path, f"{sample_name}_true_mask.png")

        # Check if required files exist
        if not os.path.exists(original_img_path):
            print(f"Warning: Original image not found for {sample_name}, skipping.")
            continue

        if not os.path.exists(predicted_mask_path):
            print(f"Warning: Predicted mask not found for {sample_name}, skipping.")
            continue

        # Create output directory
        sample_save_dir = os.path.join(save_dir, f"{sample_name}_sam_enhanced")
        os.makedirs(sample_save_dir, exist_ok=True)

        print(f"\n处理进度 {idx+1}/{len(subdirs)}: {sample_name}")

        # Read images
        image = cv2.imread(original_img_path)[..., ::-1]  # Convert BGR to RGB
        predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(true_mask_path) else None

        # Process with SAM
        enhanced_mask, metrics, comparison, processing_time = main_prediction_process(
            sam2_model=sam2_model,
            image=image,
            predicted_mask=predicted_mask,
            ground_truth_mask=ground_truth,
            save_dir=sample_save_dir
        )

        total_processing_time += processing_time

        # Store results
        results.append({
            'folder': folder_path,
            'sample_name': sample_name,
            'enhanced_mask': enhanced_mask,
            'comparison': comparison,
            'processing_time': processing_time
        })

        # Process metrics
        if metrics is not None:
            all_metrics.append(metrics)

            # Update best/worst IoU
            enhanced_iou = metrics.get('Enhanced_IoU')
            if enhanced_iou is not None and enhanced_iou > best_iou['value']:
                best_iou.update({
                    'value': enhanced_iou,
                    'index': idx,
                    'folder': folder_path,
                    'sample_name': sample_name,
                    'metrics': metrics
                })

            if enhanced_iou is not None and enhanced_iou < worst_iou['value']:
                worst_iou.update({
                    'value': enhanced_iou,
                    'index': idx,
                    'folder': folder_path,
                    'sample_name': sample_name,
                    'metrics': metrics
                })

    # If metrics were collected, calculate statistics
    if all_metrics:
        # Create statistics directory
        stats_dir = os.path.join(save_dir, "sam_statistics")
        os.makedirs(stats_dir, exist_ok=True)

        # Calculate averages and standard deviations
        enhanced_ious = [m.get('Enhanced_IoU', 0) for m in all_metrics if m.get('Enhanced_IoU') is not None]
        if enhanced_ious:
            avg_enhanced_iou = np.mean(enhanced_ious)
            std_enhanced_iou = np.std(enhanced_ious)
        else:
            avg_enhanced_iou = std_enhanced_iou = 0

        enhanced_dices = [m.get('Enhanced_Dice', 0) for m in all_metrics if m.get('Enhanced_Dice') is not None]
        if enhanced_dices:
            avg_enhanced_dice = np.mean(enhanced_dices)
            std_enhanced_dice = np.std(enhanced_dices)
        else:
            avg_enhanced_dice = std_enhanced_dice = 0

        # Calculate improvement percentages
        improvements = []
        for m in all_metrics:
            orig = m.get('Original_IoU')
            enh = m.get('Enhanced_IoU')
            if orig is not None and enh is not None and orig > 0:
                improvements.append(((enh - orig) / orig) * 100)

        avg_improvement = np.mean(improvements) if improvements else 0
        avg_processing_time = total_processing_time / len(results)

        # Find sample closest to mean
        closest_to_mean = {'diff': float('inf'), 'index': -1, 'folder': '', 'sample_name': '', 'metrics': None}

        for idx, metrics in enumerate(all_metrics):
            enhanced_iou = metrics.get('Enhanced_IoU')
            enhanced_dice = metrics.get('Enhanced_Dice')

            if enhanced_iou is None or enhanced_dice is None:
                continue

            current_diff = abs(enhanced_iou - avg_enhanced_iou) + abs(enhanced_dice - avg_enhanced_dice)
            if current_diff < closest_to_mean['diff']:
                closest_to_mean.update({
                    'diff': current_diff,
                    'index': idx,
                    'folder': results[idx]['folder'],
                    'sample_name': results[idx]['sample_name'],
                    'metrics': metrics
                })

        # Save statistics to text file
        stats_file = os.path.join(stats_dir, "batch_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"SAM批量处理统计 ({len(all_metrics)} 张图像):\n")
            f.write(f"平均增强 IoU: {avg_enhanced_iou:.4f} ± {std_enhanced_iou:.4f}\n")
            f.write(f"平均增强 Dice: {avg_enhanced_dice:.4f} ± {std_enhanced_dice:.4f}\n")
            f.write(f"平均IoU改进: {avg_improvement:.2f}%\n")
            f.write(f"平均处理时间: {avg_processing_time:.2f} 秒/样本\n")

            if best_iou['index'] != -1:
                f.write("\n最佳 IoU:\n")
                f.write(f"  样本: {best_iou['sample_name']}\n")
                f.write(f"  IoU: {best_iou['value']:.4f}, Dice: {best_iou['metrics'].get('Enhanced_Dice', 0):.4f}\n")

            if worst_iou['index'] != -1:
                f.write("\n最差 IoU:\n")
                f.write(f"  样本: {worst_iou['sample_name']}\n")
                f.write(f"  IoU: {worst_iou['value']:.4f}, Dice: {worst_iou['metrics'].get('Enhanced_Dice', 0):.4f}\n")

            if closest_to_mean['index'] != -1:
                f.write("\n最接近平均值:\n")
                f.write(f"  样本: {closest_to_mean['sample_name']}\n")
                f.write(f"  IoU: {closest_to_mean['metrics'].get('Enhanced_IoU', 0):.4f}, Dice: {closest_to_mean['metrics'].get('Enhanced_Dice', 0):.4f}\n")

            # List all samples with metrics sorted by IoU
            f.write("\n所有样本的IoU值 (降序排列):\n")
            sample_ious = [(idx, m.get('Enhanced_IoU', 0)) for idx, m in enumerate(all_metrics)]
            sorted_indices = [idx for idx, _ in sorted(sample_ious, key=lambda x: x[1], reverse=True)]

            for i, idx in enumerate(sorted_indices):
                metrics = all_metrics[idx]
                sample_name = results[idx]['sample_name']
                enhanced_iou = metrics.get('Enhanced_IoU', 0)
                enhanced_dice = metrics.get('Enhanced_Dice', 0)
                improvement = ((enhanced_iou - metrics.get('Original_IoU', 0)) / metrics.get('Original_IoU', 1)) * 100 if metrics.get('Original_IoU', 0) > 0 else 0

                f.write(f"{i+1}. {sample_name}: IoU={enhanced_iou:.4f}, Dice={enhanced_dice:.4f}, 改进={improvement:.2f}%\n")

        # Create visualizations for representative cases
        # Best IoU
        if best_iou['index'] != -1 and results[best_iou['index']]['comparison'] is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(results[best_iou['index']]['comparison'])
            plt.title(f"最佳IoU样本: {best_iou['sample_name']}\nIoU: {best_iou['value']:.4f}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, "best_iou_comparison.png"), dpi=300)
            plt.close()

        # Worst IoU
        if worst_iou['index'] != -1 and results[worst_iou['index']]['comparison'] is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(results[worst_iou['index']]['comparison'])
            plt.title(f"最差IoU样本: {worst_iou['sample_name']}\nIoU: {worst_iou['value']:.4f}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, "worst_iou_comparison.png"), dpi=300)
            plt.close()

        # Average IoU
        if closest_to_mean['index'] != -1 and results[closest_to_mean['index']]['comparison'] is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(results[closest_to_mean['index']]['comparison'])
            plt.title(f"平均IoU样本: {closest_to_mean['sample_name']}\nIoU: {closest_to_mean['metrics'].get('Enhanced_IoU', 0):.4f}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, "average_iou_comparison.png"), dpi=300)
            plt.close()

        # Create IoU distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(enhanced_ious, bins=10, color='steelblue', edgecolor='black')
        plt.axvline(avg_enhanced_iou, color='red', linestyle='dashed', linewidth=1, label=f'平均值: {avg_enhanced_iou:.4f}')
        plt.xlabel('增强IoU值')
        plt.ylabel('样本数量')
        plt.title('IoU分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, "iou_distribution.png"), dpi=300)
        plt.close()

        # Create original vs enhanced IoU scatter plot
        original_ious = [m.get('Original_IoU', 0) for m in all_metrics if m.get('Original_IoU') is not None]
        enhanced_ious = [m.get('Enhanced_IoU', 0) for m in all_metrics if m.get('Enhanced_IoU') is not None]

        if len(original_ious) == len(enhanced_ious) and len(original_ious) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(original_ious, enhanced_ious, alpha=0.7)

            # Add diagonal line (y=x)
            min_val = min(min(original_ious), min(enhanced_ious))
            max_val = max(max(original_ious), max(enhanced_ious))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='无改进线 (y=x)')

            plt.xlabel('原始IoU')
            plt.ylabel('增强IoU')
            plt.title('原始 vs 增强IoU比较')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, "iou_improvement_scatter.png"), dpi=300)
            plt.close()

        # Print statistics
        print("\n----- SAM批量处理统计信息 -----")
        print(f"平均增强 IoU: {avg_enhanced_iou:.4f} ± {std_enhanced_iou:.4f}")
        print(f"平均增强 Dice: {avg_enhanced_dice:.4f} ± {std_enhanced_dice:.4f}")
        print(f"平均IoU改进: {avg_improvement:.2f}%")
        print(f"平均处理时间: {avg_processing_time:.2f} 秒/样本")
        print(f"最佳 IoU: {best_iou['value']:.4f} (样本: {best_iou['sample_name']})")
        print(f"最差 IoU: {worst_iou['value']:.4f} (样本: {worst_iou['sample_name']})")
        print(f"统计信息已保存至: {stats_dir}")

        # Return results and statistics
        return {
            'results': results,
            'statistics': {
                'average_metrics': {
                    'Enhanced_IoU': avg_enhanced_iou,
                    'Enhanced_Dice': avg_enhanced_dice,
                    'Avg_Improvement': avg_improvement,
                    'Avg_Processing_Time': avg_processing_time
                },
                'std_metrics': {
                    'Enhanced_IoU': std_enhanced_iou,
                    'Enhanced_Dice': std_enhanced_dice
                },
                'best_iou': best_iou,
                'worst_iou': worst_iou,
                'closest_to_mean': closest_to_mean,
                'all_metrics': all_metrics,
                'stats_dir': stats_dir
            }
        }

    print("\n处理完成，但未收集到评估指标。可能是因为没有提供真实掩码。")
    return {'results': results}
