import os
import cv2
import numpy as np
from torch.utils.data import Dataset

from skimage import measure

def evaluate_model(predictor, test_dataset, save_dir=None):
    """
    评估模型在测试数据集上的性能
    
    Args:
        predictor: 加载了预训+练模型的SAM2预测器
        test_dataset: 测试数据集
        save_dir: 可选，保存预测结果的目录
    """
    predictor.model.eval()  # 设置为评估模式
    all_ious = []
    
    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with torch.no_grad():  # 不计算梯度
        for idx, (image, gt_mask, input_point, num_masks) in enumerate(tqdm(test_dataset)):
            if image is None or gt_mask is None or num_masks == 0:
                continue
                
            # 准备输入数据
            input_point = np.array(input_point)
            input_label = np.ones((num_masks, 1))
            
            # 基本的数据检查
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue
            if input_point.size == 0 or input_label.size == 0:
                continue
                
            # 设置图像并获取预测
            predictor.set_image(image)
            pred_masks = predictor.predict(
                point_coords=input_point,
                point_labels=input_label
            )
            
            # 计算IoU
            gt_mask = torch.tensor(gt_mask.astype(np.float32)).cuda()
            pred_mask = torch.tensor(pred_masks > 0.5).cuda().float()
            
            intersection = (gt_mask * pred_mask).sum((1, 2))
            union = gt_mask.sum((1, 2)) + pred_mask.sum((1, 2)) - intersection
            iou = (intersection / (union + 1e-6)).cpu().numpy()
            
            all_ious.extend(iou)
            
            # 可选：保存预测结果
            if save_dir:
                save_path = os.path.join(save_dir, f'pred_{idx}.png')
                # 这里添加保存预测mask的代码
                
    # 计算统计信息
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious)
    
    return mean_iou, std_iou