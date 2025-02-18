import gradio as gr
import torch
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class LucchiPPDataset(Dataset):
    """
    Lucchi++数据集加载器
    数据集结构：
    dataset/Lucchi++/
        ├── Train_In/    
        ├── Train_Out/   
        ├── Test_In/     
        └── Test_Out/    
    """
    def __init__(self, data_dir, split='test', transform=None):
        """
        参数:
            data_dir (str): Lucchi++数据集的根目录
            split (str): 'train' 或 'test'
            transform: 可选的图像变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 设置图像和掩码目录
        if split == 'train':
            self.image_dir = os.path.join(data_dir, "Train_In")
            self.mask_dir = os.path.join(data_dir, "Train_Out")
        else:
            self.image_dir = os.path.join(data_dir, "Test_In")
            self.mask_dir = os.path.join(data_dir, "Test_Out")
            
        # 获取所有图像文件
        self.image_files = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像路径
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, f"{idx}.png")
        
        # 读取图像和掩码
        image = cv2.imread(image_path)[..., ::-1]  # BGR转RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"无法读取图像或掩码: {image_path}, {mask_path}")
            
        # 调整大小
        r = min(1024 / image.shape[1], 1024 / image.shape[0])
        image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                         interpolation=cv2.INTER_NEAREST)
        
        # 二值化掩码
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 获取点标注
        eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
        labels = measure.label(eroded_mask)
        regions = measure.regionprops(labels)
        
        points = []
        for region in regions:
            y, x = region.coords[np.random.randint(len(region.coords))]
            points.append([x, y])
            
        points = np.array(points)
        
        # 调整维度
        binary_mask = np.expand_dims(binary_mask, axis=0)  # (1, H, W)
        if len(points) > 0:
            points = np.expand_dims(points, axis=1)  # (N, 1, 2)
            
        num_masks = len(regions)
        
        return image, binary_mask, points, num_masks

def load_lucchi_dataset(data_dir="dataset/Lucchi++", split='test'):
    """
    加载Lucchi++数据集
    
    参数:
        data_dir (str): 数据集根目录
        split (str): 'train' 或 'test'
    
    返回:
        LucchiPPDataset对象
    """
    return LucchiPPDataset(data_dir, split)

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

class SAM2Interface:
    def __init__(self):
        # 初始化設置，判斷是否有可用的GPU，如果有則設置為cuda，否則設置為cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.predictor = None
        
    def load_model(self, model_path, config_path):
        """加載SAM2模型"""
        try:
            # 根據配置文件加載SAM2模型
            self.model = build_sam2(config_path)
            # 從模型路徑加載模型參數
            checkpoint = torch.load(model_path, map_location=self.device)
            # 將模型參數加載到模型中
            self.model.load_state_dict(checkpoint, strict=False)
            # 將模型轉移到設備上
            self.model = self.model.to(self.device)
            # 初始化預測器
            self.predictor = SAM2ImagePredictor(self.model)
            return "模型加載成功!"
        except Exception as e:
            return f"模型加載失敗: {str(e)}"

    def process_image(self, input_image, point_x, point_y):
        """處理單張圖像"""
        if self.predictor is None:
            return None, "請先加載模型!"
        
        # 轉換輸入圖像
        if isinstance(input_image, str):
            # 如果輸入的是圖像路徑，則讀取圖像
            image = cv2.imread(input_image)
            # 將圖像從BGR格式轉換為RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 如果輸入的是圖像數據，則直接轉換為numpy數組
            image = np.array(input_image)
        
        # 準備點標註
        input_point = np.array([[point_x, point_y]])
        input_label = np.array([1])
        
        # 預測
        self.predictor.set_image(image)
        masks = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label
        )
        
        # 可視化結果
        mask = masks[0] > 0.5
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [255, 0, 0]  # 紅色標註分割區域
        
        result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        return Image.fromarray(result), "處理成功!"

    def evaluate_dataset(self, dataset_path, split='test'):
        """評估數據集"""
        if self.predictor is None:
            return "請先加載模型!"
        
        # 加載數據集
        test_dataset = load_lucchi_dataset(dataset_path, split)
        # 評估模型
        mean_iou, std_iou = evaluate_model(self.predictor, test_dataset)
        
        return f"評估結果:\nMean IoU: {mean_iou:.3f}\nStd IoU: {std_iou:.3f}"

def create_interface():
    sam2_interface = SAM2Interface()
    
    with gr.Blocks() as demo:
        gr.Markdown("# SAM2 圖像分割演示")
        
        with gr.Tab("模型配置"):
            model_path = gr.Textbox(label="模型檔案路徑", value="sam2_lora_checkpoint_3000.pth")
            config_path = gr.Textbox(label="配置檔案路徑", value="configs/sam2.1/sam2.1_hiera_l.yaml")
            load_btn = gr.Button("加載模型")
            load_output = gr.Textbox(label="加載狀態")
            
            load_btn.click(
                fn=sam2_interface.load_model,
                inputs=[model_path, config_path],
                outputs=load_output
            )
        
        with gr.Tab("圖像分割"):
            with gr.Row():
                input_image = gr.Image(label="輸入圖像")
                output_image = gr.Image(label="分割結果")
            
            with gr.Row():
                point_x = gr.Number(label="點擊X座標")
                point_y = gr.Number(label="點擊Y座標")
            
            process_btn = gr.Button("執行分割")
            process_output = gr.Textbox(label="處理狀態")
            
            process_btn.click(
                fn=sam2_interface.process_image,
                inputs=[input_image, point_x, point_y],
                outputs=[output_image, process_output]
            )
        
        with gr.Tab("數據集評估"):
            dataset_path = gr.Textbox(label="數據集路徑", value="dataset/Lucchi++")
            split_choice = gr.Radio(choices=["train", "test"], label="數據集分割", value="test")
            eval_btn = gr.Button("開始評估")
            eval_output = gr.Textbox(label="評估結果")
            
            eval_btn.click(
                fn=sam2_interface.evaluate_dataset,
                inputs=[dataset_path, split_choice],
                outputs=eval_output
            )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0")