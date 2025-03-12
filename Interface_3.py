import gradio as gr
import torch
import cv2
import numpy as np
import time
import os
import glob
from functions.model import load_model, predict, save_prediction_results
from functions.samPrediction import main_prediction_process

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 1

# 预设SAM模型配置
SAM_CONFIGS = [
    ("SAM-Hiera-Large", "./checkpoints/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    ("SAM-Hiera-Medium","./checkpoints/sam2.1_hiera_medium.pt","configs/sam2.1/sam2.1_hiera_m.yaml"),
    # 可在此添加更多SAM模型配置
]

# 全局模型缓存
model_cache = {
    "unet": None,
    "sam": None
}

def get_unet_models():
    """获取UNet模型列表"""
    model_dir = "models/UnetTrain"
    if not os.path.exists(model_dir):
        return []
    return glob.glob(os.path.join(model_dir, "*.pth"))

def load_unet_model(model_path):
    """加载UNet模型并计时"""
    start_time = time.time()
    try:
        model = load_model(model_path, device, model_type="UNET")
        load_time = time.time() - start_time
        model_cache["unet"] = model
        print(f"UNet模型加载成功: {model_path} ({load_time:.2f}s)")
        return f"加载成功 ({load_time:.2f}s)"
    except Exception as e:
        model_cache["unet"] = None
        print(f"加载失败: {str(e)}")
        return f"加载失败: {str(e)}"

from functions.samPrediction import build_sam2_model

def load_sam_model(sam_name):
    """加载SAM模型配置"""
    for config in SAM_CONFIGS:
        if config[0] == sam_name:
            model_cache["sam"] = build_sam2_model(config[1], config[2])
            print(f'SAM模型配置加载成功: {config[0]}')
            return "配置加载成功"
        
    # 如果找不到指定SAM模型，则清空缓存
    model_cache["sam"] = None
    print(f'找不到指定SAM模型配置: {sam_name}')
    return "找不到指定SAM模型"

def validate_models():
    """验证模型是否已加载"""
    if not model_cache["unet"]:
        return False, "UNet模型未加载"
    if not model_cache["sam"]:
        return False, "SAM模型未加载"
    return True, "模型加载完成"

def process_image(input_image):
    """图像处理流程"""
    # 验证模型状态
    is_valid, msg = validate_models()
    if not is_valid:
        raise gr.Error(msg)
    
    # 转换并保存临时图像
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, input_image)

    # 设置参数
    # model_path = "models/UnetTrain/overlaping_unet_segmentation_3.9_V0A1K.pth"  # 模型路径
    model_path = model_cache["unet"]
    image_path = "dataset/Lucchi++/Train_In/mask0000.png"  # 测试图片路径
    mask_path = "dataset/Lucchi++/Train_Out/0.png"  # 真实掩码路径（如果有）
    save_dir = "test/predictData"  # 结果保存目录
    patch_size = 256  # patch大小，建议与训练时一致
    stride = 64  # 步长，建议与训练时一致
    v = -30  # 阈值，建议与训练时一致
    alpha = 1.3  # 阈值，建议与训练时一致

    
    try:
        # UNet预测
        # 执行预测并保存结果
        pred_mask, metrics = save_prediction_results(
            model_path=model_path,
            image_path=image_path,
            mask_path=mask_path,
            save_dir=save_dir,
            patch_size=patch_size,
            stride=stride,  # 明确指定步长
            value=v,
            alpha=alpha
        )

        # 保存UNet预测结果
        cv2.imwrite("unet_prediction.jpg", pred_mask * 255)

        try:
            # SAM细化
            main_prediction_process(
                sam2_model=model_cache["sam"],
                image=input_image,
                predicted_mask=pred_mask
            )

        except:
            raise gr.Error("SAM细化失败")
        
        # 读取并转换结果
        return [
            cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            (pred_mask * 255).astype(np.uint8),
            cv2.cvtColor(cv2.imread("final_segmentation.jpg"), cv2.COLOR_BGR2RGB)
        ]
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")
    
    

# 创建Gradio界面
with gr.Blocks(title="医学图像分割系统") as demo:
    gr.Markdown("## 医学影像智能分割系统")
    
    with gr.Row():
        # 控制面板
        with gr.Column(scale=1):
            # 模型选择
            unet_selector = gr.Dropdown(
                label="选择UNet模型",
                choices=get_unet_models(),
                interactive=True
            )
            unet_status = gr.Text(label="UNet加载状态")
            
            sam_selector = gr.Dropdown(
                label="选择SAM模型",
                choices=[x[0] for x in SAM_CONFIGS]
            )
            sam_status = gr.Text(
                label="SAM加载状态"
            )
            
            # 图像输入
            image_input = gr.Image(label="输入CT影像")
            submit_btn = gr.Button("执行分割")
        
        # 结果展示
        with gr.Column(scale=2):
            original_view = gr.Image(label="原始影像", interactive=False)
            mask_view = gr.Image(label="初部分割结果", interactive=False)
            final_view = gr.Image(label="精修分割结果", interactive=False)

    # 事件处理
    unet_selector.change(
        load_unet_model,
        inputs=unet_selector,
        outputs=unet_status
    )
    
    sam_selector.change(
        load_sam_model,
        inputs=sam_selector,
        outputs=sam_status
    )
    
    submit_btn.click(
        process_image,
        inputs=image_input,
        outputs=[original_view, mask_view, final_view]
    )

    # 通过CSS强制等高布局
    css = """
    .col { min-height: 200px !important; }
    .col .image-preview { height: 100px !important; }
    """
    demo.css = css

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)