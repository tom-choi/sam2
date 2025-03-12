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
    ("SAM-Hiera-Small", "./checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
]

# 全局模型缓存
model_cache = {
    "unet": None,
    "sam": None
}

# 定义所有可能生成的PNG文件路径
OUTPUT_FILES = [
    "original.png",
    "processed.png",
    "processed_prediction.png",
    "prediction.png",
    "true_mask_binary.png",
    "comparison.png",
    "color_mask.png",
    "overlay.png",
    "prediction_binary.png",
    "with_contours.png",
    "prediction_overlay.png",
    "visualization.png",
    "histograms.png"
]

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

def load_output_images(save_dir):
    """加载保存目录中的所有输出图像"""
    images = {}
    for filename in OUTPUT_FILES:
        path = os.path.join(save_dir, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # 转换颜色空间并调整通道顺序
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                images[filename] = img
    return images

from functools import wraps
import time
import uuid

def timeout_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"处理超时: {str(e)}")
        finally:
            elapsed = time.time() - start_time
            print(f"处理耗时: {elapsed:.2f}s")
        return result
    return wrapper

@timeout_handler
def process_image(input_image, save_dir="test/predictData"):
    """完整的图像处理流程"""
    # 创建唯一会话目录
    session_id = str(uuid.uuid4())[:8]
    save_dir = f"temp_output/{session_id}"
    os.makedirs(save_dir, exist_ok=True)

    # 验证模型状态
    is_valid, msg = validate_models()
    if not is_valid:
        raise gr.Error(msg)
    
    # 转换并保存临时图像
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, input_image)

    # 设置参数
    model = model_cache["unet"]
    image_path = "dataset/Lucchi++/Train_In/mask0000.png"
    mask_path = "dataset/Lucchi++/Train_Out/0.png"
    patch_size = 256
    stride = 64
    v = -30
    alpha = 1.3

    try:
        # 执行预测流程
        pred_mask, _ = save_prediction_results(
            model=model,
            image_path=image_path,
            mask_path=mask_path,
            save_dir=save_dir,
            patch_size=patch_size,
            stride=stride,
            value=v,
            alpha=alpha
        )

        # SAM细化处理
        main_prediction_process(
            sam2_model=model_cache["sam"],
            image=input_image,
            predicted_mask=pred_mask
        )

        # 加载结果（确保返回7个图像）
        output_images = load_output_images(save_dir)
        
        # 验证并确保返回7个有效图像
        required_outputs = [
            output_images.get("original.png", generate_error_image("original")),
            output_images.get("processed.png", generate_error_image("processed")),
            output_images.get("prediction.png", generate_error_image("prediction")),
            output_images.get("comparison.png", generate_error_image("comparison")),
            output_images.get("overlay.png", generate_error_image("overlay")),
            output_images.get("histograms.png", generate_error_image("histogram")),
            output_images.get("visualization.png", generate_error_image("visualization"))
        ]
        
        # 转换为numpy数组并验证形状
        validated_outputs = []
        for img in required_outputs:
            if img is None:
                validated_outputs.append(generate_error_image("unknown"))
            elif not isinstance(img, np.ndarray):
                validated_outputs.append(generate_error_image("invalid type"))
            else:
                validated_outputs.append(img)
                
        return validated_outputs
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        # 返回错误提示图像
        error_img = np.zeros((512,512,3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {str(e)}", (10, 256), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return [error_img] * 7  # 所有位置显示相同错误
    
    # finally:
    #     # 清理临时文件（可选）
    #     try:
    #         shutil.rmtree(save_dir)
    #     except:
    #         pass

def generate_error_image(message):
    """生成统一格式的错误提示图像"""
    error_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(error_img, f"Error: {message[:50]}", (10, 256), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return error_img

def load_output_images(save_dir):
    """增强版图像加载函数"""
    images = {}
    for filename in OUTPUT_FILES:
        path = os.path.join(save_dir, filename)
        try:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("Invalid image file")
                
                # 标准化图像格式
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 确保尺寸兼容
                if img.shape[0] < 256 or img.shape[1] < 256:
                    img = cv2.resize(img, (512, 512))
                
                images[filename] = img
            else:
                images[filename] = None
        except Exception as e:
            print(f"加载错误 {filename}: {str(e)}")
            images[filename] = None
    return images

# 创建Gradio界面
with gr.Blocks(title="医学图像分割系统", css=".col {min-height: 300px}") as demo:
    gr.Markdown("## 医学影像智能分割系统 v2.0")
    
    with gr.Row():
        # 控制面板
        with gr.Column(scale=1):
            gr.Markdown("### 模型配置")
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
            sam_status = gr.Text(label="SAM加载状态")
            
            gr.Markdown("### 影像输入")
            image_input = gr.Image(label="输入CT影像", type="numpy")
            submit_btn = gr.Button("执行分割", variant="primary")

        # 主要结果展示
        with gr.Column(scale=2):
            gr.Markdown("### 核心结果")
            with gr.Tabs():
                with gr.TabItem("基础结果"):
                    original_view = gr.Image(label="原始影像")
                    processed_view = gr.Image(label="预处理结果")
                    prediction_view = gr.Image(label="最终预测")
                
                with gr.TabItem("分析视图"):
                    comparison_view = gr.Image(label="对比视图")
                    overlay_view = gr.Image(label="叠加视图")
                
                with gr.TabItem("质量分析"):
                    histogram_view = gr.Image(label="直方图分析")
                    visualization_view = gr.Image(label="综合可视化")

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
        outputs=[
            original_view,
            processed_view,
            prediction_view,
            comparison_view,
            overlay_view,
            histogram_view,
            visualization_view
        ]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)