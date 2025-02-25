import gradio as gr
import torch
import cv2
import numpy as np
from functions.model import load_model, predict
from functions.samPrediction import main_prediction_process

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 1
model_path = "simple_unet_segmentation.pth"
model = load_model(model_path, num_classes, device)

def process_image(input_image):
    # 转换图像格式
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # 临时保存上传的图片
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, input_image)
    
    try:
        # 第一步：使用UNet生成预测mask
        predicted_mask = predict(model, temp_path, device)
        
        # 第二步：使用SAM进行细化处理
        main_prediction_process(
            image=input_image,
            predicted_mask=predicted_mask,
            sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        
        # 读取处理结果
        results = {
            'original': cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            'binary_mask': (predicted_mask * 255).astype(np.uint8),
            'final_result': cv2.imread("final_segmentation.jpg")  # SAM处理结果需要保存后读取
        }

        return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),(predicted_mask * 255).astype(np.uint8),cv2.imread("final_segmentation.jpg")
    except Exception as e:
        print(f"处理出错: {str(e)}")
        return None

# 创建Gradio界面组件
with gr.Blocks() as demo:
    gr.Markdown("# 医学图像分割系统")
    
    with gr.Row():
        # 左侧输入列
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传CT图像", type="numpy", height=300)
            submit_btn = gr.Button("开始分析")
        
        # 右侧输出列
        with gr.Column(scale=1):
            # 设置统一高度，使用min_height确保最小显示高度
            original_output = gr.Image(label="原始图像", height=200)
            mask_output = gr.Image(label="初部分割结果", height=200)
            final_output = gr.Image(label="精修分割结果", height=200)

    # 通过CSS强制等高布局
    css = """
    .col { min-height: 600px !important; }
    .col .image-preview { height: 400px !important; }
    """
    demo.css = css

    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[original_output, mask_output, final_output]
    )

    gr.Examples(
        examples=[["mask0000.png"], ["dataset\Lucchi++\Test_In\mask0014.png"], ["dataset\Lucchi++\Test_In\mask0016.png"]],
        inputs=input_image,
        outputs=[original_output, mask_output, final_output],
        fn=process_image,
        cache_examples=True
    )

    # examples=[["mask0000.png"], ["mask0000.png"], ["mask0000.png"]]

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)