# # 处理特定图像列表
# from functions.model import batch_segmentation_pipeline

# results = batch_segmentation_pipeline(
#     model_path="models/UnetTrain/overlaping_unet_segmentation_try3.24V0A1K_PGD.pth",
#     image_paths="dataset/Lucchi++/Test_In",  # 图像目录
#     mask_paths="dataset/Lucchi++/Test_Out",  # 掩码目录
#     save_dir="test/batch_predictData",
#     patch_size=256,
#     stride=128,
#     value=-30,
#     alpha=1.0
# )

from matplotlib import pyplot as plt
from functions.samPrediction import main_prediction_process_batch
from functions.samPrediction import main_prediction_process
from functions.samPrediction import build_sam2_model

sam2_model = build_sam2_model(
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt", 
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml",
)

sam_result = main_prediction_process_batch(
    sam2_model=sam2_model,
    input_dir="test/batch_predictData",
    save_dir="test/batch_sampredictData"
)