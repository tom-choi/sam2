import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import gradio as gr
import os
from PIL import Image
import io
import glob

# 自定义的函数模块
from functions.data import prepare_dataset
from functions.model import UNet, reconstruct_from_patches, train_model, predict, segmentation_pipeline
from functions.samPrediction import read_image, main_prediction_process, build_sam2_model
from functions.imagePreprocessing import ImagePreprocessor

def get_output_files(image_path, save_dir="test/predictData"):
    """Get prediction output files for a given image"""
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]

    # Find all files in the save_dir that start with the image name
    pattern = os.path.join(save_dir, f"{name_without_ext}_*.png")
    output_files = glob.glob(pattern)

    # Add labels based on file names
    labeled_files = []
    for file in output_files:
        file_basename = os.path.basename(file)
        # Create a human-readable label from filename
        label = file_basename.replace(f"{name_without_ext}_", "").replace(".png", "").replace("_", " ").title()
        labeled_files.append((file, label))

    pattern = os.path.join(save_dir, f"sam_*.png")
    output_files = glob.glob(pattern)

    # Add labels based on file names
    for file in output_files:
        file_basename = os.path.basename(file)
        # Create a human-readable label from filename
        label = file_basename.replace(f"{name_without_ext}_", "").replace(".png", "").replace("_", " ").title()
        labeled_files.append((file, label))

    # Also look for metrics.txt file
    metrics_file = os.path.join(save_dir, f"{name_without_ext}_metrics.txt")
    metrics_text = ""
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_text = f.read()

    return labeled_files, metrics_text

def process_images(unet_model_path, sam_model_path, sam_config_path, image_path, mask_path=None, save_dir="test/predictData"):
    try:
        # Step 1: UNet segmentation
        predicted_mask, metrics = segmentation_pipeline(
            unet_model_path,
            image_path,
            mask_path,
            patch_size=256,
            stride=128
        )

        # Read the original image and mask
        image, org_mask = read_image(image_path=image_path, mask_path=mask_path)

        # Step 2: Build SAM model
        sam2_model = build_sam2_model(
            sam2_checkpoint=sam_model_path,
            model_cfg=sam_config_path
        )

        # Step 3: Process with SAM model
        # This function should save results to save_dir
        main_prediction_process(
            sam2_model=sam2_model,
            image=image,
            predicted_mask=predicted_mask,
            ground_truth_mask=org_mask,
            save_dir = save_dir
        )

        # Get the output files with labels
        output_files, metrics_text = get_output_files(image_path, save_dir)

        # If no metrics text from file, format the metrics dictionary
        if not metrics_text and metrics:
            metrics_text = "Performance Metrics:"
            for key, value in metrics.items():
                metrics_text += f"{key}: {value:.4f}"

        return output_files, metrics_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [], f"Error: {str(e)}"

def list_models(directory):
    """List all model files in the specified directory"""
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]

def list_sam_configs(directory):
    """List all SAM config files in the specified directory"""
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.yaml')]

def list_images(directory):
    """List all image files in the specified directory"""
    if not os.path.exists(directory):
        return []
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            # only join 
            if file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                result.append(os.path.join(root, file))
    return result

with gr.Blocks(title="Mitochondria Segmentation") as demo:
    gr.Markdown("# Mitochondria Segmentation with UNet and SAM")

    with gr.Row():
        with gr.Column(scale=1):
            # Model selection
            unet_models = list_models("models/UnetTrain")
            unet_model = gr.Dropdown(
                label="UNet Model",
                choices=unet_models,
                value=unet_models[0] if unet_models else None
            )

            # For SAM model, add it to the choices list explicitly
            sam_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            sam_model = gr.Dropdown(
                label="SAM Checkpoint",
                choices=[sam_checkpoint],
                value=sam_checkpoint if os.path.exists(sam_checkpoint) else None,
                allow_custom_value=True
            )

            sam_config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam_config = gr.Dropdown(
                label="SAM Config",
                choices=[sam_config_path],
                value=sam_config_path if os.path.exists(sam_config_path) else None,
                allow_custom_value=True
            )

            # Output directory
            save_dir = gr.Textbox(
                label="Output Directory",
                value="test/predictData",
                placeholder="Path to save prediction files"
            )

            # Image selection
            # all_images = list_images("dataset")
            # dataset\Kasthuri++\Test_In
            # dataset\Kasthuri++\Train_In
            # dataset\Lucchi++\Test_In
            # dataset\Lucchi++\Train_In
            # dataset\Kasthuri++\Test_Out
            # dataset\Kasthuri++\Train_Out
            # dataset\Lucchi++\Test_Out
            # dataset\Lucchi++\Train_Out
            Kasthuri_test_in = list_images("dataset\Kasthuri++\Test_In")
            Kasthuri_train_in = list_images("dataset\Kasthuri++\Train_In")
            Lucchi_test_in = list_images("dataset\Lucchi++\Test_In")
            Lucchi_train_in = list_images("dataset\Lucchi++\Test_In")

            Kasthuri_test_out = list_images("dataset\Kasthuri++\Test_Out")
            Kasthuri_train_out = list_images("dataset\Kasthuri++\Train_Out")
            Lucchi_test_out = list_images("dataset\Lucchi++\Test_Out")
            Lucchi_train_out = list_images("dataset\Lucchi++\Test_Out")

            image_input = gr.Dropdown(
                label="Input Image",
                choices=Kasthuri_test_in + Kasthuri_train_in + Lucchi_test_in + Lucchi_train_in + ["custom"],
                value=Kasthuri_test_in[0] if Kasthuri_test_in else None
            )

            custom_image = gr.Image(label="Or upload custom image", visible=False, type="filepath")

            def update_custom_image_visibility(choice):
                return gr.update(visible=choice == "custom")

            image_input.change(update_custom_image_visibility, inputs=image_input, outputs=custom_image)

            mask_input = gr.Dropdown(
                label="Ground Truth Mask (Optional)",
                choices=Kasthuri_test_out + Kasthuri_train_out + Lucchi_test_out + Lucchi_train_out + ["custom", "none"],
                value=Kasthuri_test_out[0] if Kasthuri_test_out else None
            )

            custom_mask = gr.Image(label="Or upload custom mask", visible=False, type="filepath")

            def update_custom_mask_visibility(choice):
                return gr.update(visible=choice == "custom")

            mask_input.change(update_custom_mask_visibility, inputs=mask_input, outputs=custom_mask)

            # Process button
            process_btn = gr.Button("Process Images")

        with gr.Column(scale=2):
            # Output displays
            output_gallery = gr.Gallery(label="Results", columns=3, height=600)
            metrics_output = gr.Textbox(label="Metrics", lines=10)
    # Add a new row for the comparison images
    with gr.Row():
        gr.Markdown("## Key Comparison Images")
        unet_comparison = gr.Image(label="UNet Comparison", show_label=True)
        sam_comparison = gr.Image(label="SAM Ground Truth vs Prediction", show_label=True)

    def get_actual_paths(image_choice, custom_image_path, mask_choice, custom_mask_path):
        """Get the actual file paths based on user selection"""
        image_path = custom_image_path if image_choice == "custom" else image_choice

        if mask_choice == "none":
            mask_path = None
        elif mask_choice == "custom":
            mask_path = custom_mask_path
        else:
            mask_path = mask_choice

        return image_path, mask_path

    def process_and_display(unet_model_path, sam_model_path, sam_config_path, save_dir,
                            image_choice, custom_image, mask_choice, custom_mask):
        # Existing code
        image_path, mask_path = get_actual_paths(image_choice, custom_image, mask_choice, custom_mask)

        if not image_path:
            return None, "Please select an input image.", None, None

        try:
            # Make sure the save directory exists
            os.makedirs(save_dir, exist_ok=True)

            output_files, metrics_text = process_images(
                unet_model_path, sam_model_path, sam_config_path,
                image_path, mask_path, save_dir
            )

            # If no output files were found, check if they exist already
            if not output_files:
                output_files, existing_metrics = get_output_files(image_path, save_dir)
                if existing_metrics and not metrics_text:
                    metrics_text = existing_metrics

            # Get paths for the specific comparison images
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]

            unet_comparison_path = os.path.join(save_dir, f"{name_without_ext}_comparison.png")
            sam_comparison_path = os.path.join(save_dir, "sam_gt_pred_comparison.png")

            # Load images if they exist
            unet_img = Image.open(unet_comparison_path) if os.path.exists(unet_comparison_path) else None
            sam_img = Image.open(sam_comparison_path) if os.path.exists(sam_comparison_path) else None

            return output_files, metrics_text, unet_img, sam_img

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", None, None

    process_btn.click(
        process_and_display,
        inputs=[unet_model, sam_model, sam_config, save_dir,
                image_input, custom_image, mask_input, custom_mask],
        outputs=[output_gallery, metrics_output, unet_comparison, sam_comparison]
    )

if __name__ == "__main__":
    demo.launch()
