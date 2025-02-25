import os
import pandas as pd
import cv2
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
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
from samPrediction import read_batch

# 這裡是SAM2的訓練函數，用於把整個sam扔到數據集上進行訓練


# Lora
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, scaling=1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # 保存原始層
        self.linear = linear_layer

        # LoRA 組件
        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)
        self.scaling = scaling

        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # 凍結原始權重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        # 原始層的輸出
        orig_output = self.linear(x)
        # LoRA 路徑
        lora_output = self.lora_up(self.lora_down(x)) * self.scaling
        return orig_output + lora_output
    
def add_lora_to_model(model, rank=4, scaling=1.0, device="cuda"):
    """
    將 LoRA 添加到模型的關鍵組件，並確保所有組件都在正確的設備上
    """
    modified_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in [
            'sam_prompt_encoder', 'sam_mask_decoder'
        ]):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent_module = model

            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)

            original_layer = getattr(parent_module, child_name)
            # 創建 LoRA 層並移到指定設備
            lora_layer = LoRALinear(original_layer, rank=rank, scaling=scaling).to(device)
            setattr(parent_module, child_name, lora_layer)
            modified_layers.append((name, lora_layer))

    return modified_layers

def train_with_lora(predictor, train_data, num_steps=3000, device="cuda"):
    # 確保模型在正確的設備上
    predictor.model = predictor.model.to(device)

    # 添加 LoRA 層
    modified_layers = add_lora_to_model(predictor.model, rank=4, scaling=1.0, device=device)

    if not modified_layers:
        raise ValueError("No layers were modified with LoRA!")

    # 收集需要訓練的參數
    trainable_params = []
    for _, layer in modified_layers:
        trainable_params.extend([
            layer.lora_down.weight,
            layer.lora_up.weight
        ])

    if not trainable_params:
        raise ValueError("No trainable parameters found!")

    # 配置優化器
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4

    print(f"Training {len(trainable_params)} LoRA parameters")
    mean_iou = 0

    for step in range(1, num_steps + 1):
        with torch.cuda.amp.autocast():
            image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            # 將數據移到 GPU
            predictor.set_image(image)
            mask_input, ucc, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )

            if ucc is None or labels is None or ucc.shape[0] == 0 or labels.shape[0] == 0:
                continue

            for i in range(ucc.shape[0]):
                uc = ucc[i:i+1, :, :]
                # 確保輸入在正確的設備上
                uc = uc.to(device)
                labels = labels.to(device)

                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(uc, labels), boxes=None, masks=None,
                )

                batched_mode = uc.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0).to(device) for feat_level in predictor._features["high_res_feats"]]

                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0).to(device),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe().to(device),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                gt_mask = torch.tensor(mask.astype(np.float32), device=device)
                prd_mask = torch.sigmoid(prd_masks[:, 0])

                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) -
                          (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()

                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
                iou = inter / union
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                loss = seg_loss + score_loss * 0.05
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step()

            if step % 500 == 0:
                lora_state = {}
                for name, layer in modified_layers:
                    lora_state[f"{name}.lora_down.weight"] = layer.lora_down.weight
                    lora_state[f"{name}.lora_up.weight"] = layer.lora_up.weight
                torch.save(lora_state, f"sam2_lora_checkpoint_{step}.pth")

            mean_iou = mean_iou * 0.99 + 0.01 * torch.mean(iou).item()
            if step % 100 == 0:
                print(f"Step {step}:\tAccuracy (IoU) = {mean_iou:.4f}")

    return predictor





