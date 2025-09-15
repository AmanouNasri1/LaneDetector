# src/deep_learning/utils.py
import os
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask_on_image(img_np, mask_np, color=(255,0,0), alpha=0.6):
    """
    img_np: uint8 HxWx3 (RGB)
    mask_np: binary HxW (0/1)
    returns overlay uint8 HxWx3
    """
    overlay = img_np.copy()
    mask_bool = mask_np.astype(bool)
    overlay[mask_bool] = (np.array(color) * alpha + overlay[mask_bool] * (1.0 - alpha)).astype(np.uint8)
    return overlay

def save_prediction_visual(img_np, gt_mask, pred_mask, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Image"); plt.imshow(img_np); plt.axis("off")
    plt.subplot(1,3,2); plt.title("GT Mask"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Prediction Overlay"); plt.imshow(overlay_mask_on_image(img_np, pred_mask)); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
