# src/deep_learning/inference.py
import os
import torch
import cv2
from unet_model import UNet
from utils import overlay_mask_on_image
import numpy as np

def inference_folder(input_dir, output_dir, ckpt_path=None, img_size=(512,288), device=None):
    input_dir = os.path.normpath(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=1).to(device)
    if ckpt_path is None:
        ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "unet_tusimple_best.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])
    with torch.no_grad():
        for fname in files:
            path = os.path.join(input_dir, fname)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, img_size)
            inp = (img_resized.astype("float32") / 255.0).transpose(2,0,1)
            inp = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = torch.sigmoid(model(inp))[0][0].cpu().numpy()
            pred_mask = (out > 0.5).astype(np.uint8)
            # resize mask back to original image size
            mask_big = cv2.resize(pred_mask.astype("uint8")*255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = overlay_mask_on_image(img_rgb, mask_big//255)
            out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, fname), out_bgr)
    print(f"[INFO] Inference done. Outputs saved to {output_dir}")

if __name__ == "__main__":
    # edit these paths as needed
    test_images_dir = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\test_set\clips"  # or flattened test folder
    out_dir = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\test_predictions"
    # If test_set is nested, pass the flattened folder instead
    inference_folder(test_images_dir, out_dir)
