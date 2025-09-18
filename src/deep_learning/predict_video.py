import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from unet_model import UNet

# === Paths ===
video_input_path = r"C:\Users\amanu\Desktop\Projects\LaneDetector\videos\input\test_video.mp4"
video_output_path = r"C:\Users\amanu\Desktop\Projects\LaneDetector\videos\output\test_video_result.mp4"
model_path = r"C:\Users\amanu\Desktop\Projects\LaneDetector\src\deep_learning\unet_tusimple.pth"

# === Settings ===
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = (256, 256)  # UNet input size

# === Load model ===
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Video setup ===
cap = cv2.VideoCapture(video_input_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames...")

# === Process video frame by frame ===
with torch.no_grad():
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and normalize
        frame_resized = cv2.resize(frame, input_size)
        img = frame_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

        # Predict mask
        mask_pred = model(img_tensor)
        mask_pred = torch.sigmoid(mask_pred)
        mask_pred = mask_pred.squeeze().cpu().numpy()
        mask_pred = (mask_pred > 0.5).astype(np.uint8)  # threshold
        mask_pred = cv2.resize(mask_pred, (width, height))

        # Overlay mask (red lanes)
        overlay = frame.copy()
        overlay[mask_pred>0] = [0,0,255]

        # Write frame to output video
        out.write(overlay)

cap.release()
out.release()
print(f"[INFO] Video processed! Saved to {video_output_path}")
