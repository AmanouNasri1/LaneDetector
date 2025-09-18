import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from unet_model import UNet

# === Paths ===
test_base = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\test_set\clips"
output_dir = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\test_prediction"
os.makedirs(output_dir, exist_ok=True)

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load model ===
model_path = r"C:\Users\amanu\Desktop\Projects\LaneDetector\src\deep_learning\unet_tusimple.pth"
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Helper function to get all images recursively ===
def get_all_images(base_dir):
    img_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_paths.append(os.path.join(root, file))
    return img_paths

# === Prediction ===
all_images = get_all_images(test_base)
print(f"Found {len(all_images)} images in test set.")

for img_path in tqdm(all_images, desc="Predicting"):
    # Load original image
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]

    # Preprocess: resize to model input
    img_resized = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img_resized.transpose(2,0,1)).unsqueeze(0).float()/255.0
    img_tensor = img_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        mask_pred = model(img_tensor)
        mask_pred = torch.sigmoid(mask_pred).squeeze().cpu().numpy()
        mask_pred = (mask_pred > 0.5).astype(np.uint8)  # binary mask

    # Resize mask back to original image size
    mask_pred_resized = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Overlay lanes on original image
    overlay = img.copy()
    overlay[mask_pred_resized > 0] = [0, 0, 255]  # red lanes

    # Save overlay
    rel_path = os.path.relpath(img_path, test_base)
    save_path = os.path.join(output_dir, rel_path.replace(os.sep, "_"))
    cv2.imwrite(save_path, overlay)
