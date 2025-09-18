# src/deep_learning/evaluate.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TuSimpleDataset
from unet_model import UNet
from utils import save_prediction_visual
import cv2

def iou_score(pred, target, eps=1e-7):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + eps)

def dice_score(pred, target, eps=1e-7):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    inter = (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return 2 * inter / (denom + eps)

def evaluate_main():
    images_dir = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\images"
    masks_dir  = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\masks"
    ckpt_path  = os.path.join(os.path.dirname(__file__), "checkpoints", "unet_tusimple_best.pth")
    out_dir = os.path.join(os.path.dirname(__file__), "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = (512, 288)
    batch_size = 4

    dataset = TuSimpleDataset(images_dir, masks_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(n_classes=1).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ious, dices, accs = [], [], []

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = torch.sigmoid(model(imgs))
            preds = (outputs > 0.5).float()

            imgs_np = (imgs.cpu().numpy().transpose(0,2,3,1) * 255).astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)
            preds_np = preds.cpu().numpy().astype(np.uint8)

            for b in range(imgs_np.shape[0]):
                gt = masks_np[b][0]
                pr = preds_np[b][0]
                ious.append(iou_score(pr, gt))
                dices.append(dice_score(pr, gt))
                accs.append((pr == gt).mean())

                out_path = os.path.join(out_dir, f"sample_{idx}_{b}.png")
                save_prediction_visual(imgs_np[b], gt, pr, out_path)

    print(f"[RESULT] Mean IoU: {np.mean(ious):.4f}, Mean Dice: {np.mean(dices):.4f}, Pixel Acc: {np.mean(accs):.4f}")
    print(f"[INFO] Visualizations saved to {out_dir}")

if __name__ == "__main__":
    evaluate_main()
