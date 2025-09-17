# src/deep_learning/train.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import TuSimpleDataset
from unet_model import UNet

def train_main():
    # paths - edit if needed
    images_dir = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\images"
    masks_dir  = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\masks"
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # hyperparams
    batch_size = 2            # safe for RTX 3050
    img_size = (512, 288)     # (width, height)
    epochs = 20
    lr = 1e-3
    val_split = 0.1
    num_workers = 2 if os.name == "nt" else 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    dataset = TuSimpleDataset(images_dir, masks_dir, img_size)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Training on {len(train_ds)} images, validating on {len(val_ds)} images, device={device}")

    model = UNet(n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Grad scaler: use torch.cuda.amp for compat (older/newer)
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}  ({elapsed:.1f}s)")

        # checkpoint best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "unet_tusimple_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved best model to {best_path}")

    total = time.time() - start_time
    last_path = os.path.join(ckpt_dir, "unet_tusimple_last.pth")
    torch.save(model.state_dict(), last_path)
    print(f"[INFO] Training complete ({total/60:.1f} min). Last model saved to {last_path}")

if __name__ == "__main__":
    train_main()
