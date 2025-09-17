# src/deep_learning/flatten_tusimple.py
import os
import shutil

def flatten_clips_masks(clips_root, masks_root, out_images, out_masks):
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    count = 0
    for root, dirs, files in os.walk(clips_root):
        for file in files:
            if not file.lower().endswith(".jpg"):
                continue
            rel = os.path.relpath(root, clips_root)
            mask_path = os.path.join(masks_root, rel, os.path.splitext(file)[0] + ".png")
            if os.path.exists(mask_path):
                new_name = rel.replace(os.sep, "_") + "_" + file
                shutil.copy2(os.path.join(root, file), os.path.join(out_images, new_name))
                shutil.copy2(mask_path, os.path.join(out_masks, os.path.splitext(new_name)[0] + ".png"))
                count += 1
    print(f"[INFO] Copied {count} pairs to {out_images} and {out_masks}")

if __name__ == "__main__":
    clips = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\clips"
    masks = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\seg_label"
    out_images = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\images"
    out_masks  = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\masks"
    flatten_clips_masks(clips, masks, out_images, out_masks)
