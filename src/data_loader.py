import os
import cv2
import matplotlib.pyplot as plt

# === Base paths ===
BASE_IMG = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\clips"
BASE_MASK = r"C:\Users\amanu\Desktop\Projects\LaneDetector\TUSimple\train_set\seg_label"

def get_image_and_mask(sequence_id: str, frame_name: str):
    """
    Load an image and its corresponding mask.
    sequence_id: folder name inside clips/ and seg_label/
    frame_name: image filename, e.g., "20.jpg"
    """
    folder_img = os.path.join(BASE_IMG, sequence_id)
    folder_mask = os.path.join(BASE_MASK, sequence_id)

    img_path = os.path.join(folder_img, frame_name)
    mask_name = os.path.splitext(frame_name)[0] + ".png"
    mask_path = os.path.join(folder_mask, mask_name)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")

    return img, mask

def visualize_image_mask_overlay(img, mask):
    """
    Display image, mask, and overlay.
    """
    overlay = img.copy()
    overlay[mask > 0] = [255, 0, 0]

    plt.figure(figsize=(12, 6))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Image"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("Overlay"); plt.axis("off")
    plt.show()


# === Test ===
if __name__ == "__main__":
    seq_id = "0531/1492626270684175793"
    frame = "20.jpg"
    img, mask = get_image_and_mask(seq_id, frame)
    visualize_image_mask_overlay(img, mask)
