import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import get_image_and_mask, visualize_image_mask_overlay

# === Utility functions ===

def region_of_interest(img):
    """
    Mask the image to keep only the road lane area.
    Adjusted for TuSimple images.
    """
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # TuSimple-optimized trapezoid
    polygon = np.array([[
        (int(0.05*width), height),
        (int(0.45*width), int(0.65*height)),
        (int(0.55*width), int(0.65*height)),
        (int(0.95*width), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def canny_edge_detector(img, low_threshold=50, high_threshold=150):
    """
    Apply grayscale -> Gaussian blur -> Canny edge detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    return edges

def hough_lines(img, min_line_length=20, max_line_gap=30):
    """
    Probabilistic Hough transform to detect lines.
    Returns image with lines drawn.
    """
    lines = cv2.HoughLinesP(img,
                            rho=1,
                            theta=np.pi/180,
                            threshold=30,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), 5)
    return line_img

def overlay_lines(original_img, line_img):
    """Overlay detected lines on original image"""
    return cv2.addWeighted(original_img, 0.8, line_img, 1, 1)

# === Main Demo ===
if __name__ == "__main__":
    seq_id = "0531/1492626270684175793"
    frame_name = "20.jpg"

    # Load image + mask
    img, mask = get_image_and_mask(seq_id, frame_name)
    visualize_image_mask_overlay(img, mask)  # optional for reference

    # 1️⃣ Canny edges
    edges = canny_edge_detector(img, low_threshold=40, high_threshold=120)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edges")
    plt.axis("off")
    plt.show()

    # 2️⃣ Apply ROI
    roi_edges = region_of_interest(edges)
    plt.imshow(roi_edges, cmap="gray")
    plt.title("Edges in ROI")
    plt.axis("off")
    plt.show()

    # 3️⃣ Hough lines
    line_img = hough_lines(roi_edges, min_line_length=20, max_line_gap=30)
    overlay = overlay_lines(img, line_img)

    # 4️⃣ Display final detected lanes
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lanes")
    plt.axis("off")
    plt.show()
