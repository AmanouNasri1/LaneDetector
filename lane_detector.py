import cv2
import numpy as np
import csv
import os

# Configuration constants
ROI_VERTICES_RATIO = [(0.1, 1), (0.9, 1), (0.55, 0.6), (0.45, 0.6)]
CANNY_THRESHOLDS = (50, 150)
HOUGH_PARAMS = {
    "rho": 2,
    "theta": np.pi / 180,
    "threshold": 100,
    "min_line_len": 40,
    "max_line_gap": 5,
}
LANE_WIDTH_METERS = 3.7
PIXELS_PER_METER = 700  # approx lane width in pixels

def region_of_interest(img):
    height, width = img.shape[:2]
    vertices = np.array([[
        (int(x_ratio * width), int(y_ratio * height)) for x_ratio, y_ratio in ROI_VERTICES_RATIO
    ]], dtype=np.int32)
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(lines):
    if lines is None:
        return None, None
    
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2 - x1) < 20:  # filter near-vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Slope threshold to filter out noise and assign side
            if slope < -0.5:
                left_lines.append((slope, intercept))
            elif slope > 0.5:
                right_lines.append((slope, intercept))

    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None
    return left_avg, right_avg

def make_line_coordinates(image, line_params):
    if line_params is None:
        return None
    
    slope, intercept = line_params
    height = image.shape[0]
    y1 = height
    y2 = int(y1 * 0.6)
    
    # Protect against division by zero
    if slope == 0:
        return None
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(image, lines, color=(0, 255, 0), thickness=10):
    line_img = np.zeros_like(image)
    if lines:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def process_frame(frame, csv_writer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, *CANNY_THRESHOLDS)
    cropped_edges = region_of_interest(edges)
    
    lines = cv2.HoughLinesP(
        cropped_edges,
        HOUGH_PARAMS["rho"],
        HOUGH_PARAMS["theta"],
        HOUGH_PARAMS["threshold"],
        minLineLength=HOUGH_PARAMS["min_line_len"],
        maxLineGap=HOUGH_PARAMS["max_line_gap"]
    )
    
    left_avg, right_avg = average_slope_intercept(lines)
    output = frame.copy()

    if left_avg is not None and right_avg is not None:
        left_line = make_line_coordinates(frame, left_avg)
        right_line = make_line_coordinates(frame, right_avg)

        if left_line is not None and right_line is not None:
            lane_center = (left_line[2] + right_line[2]) // 2
            frame_center = frame.shape[1] // 2
            
            offset = (frame_center - lane_center) * LANE_WIDTH_METERS / PIXELS_PER_METER
            
            lane_lines_img = draw_lines(frame, [left_line, right_line])
            combined = cv2.addWeighted(frame, 0.8, lane_lines_img, 1, 1)

            # Draw vertical reference lines
            cv2.line(combined, (frame_center, frame.shape[0]), (frame_center, int(frame.shape[0]*0.6)), (255, 0, 0), 2)
            cv2.line(combined, (lane_center, frame.shape[0]), (lane_center, int(frame.shape[0]*0.6)), (0, 255, 255), 2)
            
            cv2.putText(combined, f"Offset: {offset:.2f} m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            csv_writer.writerow([offset])
            return combined

    # If detection fails, mark frame accordingly and log NaN offset
    cv2.putText(output, "Lane not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    csv_writer.writerow(["NaN"])
    return output

def main():
    os.makedirs("output", exist_ok=True)
    cap = cv2.VideoCapture('videos/test_video.mp4')

    if not cap.isOpened():
        print("❌ Error: Could not open input video!")
        return

    with open('lane_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Offset'])

        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read first frame.")
            return

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output/result_video.mp4', fourcc, 20.0, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, writer)
            out.write(processed_frame)
            cv2.imshow("Lane Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("✅ Done: Video saved to output/result_video.mp4")

if __name__ == "__main__":
    main()
