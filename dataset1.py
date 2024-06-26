# dataset1.py
import cv2
import glob
import numpy as np
from utils import * 

# Biến toàn cục
memory = []  # Lưu trữ lịch sử làn đường
framerate = 20  # Tốc độ khung hình
K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
            [0.000000e+00, 9.019653e+02, 2.242509e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]])  # Ma trận camera
dist = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])  # Hệ số méo

def line_selection(img, lines):
    """
    Lựa chọn hai đường thẳng đại diện cho hai làn đường.
    """
    try:
        upper_thresh_slope_r = 3
        upper_thresh_slope_l = 1.5
        lower_thresh_slope = 0.4
        y_min, y_max = img.shape[0], img.shape[0]
        left_slope, left_lane, right_slope, right_lane = [], [], [], []

        for line in lines:
            for x1, y1, x2, y2 in line:
                # Kiểm tra điều kiện trước khi chia
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                else:
                    # Xử lý trường hợp x2 - x1 = 0
                    slope = float('inf') 

                if slope < upper_thresh_slope_r and slope > lower_thresh_slope:
                    right_slope.append(slope)
                    right_lane.append(line)
                elif slope > -upper_thresh_slope_l and slope < -lower_thresh_slope:
                    left_slope.append(slope)
                    left_lane.append(line)

                y_min = min(y_min, y1, y2)

        left_slope_mean = np.mean(left_slope) if left_slope else 0  # Handle empty list case
        right_slope_mean = np.mean(right_slope) if right_slope else 0
        left_mean = np.mean(np.array(left_lane), axis=0) if left_lane else [0, 0, 0, 0]
        right_mean = np.mean(np.array(right_lane), axis=0) if right_lane else [0, 0, 0, 0]
 
        # b = y - m*x
        left_b = left_mean[0][1] - (left_slope_mean * left_mean[0][0])
        right_b = right_mean[0][1] - (right_slope_mean * right_mean[0][0])

        left_x1 = int((y_min - left_b) / left_slope_mean) if left_slope_mean else 0 # Handle zero division
        left_x2 = int((y_max - left_b) / left_slope_mean) if left_slope_mean else 0
        right_x1 = int((y_min - right_b) / right_slope_mean) if right_slope_mean else 0
        right_x2 = int((y_max - right_b) / right_slope_mean) if right_slope_mean else 0

        if left_x1 > right_x1:
            left_x1 = (left_x1 + right_x1) // 2
            right_x1 = left_x1
            left_y1 = int((left_slope_mean * left_x1) + left_b)
            right_y1 = int((right_slope_mean * right_x1) + right_b)
            left_y2 = int((left_slope_mean * left_x2) + left_b)
            right_y2 = int((right_slope_mean * right_x2) + right_b)
        else:
            left_y1 = y_min
            left_y2 = y_max
            right_y1 = y_min
            right_y2 = y_max
        
        return [(left_x1, left_y1), (left_x2, left_y2), (right_x1, right_y1), (right_x2, right_y2)]
    
    except Exception as e:
        print(e)
        return []

def moving_avg(curr_lane):
    """
    Làm mượt kết quả bằng moving average filter.
    """
    # Moving average filter on last 5 frames
    if len(memory) < 5:
        memory.append(curr_lane)
    else:
        memory.popleft()
        memory.append(curr_lane)

    left_x1, left_y1, left_x2, left_y2 = 0, 0, 0, 0
    right_x1, right_y1, right_x2, right_y2 = 0, 0, 0, 0
    for lane in memory:
        left_x1 += lane[0][0]
        left_y1 += lane[0][1]
        left_x2 += lane[1][0]
        left_y2 += lane[1][1]
        right_x1 += lane[2][0]
        right_y1 += lane[2][1]
        right_x2 += lane[3][0]
        right_y2 += lane[3][1]

    left_x1 //= len(memory)
    left_y1 //= len(memory)
    left_x2 //= len(memory)
    left_y2 //= len(memory)
    right_x1 //= len(memory)
    right_y1 //= len(memory)
    right_x2 //= len(memory)
    right_y2 //= len(memory)

    return [(left_x1, left_y1), (left_x2, left_y2), (right_x1, right_y1), (right_x2, right_y2)]

def turn_prediction(extreme_points):
    """
    Dự đoán khúc cua dựa trên độ dốc của làn đường.
    """
    bottom_centre_x = (extreme_points[1][0] + extreme_points[3][0]) // 2
    bottom_centre_y = (extreme_points[1][1] + extreme_points[3][1]) // 2
    top_centre_x = (extreme_points[0][0] + extreme_points[2][0]) // 2
    top_centre_y = (extreme_points[0][1] + extreme_points[2][1]) // 2
    slope = (top_centre_y - bottom_centre_y) / (top_centre_x - bottom_centre_x)

    if -5 < slope < 0:
        return "Turn Left"
    elif 5 > slope > 0:
        return "Turn Right"
    else:
        return "Move Straight"
    
def pipeline(img):
    """
    Pipeline xử lý ảnh cho Dataset 1.
    """
    global memory
    dst = undistort_image(img, K, dist)
    gray = preprocess_image(dst)
    edges = detect_edges(gray)
    lines = hough_lines(edges)

    lanes = line_selection(dst, lines)
    try:
        lanes = moving_avg(lanes)
    except:
        pass

    if len(lanes) != 0:
        cv2.line(dst, (lanes[0][0], lanes[0][1]), (lanes[1][0], lanes[1][1]), (0, 0, 255), 6)
        cv2.line(dst, (lanes[2][0], lanes[2][1]), (lanes[3][0], lanes[3][1]), (0, 0, 255), 6)

        bottom_centre_x, bottom_centre_y = (lanes[1][0] + lanes[3][0]) // 2, (lanes[1][1] + lanes[3][1]) // 2
        top_centre_x, top_centre_y = (lanes[0][0] + lanes[2][0]) // 2, (lanes[0][1] + lanes[2][1]) // 2
        turn = turn_prediction(lanes)
        cv2.line(dst, (bottom_centre_x, bottom_centre_y), (top_centre_x, top_centre_y), (0, 0, 255), 3)

        # Filling the lane mesh
        new_img = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)
        cnt = np.array([[[lanes[0][0], lanes[0][1]], [lanes[1][0], lanes[1][1]], [lanes[3][0], lanes[3][1]],
                        [lanes[2][0], lanes[2][1]]]], dtype=np.int32)
        cv2.fillPoly(new_img, cnt, (255, 0, 0))
        dst = cv2.bitwise_or(dst, new_img)

    return dst

if __name__ == "__main__":
    for img_path in sorted(glob.glob('Data1/data/*.png')):
        img = cv2.imread(img_path)
        dst = pipeline(img)
        cv2.imshow("Lane Detection", dst)
        if cv2.waitKey(framerate) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()