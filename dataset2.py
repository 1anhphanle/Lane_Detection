# dataset2.py
import cv2
import numpy as np
from utils import *
from matplotlib import pyplot as plt

# === Biến toàn cục ===
memory = []  # Lưu trữ lịch sử ảnh
framerate = 20  # Tốc độ khung hình
K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
              [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])  # Ma trận camera
dist = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03,
                 -8.79107779e-05, 2.20573263e-02]])  # Hệ số méo
# Điểm nguồn và đích cho biến đổi phối cảnh
source_points = np.float32([[317.2, 203.6], [370.9, 203.2], [470.4, 291.6], [160.3, 291.6]])
dst_points = np.float32([[300, 70], [370, 70], [370, 291.6], [300, 291.6]])
H = cv2.getPerspectiveTransform(source_points, dst_points)  # Ma trận homography
H_inv = np.linalg.inv(H)


# === Các hàm xử lý ===
def std_least_square(data_points):
    """
    Khớp đường cong (parabol) sử dụng phương pháp bình phương tối thiểu.
    """
    if len(data_points) == 0:
        return None  # Trả về None nếu mảng đầu vào rỗng
    X = []
    Y = []
    for point in data_points:
        X.append([point[1] ** 2, point[1], 1])
        Y.append(point[0])

    X = np.array(X)
    Y = np.array(Y)

    try:
        Xt_X_inv = np.linalg.pinv(np.dot(X.T, X))
        Xt_Y = np.dot(X.T, Y)
        result = np.dot(Xt_X_inv, Xt_Y)
        return result
    except:
        return None  # Trả về None nếu có lỗi xảy ra


def trapezium_coordinates(x_list, result_sls, y_limit):
    """
    Tìm điểm cực trị của đường cong.
    """
    if len(x_list) == 0 or result_sls is None:
        return ((0, 0), (0, 0))  # Trả về giá trị mặc định nếu đầu vào không hợp lệ
    max_x, max_y = 0, 0
    min_x, min_y = 999999, 999999

    for x in x_list:
        y = (result_sls[0]) * x ** 2 + (result_sls[1]) * x + result_sls[2]
        if y > max_y:
            if y > y_limit:
                max_y = y_limit
                max_x = x
            else:
                max_y = y
                max_x = x

        if y < min_y:
            if y < 0:
                min_y = 0
                min_x = x
            else:
                min_y = y
                min_x = x

    return ((max_x, max_y), (min_x, min_y))


def turn_prediction(extreme_points, counter):
    """
    Dự đoán khúc cua dựa trên độ dốc của làn đường.
    """
    bottom_centre_x, bottom_centre_y = (extreme_points[0][0] + extreme_points[2][0]) // 2, (
            extreme_points[0][1] + extreme_points[2][1]) // 2
    top_centre_x, top_centre_y = (extreme_points[1][0] + extreme_points[3][0]) // 2, (
            extreme_points[1][1] + extreme_points[3][1]) // 2

    # Kiểm tra mẫu số trước khi chia
    if top_centre_x - bottom_centre_x != 0:
        slope = (top_centre_y - bottom_centre_y) / (top_centre_x - bottom_centre_x)
    else:
        slope = 0  # Gán độ dốc bằng 0 khi mẫu số bằng 0

    if -8 < slope < 0:
        if counter < 400:
            return "Turn Left"
        else:
            return "Turn Right"
    elif 8 > slope > 0:
        if counter < 400:
            return "Turn Right"
        else:
            return "Turn Left"
    else:
        return "Move Straight"

def moving_avg(img, result_sls):
    """
    Làm mượt kết quả bằng moving average filter.
    """
    if len(memory) <= 3:
        if result_sls is not None:  # Kiểm tra result_sls trước khi thêm vào memory
            memory.append(img)
    else:
        if result_sls is not None:
            memory.pop(0)
            memory.append(img)

    avg = memory[0]
    for i in range(1, len(memory)):
        avg = cv2.addWeighted(avg, 0.8, memory[i], 0.2, 0)

    return avg


def pipeline(img, counter):
    """
    Pipeline xử lý ảnh cho Dataset 2.
    """
    global memory

    dst = undistort_image(img, K, dist)
    img_dst = cv2.warpPerspective(dst, H, (dst.shape[1], dst.shape[0]))  # Biến đổi phối cảnh

    # Tách làn đường màu vàng
    hsv = cv2.cvtColor(img_dst, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    thresh_yellow = cv2.bitwise_and(img_dst, img_dst, mask=mask_yellow)
    thresh_yellow = cv2.cvtColor(thresh_yellow, cv2.COLOR_BGR2GRAY)
    ret, thresh_yellow = cv2.threshold(thresh_yellow, 100, 255, cv2.THRESH_BINARY)

    # Tách làn đường màu trắng
    gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    ret, thresh_white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Kết hợp hai làn đường
    addition = cv2.bitwise_or(thresh_yellow, thresh_white)

    # Làm mịn
    blur = cv2.GaussianBlur(addition, (5, 5), 0)

    line_img = np.zeros_like(img_dst)

    # Khớp đường cong cho làn đường trắng
    w_indices = np.argwhere(thresh_white == 255)
    result_sls_w = std_least_square(w_indices)
    if result_sls_w is not None:
        wx = np.linspace(335, 385, 100)
        wy = (result_sls_w[0]) * wx ** 2 + (result_sls_w[1]) * wx + result_sls_w[2]
        white_points = trapezium_coordinates(wx, result_sls_w, img_dst.shape[0])
        wx = np.linspace(white_points[1][0], white_points[0][0], 100)
        wy = (result_sls_w[0]) * wx ** 2 + (result_sls_w[1]) * wx + result_sls_w[2]
        if abs(white_points[0][1] - img_dst.shape[0]) > 10:
            wx = np.append(wx, white_points[0][0])
            wy = np.append(wy, img_dst.shape[0])

    # Khớp đường cong cho làn đường vàng
    y_indices = np.argwhere(thresh_yellow == 255)
    result_sls_y = std_least_square(y_indices)
    if result_sls_y is not None:
        yx = np.linspace(290, 325, 100)
        yy = (result_sls_y[0]) * yx ** 2 + (result_sls_y[1]) * yx + result_sls_y[2]
        yellow_points = trapezium_coordinates(yx, result_sls_y, img_dst.shape[0])
        yx = np.linspace(yellow_points[1][0], yellow_points[0][0], 100)
        yy = (result_sls_y[0]) * yx ** 2 + (result_sls_y[1]) * yx + result_sls_y[2]
        if abs(yellow_points[0][1] - img_dst.shape[0]) > 10:
            yx = np.append(yx, yellow_points[0][0])
            yy = np.append(yy, img_dst.shape[0])

    # Vẽ làn đường
    if result_sls_w is not None and result_sls_y is not None:
        pts_left = np.array([np.transpose(np.vstack([yx, yy]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([wx, wy])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(line_img, np.int_(pts), (255, 0, 0))
        extreme_points = np.array([white_points[0], white_points[1], yellow_points[0], yellow_points[1]])
        turn = turn_prediction(extreme_points, counter)
    else:
        turn = "Move Straight"
        extreme_points = []

    # Biến đổi ngược phối cảnh
    line_img = cv2.warpPerspective(line_img, H_inv, (img_dst.shape[1], img_dst.shape[0]))
    final = cv2.bitwise_or(dst, line_img)
    final = moving_avg(final, result_sls_w)

    # Làm sắc nét ảnh
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    final = cv2.filter2D(final, -1, kernel)

    cv2.putText(final, turn, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return final


# === Xử lý chính ===
if __name__ == "__main__":
    cap = cv2.VideoCapture('Data2/challenge_video.mp4')
    counter = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        try:
            final = pipeline(frame, counter)
        except Exception as e:
            print("Lỗi xử lý frame:", e)
            continue

        cv2.imshow('Lane Detection', final)
        if cv2.waitKey(framerate) & 0xFF == ord('q'):
            break
        counter += 1

    cap.release()
    cv2.destroyAllWindows()