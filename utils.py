# utils.py
import cv2
import numpy as np

def undistort_image(img, K, dist):
    """
    Khử méo ảnh sử dụng ma trận camera K và hệ số méo dist.
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, K, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def preprocess_image(img, roi_height=0.5):
    """
    Chuyển đổi ảnh sang grayscale và xác định vùng quan tâm (ROI).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    gray[0:int(rows * roi_height), :] = 0  # Loại bỏ phần ảnh trên đường chân trời
    return gray

def detect_edges(gray):
    """
    Phân ngưỡng, làm mịn và phát hiện cạnh của làn đường.
    """
    ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    edges = cv2.Canny(blur, 130, 200)
    return edges

def hough_lines(edges):
    """
    Sử dụng Probabilistic Hough Transform để tìm các đường thẳng.
    """
    lines = cv2.HoughLinesP(edges, rho=4, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=180)
    return lines