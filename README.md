## Lane Detection - Dự án Phát hiện Làn Đường

Dự án này được viết bằng Python và OpenCV để thực hiện phát hiện làn đường, mô phỏng hệ thống cảnh báo chệch làn đường được sử dụng trong xe tự lái. Dự án sử dụng hai bộ dữ liệu video khác nhau được ghi từ camera trên xe tự lái, từ đó phát triển thuật toán phát hiện làn đường và ước tính độ cong của làn đường để dự đoán hướng rẽ.

---

## Cấu trúc Dự án

Dự án bao gồm các file sau:

- **dataset1.py:** Xử lý ảnh tĩnh (.png) với làn đường tương đối thẳng.
- **dataset2.py:** Xử lý video (.mp4) với làn đường có độ cong rõ rệt.
- **utils.py:** Chứa các hàm chung được sử dụng bởi cả `dataset1.py` và `dataset2.py`.

## Phương pháp tiếp cận

Dự án sử dụng các kỹ thuật xử lý ảnh và thị giác máy tính sau:

- **Khử méo ảnh:** Sử dụng ma trận camera và hệ số méo để loại bỏ sự biến dạng hình học trong ảnh.
- **Phân ngưỡng:** Chuyển đổi ảnh sang dạng nhị phân để tách làn đường khỏi nền.
- **Làm mịn:** Sử dụng Gaussian Blur để giảm nhiễu trong ảnh.
- **Phát hiện cạnh:** Sử dụng Canny Edge Detection để phát hiện cạnh của làn đường.
- **Hough Transform:** Tìm kiếm các đường thẳng trong ảnh cạnh (dùng cho `dataset1`).
- **Biến đổi phối cảnh (Bird-eye View Transformation):** Chuyển đổi góc nhìn từ camera sang bird-eye view (dùng cho `dataset2`).
- **Khớp đường cong:** Sử dụng phương pháp bình phương tối thiểu để khớp một đường cong (parabol) vào các điểm dữ liệu của làn đường (dùng cho `dataset2`).
- **Dự đoán khúc cua:** Ước tính độ cong của làn đường để dự đoán hướng rẽ của xe.

## Hướng dẫn Chạy Dự án

1. **Cài đặt:** Đảm bảo bạn đã cài đặt Python 3.6 (hoặc mới hơn), OpenCV và NumPy. Bạn có thể cài đặt bằng `pip`:

   ```
   pip install opencv-python numpy
   ```

2. **Chuẩn bị dữ liệu:**
    - Tải dữ liệu ảnh/video từ [đường dẫn đến dữ liệu của bạn].
    - Tạo thư mục `Data1` và `Data2` trong thư mục dự án và giải nén dữ liệu vào thư mục tương ứng.

3. **Chạy code:**
    - Để chạy `dataset1.py`:
      ```
      python dataset1.py
      ```
    - Để chạy `dataset2.py`:
      ```
      python dataset2.py
      ```

## Kết quả

Dự án sẽ hiển thị kết quả phát hiện làn đường trực tiếp trên video/ảnh với làn đường được đánh dấu và dự đoán hướng rẽ.

---

## Tác giả

Anh Phan Le