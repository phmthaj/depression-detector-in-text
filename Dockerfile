# 1. Chọn base image (vì bạn đang làm về Depression detection - thường dùng Deep Learning)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Tạo thư mục làm việc
WORKDIR /app

# 3. Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ thư mục dự án (bao gồm cả folder scripts) vào container
COPY . .

# 5. Lệnh để chạy file train nằm trong folder scripts
CMD ["python", "scripts/train.py"]