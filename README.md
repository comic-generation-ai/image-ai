# Image AI - Comic System Generation Service (Diffusion & Post-Processing)

Dịch vụ AI sinh ảnh và xử lý hậu kỳ tranh truyện tranh tự động nằm trong hệ thống ComicSystem.
Dịch vụ được thiết kế tối ưu hóa cho phần cứng GPU, giao tiếp bất đồng bộ thông qua gRPC và xử lý hàng đợi Celery.

## Tính năng & Kỹ thuật nổi bật trong đồ án
1. **Model SDXL Turbo**: Sử dụng mô hình `Lykon/dreamshaper-xl-v2-turbo` sinh ảnh cực nhanh chỉ trong 4-8 bước (inference steps) giúp phản hồi tức thì.
2. **Hệ thống giao tiếp gRPC Asynchronous**: Độc lập hoàn toàn với backend REST thông thường, giao tiếp cực nhanh bằng Protobuf.
3. **Message Queue (Celery + Redis)**: Xếp hàng xử lý tác vụ GPU tuần tự (`concurrency=1`), loại bỏ hoàn toàn lỗi tràn bộ nhớ VRAM (`CUDA Out of Memory`).
4. **Hậu kỳ Pillow chèn chữ Tiếng Việt**: Tạo viền đen và hộp thoại màu trắng mờ (alpha channel), tự động bẻ chữ xuống dòng phù hợp kích thước khung truyện.
5. **Prompt Result Cache**: Băm chuỗi prompt (MD5) và lưu kết quả URL ảnh trên Redis Cache giúp bỏ qua sinh ảnh GPU cho prompt trùng lặp.
6. **Object Storage (MinIO)**: Tải ảnh nhị phân trực tiếp từ RAM (`BytesIO`) lên MinIO và trả về đường dẫn Presigned URL bảo mật.

---

##  Cấu trúc dự án
*   `proto/`: Định nghĩa API giao tiếp gRPC (`.proto`).
*   `scripts/`: Script tự động biên dịch Protobuf sang Python code.
*   `src/core/`: Trái tim AI (Pipeline SDXL Turbo, nạp LoRA, kiểm tra VRAM, NSFW filter).
*   `src/utils/`: Logic xử lý ảnh, ngắt dòng thoại tiếng Việt bằng Pillow.
*   `src/worker/`: Celery Worker quản lý tác vụ ngầm chạy trên GPU.
*   `src/storage/` & `src/cache/`: Xử lý MinIO và Redis Cache.
*   `src/server.py`: Điểm khởi chạy song song gRPC và FastAPI Health Monitor.

---

##  Hướng dẫn cài đặt & Chạy dưới Local

### 1. Chuẩn bị môi trường Python ảo
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Biên dịch file Protobuf sang Python
```bash
./scripts/generate_proto.sh
```

### 3. Khởi chạy các dịch vụ bổ trợ bằng Docker (Redis & MinIO)
```bash
docker-compose up -d redis minio
```

### 4. Khởi chạy Server gRPC & FastAPI Health Monitor
```bash
python src/server.py
```

### 5. Khởi chạy Celery Worker (Cần GPU để chạy Stable Diffusion)
```bash
celery -A worker.celery_app worker --loglevel=info --concurrency=1
```

---

## Triển khai Đóng gói Toàn diện (Production)
Chỉ cần chạy lệnh duy nhất để dựng toàn bộ hệ thống (gRPC Server, Celery Worker, Redis, MinIO) hỗ trợ GPU:
```bash
docker-compose up --build
```
