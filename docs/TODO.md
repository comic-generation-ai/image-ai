# Image Generation Service (image-ai) - Implementation TODO List

Đây là danh sách các tác vụ triển khai dịch vụ sinh ảnh và hậu kỳ tự động (`image-ai`) trong hệ thống ComicSystem.

## Trạng thái MVP hiện tại
*   **Đã chạy được (Kiểm tra bằng `test_client.py`)**: gRPC Server, FastAPI Health, Celery Worker, Redis Cache, MinIO Storage, SDXL Turbo Pipeline, Pillow Captioning.

---

## Phase 1: Setup & Infrastructure
- [x] **1.1 Cấu trúc & Môi trường**
    - [x] Khởi tạo virtual environment (`env`)
    - [x] Định nghĩa file dependencies `requirements.txt`
    - [x] Thiết lập Docker Compose chạy Redis & MinIO cục bộ
- [x] **1.2 Định nghĩa gRPC**
    - [x] Viết đặc tả gRPC Protobuf tại `proto/image_generation.proto`
    - [x] Viết script tự động biên dịch `scripts/generate_proto.sh`
    - [x] Biên dịch gRPC Python code thành công
- [x] **1.3 Cấu hình Tập trung (Centralized Config)**
    - [x] Định nghĩa `src/config/settings.py` sử dụng `Pydantic Settings`
    - [x] Chuyển đổi toàn bộ `os.getenv` trong mã nguồn sang sử dụng `settings` mới
    - [x] Tạo file `.env.example` làm mẫu cấu hình

---

## Phase 2: Core AI & Image Processing Components
- [x] **2.1 Stable Diffusion Pipeline**
    - [x] Triển khai `src/core/pipeline_runner.py` nạp model SDXL Turbo
    - [x] Hỗ trợ tự động chuyển thiết bị (CUDA / Apple Silicon MPS / CPU)
    - [x] Áp dụng VAE Slicing và Tiling để tối ưu bộ nhớ
- [x] **2.2 VRAM & Memory Management**
    - [x] Triển khai `src/core/vram_manager.py` dọn dẹp cache sau mỗi lần sinh ảnh
    - [x] Quản lý khóa Lock tuần tự hóa tác vụ trên GPU
- [ ] **2.3 Dynamic LoRA Loader**
    - [ ] Hoàn thiện `src/core/lora_loader.py` (Uncomment & tích hợp lệnh nạp/hủy LoRA động từ `diffusers`)
    - [ ] Hỗ trợ tải các model LoRA vẽ nhân vật từ MinIO về máy cục bộ để nạp khi có request
- [ ] **2.4 Content Safety Checker**
    - [ ] Triển khai bộ lọc NSFW thực tế tại `src/core/safety_checker.py` (Sử dụng model CLIP-safety-checker hoặc thư viện nsfw-detector siêu nhẹ)
- [x] **2.5 Pillow Post-Processing (Hậu kỳ chèn thoại)**
    - [x] Triển khai thuật toán tự động bẻ chữ xuống dòng (`wrap_text`) dựa trên kích thước font chữ thực tế
    - [x] Vẽ viền đen bao quanh tranh truyện tranh
    - [x] Tạo bong bóng thoại (Speech bubble) bo tròn có màu trắng đục (Alpha Channel) ở đáy ảnh
    - [x] Căn giữa chữ và render Tiếng Việt Unicode sắc nét

---

## Phase 3: Storage & Caching
- [x] **3.1 Object Storage (MinIO)**
    - [x] Triển khai `src/storage/minio_client.py` tự tạo bucket
    - [x] Upload trực tiếp ảnh dạng byte-stream từ RAM (không ghi xuống đĩa cứng)
    - [x] Sinh link bảo mật có thời hạn (Presigned URL)
- [x] **3.2 Caching (Redis)**
    - [x] Triển khai `src/cache/redis_cache.py` băm chuỗi tham số (MD5) thành Hash Key
    - [x] Lưu Cache Hit liên kết ảnh MinIO để tránh chạy lại GPU cho prompt trùng lặp
    - [x] Thiết lập TTL (Time-To-Live) tự động xóa cache sau 14 ngày

---

## Phase 4: gRPC Service & Celery Queue
- [x] **4.1 gRPC Servicer**
    - [x] Triển khai `GenerateImageAsync` đẩy task vào hàng đợi và phản hồi tức thì Task ID
    - [x] Triển khai `GetTaskStatus` lấy trạng thái từ Redis Backend
    - [x] Triển khai `CheckHealth` kiểm tra trạng thái sức khỏe gRPC
- [ ] **4.2 Hoàn thiện RPC Cancel Task**
    - [ ] Triển khai hàm `CancelTask` trong `src/service/image_service.py`
    - [ ] Hỗ trợ Celery Revoke/Terminate tác vụ đang chạy hoặc đang chờ trong queue
- [x] **4.3 Tránh lỗi Multi-processing trên macOS (SIGSEGV)**
    - [x] Chạy Celery Worker ở chế độ đơn tiến trình `--pool=solo` khi dev trên Mac
- [x] **4.4 Sửa lỗi gRPC Backend**
    - [x] Truyền đối tượng `celery_app` vào `AsyncResult` trong gRPC để đọc trạng thái chính xác

---

## Phase 5: Monitoring & Monitoring API
- [ ] **5.1 Prometheus Metrics**
    - [ ] Tích hợp chỉ số VRAM thực tế của GPU vào endpoint `/metrics` của FastAPI
    - [ ] Đo lường tổng số ảnh đã sinh ra, thời gian sinh trung bình (Histogram)
- [ ] **5.2 Health Monitor API**
    - [ ] FastAPI `/healthz` tự động kiểm tra trạng thái kết nối tới Redis và MinIO

---

## Phase 6: Testing & Documentation
- [x] **6.1 Client Test Script**
    - [x] Tạo file test `tests/test_client.py` mô phỏng đầy đủ hành vi gọi API từ Orchestrator
- [ ] **6.2 Unit Tests & Benchmarks**
    - [ ] Viết unit tests cho thuật toán bẻ dòng chữ `wrap_text`
    - [ ] Viết test đo hiệu năng (Inference Latency) trên thiết bị hiện tại
- [ ] **6.3 Hướng dẫn sử dụng**
    - [ ] Viết file `docs/API.md` mô tả các tham số gRPC Request/Response
    - [ ] Hướng dẫn cách cài đặt LoRA tùy chỉnh

---

## Phase 7: Production Deployment
- [ ] **7.1 Docker Production**
    - [ ] Tối ưu hóa `Dockerfile` (Sử dụng Multi-stage build để giảm kích thước ảnh docker)
    - [ ] Cấu hình chia sẻ GPU (`nvidia-container-toolkit`) trong `docker-compose.yml`
- [ ] **7.2 Khởi động hệ thống tự động**
    - [ ] Setup script hoặc supervisor tự động chạy lại gRPC server & worker khi gặp sự cố sập nguồn
