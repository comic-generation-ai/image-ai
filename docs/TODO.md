# Image Generation Service (image-ai) - Implementation TODO List

Đây là danh sách các tác vụ triển khai dịch vụ sinh ảnh và hậu kỳ tự động (`image-ai`) trong hệ thống ComicSystem.

Mục tiêu: đưa MVP hiện tại lên mức **ổn định (không sập VRAM)**, **đúng logic cache**, và **production-ready** đủ để đưa vào luận văn (kiến trúc, độ tin cậy, đo lường).

## Trạng thái MVP hiện tại (theo code hiện tại)
*   **Đã chạy được (Kiểm tra bằng `test_client.py`)**: gRPC Server, FastAPI `/healthz`, Celery Worker, Redis Cache, MinIO Storage, SDXL Turbo Pipeline, Pillow Captioning.
*   **Cancel Task (gRPC)**: đã có hàm `CancelTask` và dùng `celery_app.control.revoke(..., terminate=True)` (nhưng cần production hardening).
*   **Safety Checker**: đã dùng classifier NSFW thực (`Falconsai/nsfw_image_detection`), cần benchmark thêm độ chính xác theo tập dữ liệu luận văn.
*   **Metrics**: `/metrics` đã export Prometheus (`prometheus_client`) với counter/histogram chính.
*   **Cache**: đã nâng hash key theo schema v2 (`prompt normalize + seed + caption + width/height + steps + model_id + guidance_scale + lora_id`) và có version key.
---

## Tiêu chí “DONE” để gọi là production-ready (gợi ý)
*   Không chạy lại GPU sai do cache hit nhầm (cache key đúng tham số ảnh hưởng kết quả).
*   Có cơ chế chống thundering herd (cache miss nhiều request giống nhau chỉ tạo 1 lần inference).
*   Safety checker chặn NSFW/ nội dung không phù hợp và trả trạng thái rõ ràng.
*   Metrics Prometheus thật + log có correlation id để đối soát.
*   Có retry/timeout cho các lỗi mạng (MinIO/Redis) nhưng không retry vô hạn khi OOM.
*   Cancel làm sạch VRAM và cập nhật trạng thái hợp lý (không để task treo).

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
- [x] **1.4 Production config discipline**
    - [x] Tách rõ DEV/PROD bằng biến môi trường (ví dụ: prefix `IMAGE_AI_...`)
    - [x] Bổ sung setting cho: `PRESIGNED_TTL_SECONDS`, `REDIS_CACHE_TTL_SECONDS`, `CELERY_TASK_TIME_LIMIT`, `MAX_STEPS`, `MAX_WIDTH/HEIGHT`
    - [x] Pin version các thư viện quan trọng để tránh vỡ môi trường (diffusers/torch không khớp)
- [ ] **1.5 Model artifact quản lý**
    - [ ] Xác định model id(s), cách tải model (local path vs remote hub) và hành vi khi offline
    - [ ] Nếu dùng LoRA: quy ước nơi lưu cache weights (RAM/disk) để giảm download lặp

---

## Phase 2: Core AI & Image Processing Components
- [x] **2.1 Stable Diffusion Pipeline**
    - [x] Triển khai `src/core/pipeline_runner.py` nạp model SDXL Turbo
    - [x] Hỗ trợ tự động chuyển thiết bị (CUDA / Apple Silicon MPS / CPU)
    - [x] Áp dụng VAE Slicing và Tiling để tối ưu bộ nhớ
- [x] **2.2 Model warmup & singleton pipeline per worker**
    - [x] Đảm bảo pipeline được khởi tạo **1 lần khi worker start**, không khởi tạo lại cho mỗi task
    - [x] (Optional) warmup một lần inference cực nhẹ để “mở khóa” kernels trước khi nhận request thật
    - [x] Gắn warmup vào lifecycle Celery worker (ví dụ signal hoặc init global)
- [x] **2.3 VRAM & Memory Management**
    - [x] Triển khai `src/core/vram_manager.py` dọn dẹp cache sau mỗi lần sinh ảnh
    - [x] Quản lý khóa Lock tuần tự hóa tác vụ trên GPU
- [x] **2.4 VRAM cleanup on cancel/timeout**
    - [x] Bảo đảm `vram_manager.clear_cache()` chạy trong mọi nhánh: SUCCESS/FAILED/CANCELLED/timeout/revoke
    - [x] Xử lý trường hợp task bị terminate giữa chừng: hạn chế “kẹt” VRAM (document hạn chế + cách mitigations)
- [ ] **2.5 Dynamic LoRA Loader**
    - [ ] Hoàn thiện `src/core/lora_loader.py` để nạp/hủy LoRA thực từ diffusers (bật/tắt adapter đúng cách)
    - [ ] Thiết kế cơ chế “LoRA per request”: mỗi task có adapter name riêng để tránh xung đột request song song
    - [ ] Tích hợp tải LoRA từ MinIO (nếu lora là artifact được lưu ở object storage)
    - [ ] Cache LoRA weights (tránh tải lại cho mỗi task) + policy TTL/LRU nếu bộ nhớ hạn chế
- [ ] **2.6 Content Safety Checker (NSFW thực)**
    - [ ] Nâng cấp `src/core/safety_checker.py` từ giả lập thành check thực
        - [ ] Gợi ý công nghệ (chọn 1 để luận văn gọn):
            - [ ] Dùng safety checker tích hợp trong `diffusers` (nếu phù hợp model) hoặc
            - [ ] Dùng mô hình CLIP-based safety checker (nhanh, nhẹ cho thesis), hoặc
            - [ ] Dùng `nsfw-detector` (nếu ổn định với phiên bản môi trường)
    - [ ] Safety policy rõ ràng: nếu NSFW -> trả status `FAILED`/`BLOCKED` (nếu không đổi proto thì map vào `FAILED` + `error_message`)
    - [ ] Log lý do chặn (an toàn cho luận văn) và đảm bảo vẫn dọn VRAM
- [x] **2.7 Pillow Post-Processing (Hậu kỳ chèn thoại)**
    - [x] Triển khai thuật toán tự động bẻ chữ xuống dòng (`wrap_text`) dựa trên kích thước font chữ thực tế
    - [x] Vẽ viền đen bao quanh tranh truyện tranh
    - [x] Tạo bong bóng thoại (Speech bubble) bo tròn có màu trắng đục (Alpha Channel) ở đáy ảnh
    - [x] Căn giữa chữ và render Tiếng Việt Unicode sắc nét
- [ ] **2.8 Comic layout robustness**
    - [ ] Cache font objects (tránh load font liên tục)
    - [ ] Chuẩn hóa font fallback khi font không tồn tại
    - [ ] Giới hạn độ dài caption và fallback bố cục (tránh tràn layout)
- [x] **2.9 Cache correctness (hash key phải bao phủ mọi tham số ảnh hưởng output)**
    - [x] Bổ sung các tham số vào hash:
        - [x] `steps`
        - [x] `model_id` (và variant nếu có)
        - [x] LoRA id/name (nếu có)
        - [x] Các tham số pipeline ảnh hưởng output (guidance_scale, format nếu ảnh hưởng)
    - [x] Chuẩn hóa prompt đầu vào (trim, collapse whitespace, normalize newline) để giảm “cache miss giả”
- [ ] **2.10 Determinism & seed policy**
    - [ ] Nếu `seed=-1`: sinh seed ngẫu nhiên và **lưu seed thực** vào log/task_result để tái lập
    - [ ] Dùng `torch.Generator` theo device đúng cách để seed có ý nghĩa
- [ ] **2.11 Character Consistency (Tính nhất quán nhân vật)**
    - [ ] Tích hợp IP-Adapter hoặc ControlNet Reference-Only vào pipeline SDXL Turbo
    - [ ] Đọc ảnh tham chiếu từ `reference_image_url` được truyền từ gRPC Request
    - [ ] Cache đặc trưng nhân vật (feature embeddings) để tránh tính toán lại
- [ ] **2.12 Dynamic & Advanced Speech Bubbles (Khung thoại động)**
    - [ ] Triển khai parse danh sách `speech_bubbles` từ gRPC Request (thay cho `caption_text` cố định)
    - [ ] Cải tiến Pillow vẽ khung thoại ở tọa độ động `(x_pos, y_pos)` với nhiều style (SPEECH, THOUGHT, SCREAM)
- [ ] **2.13 AI Image Upscaling (Siêu độ phân giải)**
    - [ ] Tích hợp mô hình Real-ESRGAN hoặc thuật toán nội suy chất lượng cao để phóng to ảnh lên 2K/4K
    - [ ] Cho phép bật/tắt upscaling theo config hoặc gRPC parameter để tiết kiệm GPU
- [ ] **2.14 Pre-GPU Text Moderation (Chặn từ khóa NSFW)**
    - [ ] Thực hiện kiểm tra, lọc từ khóa nhạy cảm trên `prompt` đầu vào ngay tại gRPC Server
    - [ ] Từ chối xử lý sớm các prompt vi phạm chính sách trước khi đưa vào Celery/GPU queue
- [ ] **2.15 Page Panel Layout Assembler (Ghép trang truyện)**
    - [ ] Triển khai module ghép nhiều panel thành một trang dọc (Webtoon style) hoặc trang lưới (Manga style)
    - [ ] Vẽ viền ngăn cách (panel borders) giữa các hình ảnh thành viên trong trang

---

## Phase 3: Storage & Caching
- [x] **3.1 Object Storage (MinIO)**
    - [x] Triển khai `src/storage/minio_client.py` tự tạo bucket
    - [x] Upload trực tiếp ảnh dạng byte-stream từ RAM (không ghi xuống đĩa cứng)
    - [x] Sinh link bảo mật có thời hạn (Presigned URL)
- [ ] **3.2 MinIO reliability**
    - [ ] Retry với backoff cho lỗi mạng (timeout, 5xx)
    - [ ] Đặt timeout cho network calls
    - [ ] Chuẩn hóa format upload (JPEG/PNG/WEBP) và content_type
    - [ ] Mệnh danh object theo quy ước chống trùng (nếu cache hit thì map đúng key)
- [x] **3.3 Caching (Redis)**
    - [x] Triển khai `src/cache/redis_cache.py` băm chuỗi tham số (MD5) thành Hash Key
    - [x] Lưu Cache Hit liên kết ảnh MinIO để tránh chạy lại GPU cho prompt trùng lặp
    - [x] Thiết lập TTL (Time-To-Live) tự động xóa cache sau 14 ngày
- [x] **3.4 Thundering herd protection (cache miss đồng thời)**
    - [x] Khi cache miss: dùng Redis lock theo `hash_key` (SET NX + expire) để chỉ **một** task chạy inference
    - [x] Các request còn lại chờ/hoặc poll đến khi key được set (tránh 2-3 GPU job trùng prompt)
- [x] **3.5 Cache versioning**
    - [x] Tạo prefix version cho cache (ví dụ `img_cache_v2:`) để khi thay đổi hash logic không bị hit sai
- [ ] **3.6 Celery result vs Redis cache thống nhất**
    - [ ] Đảm bảo status `PENDING/PROCESSING/SUCCESS/FAILED/CANCELLED` được ánh xạ nhất quán
    - [ ] Nếu task bị revoke: Redis cache không được set “thành công” khi ảnh chưa tồn tại

---

## Phase 4: gRPC Service & Celery Queue
- [x] **4.1 gRPC Servicer**
    - [x] Triển khai `GenerateImageAsync` đẩy task vào hàng đợi và phản hồi tức thì Task ID
    - [x] Triển khai `GetTaskStatus` lấy trạng thái từ Celery backend
    - [x] Triển khai `CheckHealth` kiểm tra trạng thái sức khỏe gRPC
- [x] **4.2 gRPC Cancel Task (baseline đã có)**
    - [x] Triển khai hàm `CancelTask` trong `src/service/image_service.py` bằng `celery_app.control.revoke(..., terminate=True)`
    - [x] Hardening production:
        - [x] Hỗ trợ “soft revoke” trước khi terminate (giảm nguy cơ dừng giữa chừng)
        - [x] Cập nhật trạng thái task về `CANCELLED` rõ ràng (tránh để task treo)
        - [x] Đảm bảo cleanup VRAM chạy kể cả khi bị revoke/terminate
- [x] **4.3 Input validation & OOM prevention**
    - [x] Validate `width/height` nằm trong giới hạn cho thiết bị mục tiêu
    - [x] Validate `num_inference_steps` (ví dụ 1..20) để tránh request quá nặng
    - [x] Validate `caption_text` (giới hạn độ dài ký tự)
    - [x] Nếu invalid -> trả gRPC error code phù hợp và không push job vào queue
- [x] **4.4 Tránh lỗi Multi-processing trên macOS (SIGSEGV)**
    - [x] Chạy Celery Worker ở chế độ đơn tiến trình `--pool=solo` khi dev trên Mac
- [x] **4.5 Sửa lỗi gRPC Backend**
    - [x] Truyền đối tượng `celery_app` vào `AsyncResult` trong gRPC để đọc trạng thái chính xác
- [ ] **4.6 Worker reliability (timeouts/retries/circuit breaking)**
    - [x] `task_time_limit` / `soft_time_limit` để tránh task chết ngầm
    - [ ] Retry chính sách cho lỗi MinIO/Redis (network) với số lần retry giới hạn
    - [ ] Không retry khi gặp OOM (hoặc retry với steps nhỏ hơn) để tránh spam GPU
    - [ ] Bổ sung vòng bắt exception có taxonomy rõ ràng để trả `error_message` dễ hiểu
- [ ] **4.7 Backpressure & rate limiting**
    - [ ] Giới hạn số request chờ bằng queue config (CELERY/QoS)
    - [ ] Rate limit theo user (nếu có auth) hoặc theo endpoint chung
- [ ] **4.8 Correlation id & audit log**
    - [ ] Thêm correlation id (có thể dùng `task_id`) vào log gRPC + log Celery để trace end-to-end

---

## Phase 5: Monitoring & Monitoring API
- [x] **5.1 Prometheus Metrics (export thật)**
    - [x] Thay `/metrics` thành Prometheus exporter bằng `prometheus_client`
    - [x] Metrics tối thiểu nên có:
        - [x] `image_requests_total{status=...}`
        - [x] `task_duration_seconds_bucket` (Histogram) theo `cached`/`success`
        - [ ] `cache_hit_total` và `cache_hit_ratio` (nếu tính được)
        - [x] `minio_upload_errors_total`
        - [x] `safety_blocks_total`
    - [ ] GPU metrics thực tế:
        - [ ] Dùng `pynvml` để đọc VRAM/utility (nếu chạy CUDA)
        - [ ] Nếu chạy MPS/CPU: export placeholder hoặc tách label `device_type`
    - [ ] (Optional) worker multiprocess mode nếu Celery fork nhiều process
- [x] **5.2 Health Monitor API**
    - [x] FastAPI `/healthz` kiểm tra kết nối Redis & MinIO
    - [x] Mở rộng health:
        - [x] kiểm tra GPU availability (CUDA/MPS)
        - [x] kiểm tra pipeline/model artifact sẵn sàng (để tránh “server sống nhưng generate chết”)
- [ ] **5.3 Tracing (tuỳ chọn cho luận văn)**
    - [ ] Gợi ý dùng OpenTelemetry + Jaeger/Zipkin nếu muốn chương “observability” thuyết phục hơn

---

## Phase 6: Testing & Documentation
- [x] **6.1 Client Test Script**
    - [x] Tạo file test `tests/test_client.py` mô phỏng đầy đủ hành vi gọi API từ Orchestrator
- [ ] **6.2 Unit Tests & Logic Tests**
    - [ ] Unit tests cho `wrap_text` (cases: tiếng Việt có dấu, nhiều dòng, biên giới font size)
    - [ ] Unit tests cho `redis_cache.generate_hash_key` theo schema v2 (bao phủ steps/model/lora)
    - [ ] Unit tests cho safety checker (mock classifier) để đảm bảo policy fail đúng
    - [ ] Unit tests cho minio upload flow (mock client) đảm bảo không ghi file cứng
- [ ] **6.3 Integration Tests (E2E nhẹ)**
    - [ ] Test “cache hit”: gọi 2 lần cùng prompt -> lần 2 không chạy inference (cần hook/log hoặc mock pipeline)
    - [ ] Test “cancel”: gửi request rồi cancel ngay -> status `CANCELLED` hoặc `FAILED` vì revoke, và đảm bảo worker không treo
- [ ] **6.4 Benchmarks (đưa vào luận văn)**
    - [ ] Inference latency: đo cold-start vs warm-start
    - [ ] Đo theo các cấu hình steps (vd: 4/8/10) và kích thước ảnh (vd: 512/1024)
    - [ ] Đo VRAM peak (CUDA) hoặc “ước lượng” theo thiết bị
    - [ ] Đo hiệu quả cache: cache hit ratio và thời gian tiết kiệm
- [ ] **6.5 Hướng dẫn sử dụng (Documentation)**
    - [ ] Viết file `docs/API.md` mô tả các tham số gRPC Request/Response + ví dụ request
    - [ ] Viết `docs/Runbook.md` (hoặc mục trong `README.md`) gồm: cách chạy dev/prod, cách xem metrics, cách xử lý lỗi
    - [ ] Hướng dẫn cách cài đặt LoRA tùy chỉnh (quy trình upload LoRA lên MinIO + format/đặt tên)

---

## Phase 7: Production Deployment
- [ ] **7.1 Docker Production**
    - [ ] Tối ưu hóa `Dockerfile` (Multi-stage build, giảm kích thước ảnh)
    - [ ] Chạy container dưới user không root
    - [ ] Cấu hình `HEALTHCHECK` cho cả gRPC và worker
    - [ ] Pin base image + thêm file lock nếu cần
- [ ] **7.2 GPU runtime & scaling**
    - [ ] Cấu hình chia sẻ GPU (`nvidia-container-toolkit`) trong `docker-compose.yml`
    - [ ] Thiết lập `CUDA_VISIBLE_DEVICES`/resource reservation theo từng worker
    - [ ] (Nếu scale nhiều worker) thiết kế scheduling: tránh 2 worker cùng “giành” GPU nếu concurrency không đủ
    - [ ] Tùy chọn cân nhắc: thêm hàng đợi theo GPU (nếu có nhiều GPU) để tăng throughput
- [ ] **7.3 Khởi động hệ thống tự động**
    - [ ] Setup script hoặc supervisor (hoặc systemd) tự khởi động lại gRPC server & worker khi gặp sự cố
- [ ] **7.4 CI/CD cho luận văn (tuỳ chọn nhưng nên có)**
    - [ ] Thêm pipeline chạy `pytest` và lint ở chế độ CPU-only
    - [ ] Build/test docker image cho CPU mode để đảm bảo reproducible
    - [ ] (Optional) pip-audit / dependency vulnerability scan

---

## Phase 8: Deliverables để đưa vào luận văn (đề xuất)
*   1 sơ đồ kiến trúc end-to-end (gRPC -> Celery -> MinIO -> Redis cache).
*   1 bảng so sánh: cold-start vs warm-start; cache hit vs cache miss.
*   1 phần “Reliability”: timeout/retry/cancel policy, VRAM cleanup.
*   1 phần “Safety”: pipeline safety checker + chính sách chặn.
*   1 phần “Observability”: metrics + log correlation id.
