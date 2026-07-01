# Image AI - Comic System Generation Service (Diffusion & Post-Processing)

Dịch vụ AI sinh ảnh và xử lý hậu kỳ tranh truyện tranh tự động nằm trong hệ thống ComicSystem.
Dịch vụ được thiết kế tối ưu hóa cho phần cứng GPU, giao tiếp bất đồng bộ thông qua gRPC và xử lý hàng đợi Celery.

## Tính năng & Kỹ thuật nổi bật trong đồ án
1. **Model Stable Diffusion 1.5 (Dreamshaper 8)**: `.env` hiện tại chạy `Lykon/dreamshaper-8` (SD 1.5, không phải turbo) ở 512×512, ~15 steps, CFG=7.0 — cân bằng tốc độ/chất lượng trên Mac. Code cũng hỗ trợ đổi model qua `IMAGE_AI_MODEL_ID` sang `stabilityai/sd-turbo` (4-8 steps, nhanh hơn) hoặc `Lykon/dreamshaper-xl-v2-turbo` (SDXL, cần GPU khỏe hơn) — xem comment trong `.env`.
2. **Hệ thống giao tiếp gRPC Asynchronous**: Toàn bộ business logic (`GenerateImageAsync`, `GetTaskStatus`, `CancelTask`, `CheckHealth`, `CheckGpuHealth`, `CheckCpuHealth`, `ClearGpuCache`) chạy qua gRPC/Protobuf; REST (FastAPI) chỉ phục vụ `/healthz` và `/metrics`.
3. **Message Queue (Celery + Redis)**: Xếp hàng xử lý tác vụ GPU tuần tự (`concurrency=1`, `worker_prefetch_multiplier=1`), loại bỏ hoàn toàn lỗi tràn bộ nhớ VRAM (`CUDA Out of Memory`).
4. **Hậu kỳ Pillow chèn chữ Tiếng Việt**: Tạo viền đen và hộp thoại màu trắng mờ (alpha channel), tự động bẻ chữ xuống dòng phù hợp kích thước khung truyện.
5. **Prompt Result Cache**: Băm chuỗi prompt (MD5, gồm cả `style`) và lưu kết quả URL ảnh trên Redis Cache giúp bỏ qua sinh ảnh GPU cho prompt trùng lặp; có lock chống thundering herd khi nhiều request cùng cache-miss.
6. **Object Storage (MinIO)**: Tải ảnh nhị phân trực tiếp từ RAM (`BytesIO`) lên MinIO và trả về đường dẫn Presigned URL bảo mật.
7. **Style Presets**: 5 style dựng sẵn (`storybook`, `anime`, `manga`, `retro`, `american_comic`) tự thêm suffix + negative prompt phù hợp; chọn qua field `style` trong request hoặc tag `[style:xxx]` ngay trong prompt.
8. **Tối ưu Apple Silicon (MPS)**: Xử lý riêng cho Mac M-series — decode VAE trên CPU + generator CPU để tránh NaN/ảnh đen khi chạy fp16 trên MPS. Sequential CPU offload có sẵn trong code nhưng **đang tắt** trong `.env` hiện tại vì Dreamshaper 8 (SD 1.5) nhẹ hơn SDXL, không cần offload — xem các biến `IMAGE_AI_MPS_*`.
9. **Output Validation & Safety**: Kiểm tra ảnh đen/hỏng trước khi upload (`OUTPUT_MIN_MEAN_BRIGHTNESS`) và lọc NSFW bằng `Falconsai/nsfw_image_detection` trước khi trả kết quả.

---

##  Cấu trúc dự án
*   `proto/`: Định nghĩa API giao tiếp gRPC (`.proto`).
*   `scripts/`: Script tự động biên dịch Protobuf sang Python code (`generate_proto.sh`).
*   `src/config/`: Cấu hình tập trung bằng Pydantic Settings (`IMAGE_AI_*` trong `.env`).
*   `src/core/`: Trái tim AI (Pipeline SDXL Turbo, nạp LoRA, kiểm tra VRAM, NSFW filter).
*   `src/utils/`: Logic xử lý ảnh (chèn caption, hậu kỳ sharpen/color, validate ảnh đen/hỏng).
*   `src/worker/`: Celery App + Task chính (`generate_image_task`) chạy ngầm trên GPU.
*   `src/service/`: gRPC servicer (`ImageGenerationService`) + code sinh ra từ proto (`service/generated/`).
*   `src/storage/` & `src/cache/`: Xử lý MinIO và Redis Cache.
*   `src/logger/` & `src/metrics/`: Logging chuẩn hóa + Prometheus metrics.
*   `src/server.py`: Điểm khởi chạy song song gRPC và FastAPI Health Monitor.
*   `docs/`: [`production_guide.md`](docs/production_guide.md) (kiến trúc production) và [`TODO.md`](docs/TODO.md) (roadmap chi tiết).
*   `tests/test_client.py`: Script test thủ công E2E (health, gRPC generate, cancel, cache hit).

---

##  Hướng dẫn cài đặt & Chạy dưới Local

### 1. Chuẩn bị môi trường Python ảo
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Tạo file cấu hình `.env`
`.env` (không commit git) đang có sẵn, cấu hình chạy `Lykon/dreamshaper-8` (SD 1.5, 512×512, 15
steps, CFG=7.0) — phù hợp Mac. Nếu cần tạo lại từ template:
```bash
cp .env.example .env
```
`.env.example` gợi ý mặc định khác — SDXL Turbo (`Lykon/dreamshaper-xl-v2-turbo`, ~16GB disk) hoặc
`stabilityai/sd-turbo` (~6GB, nhẹ hơn); đổi `IMAGE_AI_MODEL_ID` tùy máy. Trên Mac, giữ nguyên các
biến `IMAGE_AI_MPS_*` để tránh ảnh đen/NaN khi decode.

### 3. Biên dịch file Protobuf sang Python
```bash
./scripts/generate_proto.sh
```

### 4. Khởi chạy các dịch vụ bổ trợ bằng Docker (Redis & MinIO)
```bash
docker-compose up -d redis minio
```

### 5. Khởi chạy Server gRPC & FastAPI Health Monitor
```bash
python src/server.py
```

### 6. Khởi chạy Celery Worker (Cần GPU/MPS để chạy Stable Diffusion)
```bash
cd src && celery -A worker.celery_app worker --loglevel=info --concurrency=1
```
> Celery cần chạy từ thư mục `src/` để import `worker.*`/`config.*` đúng đường dẫn. Model/LoRA/
> offload chỉ được nạp **1 lần lúc worker start** — sau khi sửa `.env` phải **restart worker**.

### 7. Test nhanh E2E (health + gRPC generate + cancel + cache hit)
```bash
python tests/test_client.py
```

---

## Triển khai Đóng gói Toàn diện (Production)
> **Lưu ý hiện tại:** `docker-compose.yml` mới bật `redis` + `minio` (hạ tầng dev). Service
> `api-server` (gRPC) và `celery-worker` (GPU) đang bị **comment** trong file — chưa bật GPU
> runtime (`nvidia`) và Dockerfile production (xem `docs/TODO.md` Phase 7). Tới lúc đó, chạy
> gRPC server + Celery worker **local** theo các bước ở trên, chỉ dùng Docker cho Redis/MinIO:
```bash
docker-compose up -d redis minio
```
> Khi Phase 7 hoàn tất (bật lại 2 service trong compose + sửa path proto trong `Dockerfile`),
> `docker-compose up --build` sẽ dựng toàn bộ stack (gRPC Server, Celery Worker GPU, Redis, MinIO)
> trong 1 lệnh.
