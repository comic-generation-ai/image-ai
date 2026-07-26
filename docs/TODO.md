# Image Generation Service (image-ai) - Implementation TODO List

Đây là danh sách các tác vụ triển khai dịch vụ sinh ảnh và hậu kỳ tự động (`image-ai`) trong hệ thống ComicSystem.

Mục tiêu cuối: **trang truyện 2×2 (4 panel)**, mỗi panel có **bong bóng thoại theo nhân vật**, **nhất quán nhân vật** giữa các khung, ảnh **sắc nét không nhòe**, đủ số liệu **benchmark Mac vs GPU cloud** cho luận văn.

---

## Lộ trình triển khai (3 giai đoạn)

| Giai đoạn | Mục tiêu | Tiêu chí hoàn thành |
|-----------|----------|---------------------|
| **A — Mac ổn định** | Pipeline chạy đúng `.env`, đo được baseline, không OOM | 1 panel 512×512 ≤ 90s (Dreamshaper 8); cache hit < 0.01s |
| **B — Tính năng truyện** | 4 panel + bubble đa nhân vật + ghép 2×2 + đồng bộ nhân vật | API `GenerateComicPage`; ảnh trang hoàn chỉnh trên MinIO |
| **C — GPU cloud (luận văn)** | Deploy worker CUDA, benchmark so sánh, demo bảo vệ | 4 panel ≤ 2 phút (1 GPU); bảng Mac vs Cloud trong thesis |

> **Lưu ý vận hành:** Sau mỗi lần sửa `.env` phải **restart Celery worker** — model/LoRA/offload được nạp **1 lần lúc worker start**, không đọc lại `.env` khi đang chạy.

---

## Benchmark đã đo (baseline — cần restart worker để cập nhật)

| Cấu hình (log 2026-07) | Diffusion | Tổng task | Ghi chú |
|---------------------------|-----------|-----------|---------|
| SDXL Turbo + LoRA + MPS sequential offload | ~192s (~48s/step) | **212s** | Log cũ 2026-06-06 — model đã đổi sang Dreamshaper 8 cho Mac |
| Cache hit (cùng prompt/seed) | 0s GPU | **0.01s** | Redis cache hoạt động đúng, trả presigned URL ngay |
| **`.env` hiện tại**: dreamshaper-8, 20 steps, DPM++ Karras, CFG=7.0, LoRA off, IP-Adapter off | ~2-4s/step | **~80-100s** | Đo thật trên Mac, RAM máy rảnh (~1.5GB free) |
| Cùng cấu hình nhưng RAM máy gần cạn (~65-100MB free) | ~30-40s/step | **~750-800s** | Swap-thrashing — do macOS thiếu RAM khi chạy nhiều app khác cùng lúc |
| + IP-Adapter bật (`IMAGE_AI_IP_ADAPTER_ENABLED=true`) | ~30-40s/step | **~745s** | CLIP vision encoder ngốn RAM, đẩy Mac 8GB vào swap. Code đã hỗ trợ đầy đủ, tắt mặc định trên Mac |

**Bài học vận hành quan trọng:** trên máy 8GB, tốc độ sinh ảnh phụ thuộc rất nhiều vào RAM còn trống lúc chạy (Chrome/IDE mở nhiều làm chậm gấp 10 lần dù code không đổi). Luôn đóng bớt app trước khi demo/benchmark.

---

## Trạng thái MVP hiện tại (theo code + log)

*   **Đã chạy E2E** (`tests/test_client.py`): gRPC, Celery, Redis, MinIO, diffusion, Pillow caption/postprocess, cache hit/miss.
*   **Pipeline**: hỗ trợ SD 1.x (Dreamshaper 8) + SDXL (SDXL Turbo, dreamshaper-xl); CUDA / MPS / CPU; warmup singleton.
*   **LoRA**: loader cơ bản có (`lora_loader.py`); hỗ trợ kiểm tra tương thích strict giữa LoRA và pipeline.
*   **Safety**: `Falconsai/nsfw_image_detection` đã tích hợp (`safety_checker.py`), đếm metric `safety_blocks_total`.
*   **Style**: 5 preset dựng sẵn (`storybook`, `anime`, `manga`, `retro`, `american_comic`) + tag `[style:xxx]` parse ngay trong prompt (`pipeline_runner.py`) — **đã xong**.
*   **Output validation**: chặn ảnh đen/hỏng (`OUTPUT_MIN_MEAN_BRIGHTNESS`), phát hiện upload rỗng (`MIN_UPLOAD_BYTES`) trước khi upload MinIO — **đã xong** (`utils/image_validation.py`).
*   **GPU/CPU health & VRAM**: `CheckGpuHealth`, `CheckCpuHealth`, `ClearGpuCache` đã implement đầy đủ trong `image_service.py` — **đã xong**.
*   **Seed in Status**: `GetTaskStatus` trả về `seed` thực tế được sử dụng trong `TaskStatusResponse` — **đã xong**.
*   **Auto-tuning & Heuristics**: gRPC `GenerateImageAsync` tự nâng steps lên default 20 nếu client gửi steps < 10 cho non-turbo model; tự boost steps khi phát hiện cảnh đông chủ thể (>=3 nhân vật) — **đã xong**.
*   **Regional Diffusion (2 nhân vật / 1 khung)**: đã nghiên cứu và phát triển module `src/core/regional_generator.py` cho SDXL. Đã thử nghiệm 6 phương pháp (full-shared-latent, physical crop-tiling, hybrid 2 pha, self-attention masking, inpaint 3 bước,...). Chốt giải pháp **physical crop tiling + linear feather blending** giải quyết triệt để lỗi "hợp thể nhân vật".
*   **Clean Reference Pipeline**: `IMAGE_AI_CAPTION_RENDER_ENABLED` mặc định `false` để trả về ảnh sạch cho FE tự render caption dưới panel (`PanelResult.caption_vi`), giúp ảnh panel 1 là reference sạch cho IP-Adapter ở panel 2-4 mà không bị rác chữ.
*   **Nhất quán nhân vật (IP-Adapter)**: **code đã xong** (`_load_reference_image`, `_blank_ip_adapter_image`, `set_ip_adapter_scale` động theo request, cache key + `cache_signature` tính cả reference/IP-Adapter state). Tắt mặc định qua `IMAGE_AI_IP_ADAPTER_ENABLED=false` trên Mac 8GB. Bật lại khi deploy GPU cloud.
*   **Prompt engineering**: tự động loại bỏ cú pháp Midjourney (`--ar 16:9`,...), ưu tiên giữ style suffix khi prompt dài (cắt bớt phần mô tả cảnh thay vì cắt rơi suffix), bỏ LoRA trigger words khi LoRA tắt.
*   **Cancel & Task Revocation**: có `CancelTask` gRPC; signal `task_revoked` tự giải phóng VRAM ngay. Với pool `solo` trên Mac, task đang generate dở không bị kill ngay lập tức mà chạy hết tự nhiên rồi bỏ kết quả (giới hạn Celery TaskPool).

---

## Bug đã fix (changelog rút gọn, 2026-07)

*   **`.env` không load được khi celery chạy từ `src/`** — `Settings.model_config.env_file` dùng đường dẫn tương đối theo cwd; celery bắt buộc chạy từ `src/` (để import `worker.*`) nhưng `.env` ở project root → Settings âm thầm rơi về default cứng. Đã fix: neo `env_file` theo vị trí `settings.py`.
*   **`/healthz` `pipeline_ready` luôn `false`** — check nhầm `pipeline_runner.pipeline` của process server.py (không bao giờ load model, chỉ Celery worker mới load). Đã fix: đọc cờ Redis `image_ai:worker_ready` do worker tự ghi sau khi warmup xong.
*   **`lora_loader.validate_compatibility()` không chặn LoRA SDXL chạy trên pipeline SD1.x** — do 2 biến `is_sdxl_pipeline`/`is_sd_pipeline` chồng lấn. Đã fix.
*   **`test_client.py`**: typo `prinst`→`print`; bỏ `continue` gây busy-loop bỏ qua `time.sleep(2)` khi poll `PROCESSING`.
*   **Prompt quá dài từ story-ai bị cắt mất style suffix** — đảo ưu tiên: giữ nguyên suffix, cắt bớt phần mô tả cảnh; thêm strip cú pháp Midjourney (`--ar 16:9`).
*   **IP-Adapter crash `NoneType is not iterable`** khi không có `reference_image_url` — diffusers bắt buộc luôn truyền `ip_adapter_image` một khi đã `load_ip_adapter()`. Đã fix: dùng ảnh trắng placeholder + `set_ip_adapter_scale(0)` khi không có reference thật.
*   **`Dockerfile` compile proto sai path** — đã sửa khớp `scripts/generate_proto.sh` (`--python_out=./src/service/generated`).

---

## Tiêu chí “DONE” production-ready

*   Cache key đúng mọi tham số ảnh hưởng output (version `.env` `IMAGE_AI_CACHE_KEY_VERSION=v6`; bao gồm prompt, seed, model, lora, scale, style, reference_image_url, format, quality).
*   Thundering herd lock khi cache miss đồng thời (`redis_cache.acquire_generation_lock`).
*   Safety checker + policy FAILED rõ ràng, đếm Prometheus metrics.
*   Metrics + benchmark reproducible (Mac + Cloud).
*   Retry MinIO/Redis; không retry vô hạn khi OOM.
*   Cancel giải phóng VRAM ngay qua Celery `task_revoked` signal.

---

## Phase 1: Setup & Infrastructure

- [x] **1.1 Cấu trúc & Môi trường** (`src/`, `proto/`, `scripts/`, `docs/`)
- [x] **1.2 Định nghĩa gRPC** (`proto/image_generation.proto`) — đầy đủ RPC health, status, cancel, clear cache, GPU/CPU stats, seed & speech bubbles schema.
- [x] **1.3 Cấu hình tập trung** (`settings.py`, prefix `IMAGE_AI_*`)
- [x] **1.4 Production config discipline** (neo `.env` tuyệt đối theo `settings.py`)
- [ ] **1.5 Model artifact quản lý**
    - [x] Document profile model: hiện tại `.env` dùng `Lykon/dreamshaper-8` (SD 1.5, Mac dev); `dreamshaper-xl-v2-turbo` cho Cloud
    - [ ] Cache HuggingFace tại `MODEL_CACHE_DIR`; hành vi offline
    - [x] Script / endpoint kiểm tra worker đang chạy model nào (`CheckGpuHealth`, `/healthz`)

---

## Phase 2: Core AI & Image Processing

### 2A — Tốc độ & chất lượng panel đơn (Giai đoạn A — Mac)

- [x] **2.1 Stable Diffusion Pipeline** (`pipeline_runner.py`)
- [x] **2.2 Warmup & singleton pipeline**
- [x] **2.3 VRAM & memory management** (`vram_manager.py`)
- [x] **2.4 VRAM cleanup on cancel/timeout** (signal `task_revoked` & `finally` block trong task)
- [x] **2.9 Cache correctness** (hash version qua `.env`, hiện tại `v6`; băm sâu đủ 13 tham số)
- [x] **2.10 Determinism & seed policy**
    - [x] `seed=-1` sinh ngẫu nhiên trong worker (`secrets.randbelow`)
    - [x] Trả `seed` thực trong `GetTaskStatus` response (`response.seed = int(result.get("seed", -1))`)
- [x] **2.10b Mac performance profiles**
    Profile Mac đang dùng: `dreamshaper-8` (SD 1.5, non-turbo), 512×512, **20 steps**, DPM++ 2M Karras scheduler, CFG=7.0, LoRA off, offload off (~80-100s/panel khi RAM rảnh).

### 2B — Nhất quán nhân vật & Bố cục đa chủ thể

- [~] **2.11 Character Consistency (IP-Adapter)** — code xong, tạm khóa do giới hạn RAM Mac
    - [x] `reference_image_url` trong proto & task payload
    - [x] Panel 1 sinh từ text; orchestrator lưu URL panel 1 làm reference cho panel 2–4
    - [x] Tích hợp **IP-Adapter** (`h94/IP-Adapter`) — ảnh trắng placeholder + scale=0 khi rỗng
    - [x] *Mac:* đã benchmark — **~745s/panel**, tắt mặc định qua `IMAGE_AI_IP_ADAPTER_ENABLED=false`
    - [ ] Cache embedding theo `character_id` trong Redis (TTL)
- [x] **2.11b Regional Diffusion (2 nhân vật / 1 khung)**
    - [x] Nghiên cứu & implement trong `src/core/regional_generator.py`
    - [x] Phương pháp **Physical crop-tiling + linear feather blending** cho SDXL latents
    - [x] Tránh hợp thể identity (rùa dính thỏ) & giữ đúng vị trí trái/phải

### 2C — Thoại & layout truyện (Giai đoạn B)

- [x] **2.7 Pillow caption cơ bản** (1 bubble đáy, controlled via `IMAGE_AI_CAPTION_RENDER_ENABLED`)
- [x] **2.8 Comic postprocessing & robustness**
    - [x] Font cache (`@lru_cache` trong `image_processing.py`)
    - [x] Enhance màu & độ nét (`enhance_comic_image`: UnsharpMask + Color/Contrast boost)
- [ ] **2.12 Dynamic Speech Bubbles**
    - [x] Schema `SpeechBuble` trong proto (`text`, `bubble_type`, `x_pos`, `y_pos`)
    - [ ] Render bong bóng thoại động theo tọa độ normalized (nếu FE không render)
- [ ] **2.15 Page Panel Layout Assembler (2×2)**
    - [ ] Module ghép 4 panel → trang truyện 2×2 với viền gutter
    - [ ] Hoặc để Orchestrator-AI đảm nhận ghép trang từ 4 presigned URL

---

## Phase 3: Storage & Caching

- [x] **3.1 MinIO upload trực tiếp từ RAM + presigned URL**
- [x] **3.3 Redis cache MD5**
- [x] **3.4 Thundering herd lock** (`acquire_generation_lock` với retry loop)
- [x] **3.5 Cache versioning** (`IMAGE_AI_CACHE_KEY_VERSION=v6`)
- [x] **3.6 Celery result vs cache thống nhất** (cùng schema dict kết quả)
- [ ] **3.2 MinIO reliability** (retry logic với exponential backoff)

---

## Phase 4: gRPC & Celery

- [x] **4.1–4.5 Async generate, status polling, cancel, request validation, macOS solo pool**
- [x] **4.6 Worker reliability & Soft time limit handling** (`SoftTimeLimitExceeded`)
- [x] **4.10 GPU/CPU health & cache clear RPC** (`CheckGpuHealth`, `CheckCpuHealth`, `ClearGpuCache`)
- [ ] **4.7 Backpressure / rate limit**
- [ ] **4.8 Correlation ID trong log**

---

## Phase 5: Monitoring & Observability

- [x] **5.1 Prometheus Metrics** (`port 9107`): `image_requests_total`, `cache_hit_total`, `cache_miss_total`, `task_duration_seconds`, `active_gpu_tasks`, `safety_blocks_total`, `minio_upload_errors_total`.
- [x] **5.2 Health Monitor `/healthz`**: HTTP port 8000, kiểm tra Redis worker flag `image_ai:worker_ready`.
- [ ] **5.1c GPU CUDA Metrics (`pynvml`)** — Cho Giai đoạn C Cloud.

---

## Phase 6: Testing & Documentation

- [x] **6.1 E2E Test script** (`tests/test_client.py`)
- [x] **6.2 Regional diffusion test script** (`tests/test_regional.py`)
- [x] **6.5 Documentation**: `README.md`, `docs/production_guide.md`, `docs/TODO.md`.

---

## Phase 7: GPU Cloud (Giai đoạn C)

- [x] **7.0 Fix Dockerfile protobuf path**: Sửa `--python_out=./src/service/generated` khớp với `scripts/generate_proto.sh`.
- [ ] **7.1 Docker production** (multi-stage, non-root, HEALTHCHECK)
- [ ] **7.2 GPU runtime** (Bật `nvidia` runtime trong `docker-compose.yml`, deploy CUDA worker)

---

## Thứ tự làm tiếp (cập nhật 2026-07)

1. ~~IP-Adapter + reference từ panel 1~~ — **xong**, tạm khóa do RAM Mac.
2. ~~Regional Diffusion 2 nhân vật~~ — **xong** (`src/core/regional_generator.py`).
3. Chuyển qua **orchestrator-ai**: tích hợp gRPC client gọi image-ai thật với 4 panel tuần tự.
4. **2.15 / Ghép trang 2×2**: chốt làm tại orchestrator-ai hay thêm API batch vào image-ai.
5. **7.2 GPU Cloud**: Bật lại IP-Adapter + benchmark cho thesis khi có GPU CUDA server.
