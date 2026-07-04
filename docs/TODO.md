# Image Generation Service (image-ai) - Implementation TODO List

Đây là danh sách các tác vụ triển khai dịch vụ sinh ảnh và hậu kỳ tự động (`image-ai`) trong hệ thống ComicSystem.

Mục tiêu cuối: **trang truyện 2×2 (4 panel)**, mỗi panel có **bong bóng thoại theo nhân vật**, **nhất quán nhân vật** giữa các khung, ảnh **sắc nét không nhòe**, đủ số liệu **benchmark Mac vs GPU cloud** cho luận văn.

---

## Lộ trình triển khai (3 giai đoạn)

| Giai đoạn | Mục tiêu | Tiêu chí hoàn thành |
|-----------|----------|---------------------|
| **A — Mac ổn định** | Pipeline chạy đúng `.env`, đo được baseline, không OOM | 1 panel 512×512 ≤ 90s (sd-turbo, offload tắt); cache hit < 1s |
| **B — Tính năng truyện** | 4 panel + bubble đa nhân vật + ghép 2×2 + đồng bộ nhân vật | API `GenerateComicPage`; ảnh trang hoàn chỉnh trên MinIO |
| **C — GPU cloud (luận văn)** | Deploy worker CUDA, benchmark so sánh, demo bảo vệ | 4 panel ≤ 2 phút (1 GPU); bảng Mac vs Cloud trong thesis |

> **Lưu ý vận hành:** Sau mỗi lần sửa `.env` phải **restart Celery worker** — model/LoRA/offload được nạp **1 lần lúc worker start**, không đọc lại `.env` khi đang chạy.

---

## Benchmark đã đo (baseline — cần restart worker để cập nhật)

| Cấu hình (log 2026-07-04) | Diffusion | Tổng task | Ghi chú |
|---------------------------|-----------|-----------|---------|
| SDXL Turbo + LoRA + MPS sequential offload | ~192s (~48s/step) | **212s** | Log cũ 2026-06-06 — model đã đổi, không còn dùng |
| Cache hit (cùng prompt/seed) | 0s GPU | **0.01s** | Redis cache hoạt động đúng |
| **`.env` hiện tại**: dreamshaper-8, 20 steps, DPM++ Karras, CFG=7.0, LoRA off, IP-Adapter off | ~2-4s/step | **~80-100s** | Đo thật, RAM máy rảnh (~1.5GB free) |
| Cùng cấu hình nhưng RAM máy gần cạn (~65-100MB free) | ~30-40s/step | **~750-800s** | Swap-thrashing — không phải bug code, do máy 8GB thiếu RAM khi chạy nhiều app khác cùng lúc |
| + IP-Adapter bật (`IMAGE_AI_IP_ADAPTER_ENABLED=true`) | ~30-40s/step | **~745s** | Đo thật — CLIP vision encoder thêm ăn RAM, đẩy máy vào swap ngay cả khi RAM còn tạm ổn trước đó. **Không khả thi trên Mac 8GB**, đã tắt lại |

**Bài học vận hành quan trọng:** trên máy 8GB, tốc độ sinh ảnh phụ thuộc rất nhiều vào RAM còn trống lúc chạy (Chrome/IDE mở nhiều làm chậm gấp 10 lần dù code không đổi). Luôn đóng bớt app trước khi demo/benchmark.

---

## Trạng thái MVP hiện tại (theo code + log)

*   **Đã chạy E2E** (`test_client.py`): gRPC, Celery, Redis, MinIO, diffusion, Pillow caption, cache hit/miss.
*   **Pipeline**: hỗ trợ SD 1.x Turbo + SDXL Turbo; CUDA / MPS / CPU; warmup singleton.
*   **LoRA**: loader cơ bản có (`lora_loader.py`); chưa LoRA per-request / MinIO artifact.
*   **Safety**: `Falconsai/nsfw_image_detection` đã tích hợp (TODO doc cũ chưa cập nhật).
*   **Style**: 5 preset dựng sẵn (`storybook`, `anime`, `manga`, `retro`, `american_comic`) + tag
    `[style:xxx]` parse ngay trong prompt (`pipeline_runner.py`) — **đã xong**, chưa có trong
    roadmap gốc bên dưới.
*   **Output validation**: chặn ảnh đen/hỏng (`OUTPUT_MIN_MEAN_BRIGHTNESS`) trước khi upload MinIO
    — **đã xong** (`utils/image_validation.py`).
*   **GPU/CPU health**: `CheckGpuHealth`, `CheckCpuHealth`, `ClearGpuCache` đã implement trong
    `image_service.py` — **đã xong**, chưa có trong roadmap gốc bên dưới.
*   **Caption**: 1 bubble cố định ở đáy ảnh (`caption_text`); proto có `speech_bubbles` nhưng **chưa implement**.
*   **Trang 2×2**: **chưa có** — cần Phase B.
*   **Nhất quán nhân vật (IP-Adapter)**: **code đã xong** (`_load_reference_image`, `_blank_ip_adapter_image`,
    `set_ip_adapter_scale` động theo request, cache key + `cache_signature` đã tính cả reference/IP-Adapter
    state) và nối dây đầy đủ gRPC → Celery → pipeline. Nhưng **tắt mặc định** qua `IMAGE_AI_IP_ADAPTER_ENABLED=false`
    vì đo thật trên Mac 8GB chậm gấp ~10 lần (xem bảng benchmark). Bật lại khi có GPU cloud, không cần sửa code.
*   **Prompt engineering**: đã strip cú pháp Midjourney (`--ar 16:9`,...), ưu tiên giữ style suffix khi
    prompt dài (cắt bớt phần mô tả cảnh thay vì làm rơi suffix), bỏ LoRA trigger words khi LoRA tắt.
*   **Cancel**: có; `test_cancel_task` gửi revoke sau khi task đã chạy xong → revoke muộn (cần hardening 4.2).
    Với pool `solo`, revoke **không dừng được** task đang chạy giữa chừng (`NotImplementedError: TaskPool
    does not implement kill_job`) — task vẫn chạy hết, chỉ được đánh dấu "revoked" trong sổ sách Celery.

---

## Bug đã fix (changelog rút gọn, 2026-07)

*   **`.env` không load được khi celery chạy từ `src/`** — `Settings.model_config.env_file` dùng
    đường dẫn tương đối theo cwd; celery bắt buộc chạy từ `src/` (để import `worker.*`) nhưng `.env`
    ở project root → Settings âm thầm rơi về default cứng (từng chạy nhầm SDXL Turbo thay vì
    Dreamshaper 8 suốt nhiều lần test). Đã fix: neo `env_file` theo vị trí `settings.py`.
*   **`/healthz` `pipeline_ready` luôn `false`** — check nhầm `pipeline_runner.pipeline` của process
    server.py (không bao giờ load model, chỉ Celery worker mới load). Đã fix: đọc cờ Redis
    `image_ai:worker_ready` do worker tự ghi sau khi warmup xong.
*   **`lora_loader.validate_compatibility()` không chặn LoRA SDXL chạy trên pipeline SD1.x** — do
    2 biến `is_sdxl_pipeline`/`is_sd_pipeline` chồng lấn (SDXL pipeline luôn khớp cả 2). Đã fix.
*   **`test_client.py`**: typo `prinst`→`print` (crash `NameError`); bỏ `continue` gây busy-loop
    bỏ qua `time.sleep(2)` khi poll `PROCESSING`.
*   **Prompt quá dài từ story-ai bị cắt mất style suffix** (bao gồm cụm chặn multi-panel) — đảo
    ưu tiên: giữ nguyên suffix, cắt bớt phần mô tả cảnh; thêm strip cú pháp Midjourney (`--ar 16:9`).
*   **IP-Adapter crash `NoneType is not iterable`** khi không có `reference_image_url` — diffusers
    bắt buộc luôn truyền `ip_adapter_image` một khi đã `load_ip_adapter()`. Đã fix: dùng ảnh trắng
    placeholder + `set_ip_adapter_scale(0)` khi không có reference thật.
*   **`Dockerfile` compile proto sai path** — đã sửa khớp `scripts/generate_proto.sh`.

---

## Tiêu chí “DONE” production-ready

*   Cache key đúng mọi tham số ảnh hưởng output (`.env` hiện tại đặt `IMAGE_AI_CACHE_KEY_VERSION=v5`; mặc định code nếu không set qua `.env` là `v3`).
*   Thundering herd lock khi cache miss đồng thời (đã có).
*   Safety checker + policy FAILED rõ ràng.
*   Metrics + benchmark reproducible (Mac + Cloud).
*   Retry MinIO/Redis; không retry vô hạn khi OOM.
*   Cancel sạch VRAM; không revoke task đã SUCCESS.

---

## Phase 1: Setup & Infrastructure

- [x] **1.1 Cấu trúc & Môi trường**
- [x] **1.2 Định nghĩa gRPC** (`proto/image_generation.proto`)
- [x] **1.3 Cấu hình tập trung** (`settings.py`, `IMAGE_AI_*`)
- [x] **1.4 Production config discipline**
- [ ] **1.5 Model artifact quản lý**
    - [ ] Document profile model: **hiện tại `.env` dùng `dreamshaper-8`** (SD 1.5, Mac dev) —
        khác kế hoạch gốc `sd-turbo`; `dreamshaper-xl-v2-turbo` + LoRA vẫn là plan cho Cloud chất lượng
    - [ ] Cache HuggingFace tại `MODEL_CACHE_DIR`; hành vi offline
    - [ ] Script kiểm tra worker đang chạy model nào (log / health endpoint)

---

## Phase 2: Core AI & Image Processing

### 2A — Tốc độ & chất lượng panel đơn (Giai đoạn A — Mac)

- [x] **2.1 Stable Diffusion Pipeline** (`pipeline_runner.py`)
- [x] **2.2 Warmup & singleton pipeline**
- [x] **2.3 VRAM & memory management**
- [x] **2.4 VRAM cleanup on cancel/timeout**
- [x] **2.9 Cache correctness** (hash version qua `.env`, hiện tại `v5` — mặc định code `v3`: prompt, seed, caption, size, steps, model, guidance, lora, format, style)
- [ ] **2.10 Determinism & seed policy**
    - [x] `seed=-1` sinh ngẫu nhiên trong worker
    - [ ] Trả `seed` thực trong `GetTaskStatus` response (proto mở rộng nếu cần)
- [x] **2.10b Mac performance profiles** *(đã chốt, không còn theo plan gốc sd-turbo)*
    Profile Mac thực tế đang dùng: `dreamshaper-8` (SD 1.5, non-turbo), 512×512, **20 steps**,
    DPM++ 2M Karras scheduler, CFG=7.0, LoRA off, offload off. Khác hẳn plan gốc (`sd-turbo` 4 steps)
    — quyết định giữ non-turbo vì chất lượng, chấp nhận chậm hơn (~80-100s/panel khi RAM rảnh).
    Profile `quality` (SDXL/Flux + LoRA) vẫn để dành GPU cloud, không kỳ vọng nhanh trên Mac.
    **Restart worker sau đổi `.env`** — đã ghi rõ trong README.

### 2B — Nhất quán nhân vật (Giai đoạn B — bắt buộc cho 4 khung)

- [~] **2.11 Character Consistency** — code xong, tạm khoá do giới hạn RAM Mac
    - [x] `reference_image_url` đã có trong proto (field 1) — không cần proto mới
    - [x] Panel 1: sinh từ text (reference rỗng); orchestrator lưu URL panel 1 làm reference cho panel 2–4
        (logic đã có sẵn trong `orchestrator-ai/src/workflow/comic_job.py`)
    - [x] Tích hợp **IP-Adapter** (`h94/IP-Adapter`) vào `pipeline_runner.py` — đọc `reference_image_url`,
        tải ảnh qua `requests`, truyền `ip_adapter_image` + `set_ip_adapter_scale` động mỗi request
    - [ ] Cache embedding theo `character_id` trong Redis (TTL) để không tính lại mỗi panel — **chưa làm**,
        không cấp thiết vì tính năng đang tắt
    - [ ] Prompt template giữ `character_id` xuyên suốt 4 panel — phụ thuộc story-ai bổ sung
        `character_ids`/character bible (xem `story-ai/TASKS_FOR_NHAN.md` — hiện đang KHÔNG yêu cầu Nhân
        làm việc này, Quý tự xử lý phía orchestrator sau)
    - [x] *Mac:* đã benchmark — **~745s/panel, không khả thi**, tắt qua `IMAGE_AI_IP_ADAPTER_ENABLED=false`.
        *Cloud:* bật lại khi có GPU, chỉ cần đổi `.env`, không sửa code.

### 2C — Thoại & layout truyện (Giai đoạn B)

- [x] **2.7 Pillow caption cơ bản** (1 bubble đáy, `caption_text`)
- [~] **2.8 Comic layout robustness** *(một phần)*
    - [x] Font cache (`@lru_cache` trong `image_processing.py`)
    - [ ] Font fallback chuẩn hóa cross-platform (Mac/Linux/Docker)
    - [ ] Giới hạn độ dài + ellipsis khi tràn bubble
- [ ] **2.12 Dynamic Speech Bubbles** — **ƯU TIÊN CAO**
    - [ ] Mở rộng proto `SpeechBubble`: thêm `speaker_id`, `speaker_name` (hiển thị optional)
    - [ ] Parse `repeated speech_bubbles` trong gRPC → worker (thay/thêm `caption_text`)
    - [ ] Vẽ bubble tại `(x_pos, y_pos)` — tọa độ normalized 0..1
    - [ ] Style: `SPEECH` (oval), `THOUGHT` (cloud), `SCREAM` (spiky)
    - [ ] Tránh chồng bubble: collision detection đơn giản
    - [ ] Hash cache key bao gồm `speech_bubbles` JSON (không chỉ `caption_text`)

- [ ] **2.15 Page Panel Layout Assembler (2×2)** — **ƯU TIÊN CAO**
    - [ ] API mới: `GenerateComicPageAsync` (hoặc orchestrator gọi 4 panel rồi ghép)
    - [ ] Module `page_assembler.py`: ghép 4 panel → lưới 2×2 + gutter + viền trang
    - [ ] Kích thước panel đề xuất: 512×512 → trang ~1080×1080 (+ gutter)
    - [ ] Upload trang hoàn chỉnh lên MinIO; cache key cấp **page** (4 panel + layout)

- [ ] **2.16 Comic Page Orchestration** *(mới)*
    - [ ] Celery task `generate_comic_page_task`: 4 panel tuần tự (Mac) / song song (Cloud multi-GPU sau)
    - [ ] Panel 1 → lưu reference URL → panel 2–4 dùng IP-Adapter
    - [ ] Mỗi panel nhận `speech_bubbles[]` riêng theo nhân vật
    - [ ] Trả progress: `panel 1/4`, `2/4`, … qua task meta hoặc gRPC stream (optional)

### 2D — Chất lượng hình (không nhòe)

- [ ] **2.13 Upscaling** — **CHỈ bật trên GPU cloud**, tắt mặc định Mac
    - [ ] Real-ESRGAN 2× sau inference (optional flag)
    - [ ] Hoặc: sinh 768×768 trên cloud thay vì upscale 512
- [~] **2.5 LoRA**
    - [x] Load/unload cơ bản, format validation
    - [ ] LoRA SD 1.x cho sd-turbo (khác EldritchComicsXL chỉ SDXL)
    - [ ] LoRA per-request + cache adapter
    - [ ] Tải LoRA từ MinIO
    - [ ] *Cloud only:* EldritchComicsXL + dreamshaper-xl cho style comic đẹp nhất

- [~] **2.6 Content Safety**
    - [x] Classifier thật (`Falconsai/nsfw_image_detection`)
    - [ ] Benchmark độ chính xác; map `BLOCKED` trong proto (optional)
    - [ ] Cho phép tắt qua env khi dev Mac (đã có implicit fallback)

- [ ] **2.14 Pre-GPU text moderation** (optional)

---

## Phase 3: Storage & Caching

- [x] **3.1 MinIO upload RAM + presigned URL**
- [x] **3.3 Redis cache MD5**
- [x] **3.4 Thundering herd lock**
- [x] **3.5 Cache versioning** (đặt qua `.env`, hiện tại `v6`)
- [ ] **3.2 MinIO reliability** (retry, timeout)
- [ ] **3.6 Celery result vs cache thống nhất**
- [ ] **3.7 Page-level cache** *(mới)*: cache cả trang 2×2, không chỉ từng panel

---

## Phase 4: gRPC & Celery

- [x] **4.1–4.5** Async generate, status, cancel, validation, macOS solo pool
- [ ] **4.2b Cancel race fix** *(mới)*: revoke task đang PROCESSING; không revoke sau SUCCESS
- [ ] **4.6 Worker reliability** (retry network, OOM taxonomy)
- [ ] **4.7 Backpressure / rate limit**
- [ ] **4.8 Correlation id trong log**
- [ ] **4.9 GenerateComicPage gRPC** *(mới)* — endpoint batch 4 panel + page assembly
- [x] **4.10 GPU/CPU health & cache clear RPC** (`CheckGpuHealth`, `CheckCpuHealth`, `ClearGpuCache` — đã implement)

---

## Phase 5: Monitoring

- [x] **5.1 Prometheus** (requests, duration, cache hit/miss, safety, active_gpu_tasks)
- [x] **5.2 `/healthz`**
- [ ] **5.1b** `cache_hit_ratio` gauge; label `device=mps|cuda`
- [ ] **5.1c** GPU metrics CUDA (`pynvml`) — Giai đoạn C
- [ ] **5.3 Tracing** (optional OpenTelemetry)

---

## Phase 6: Testing & Documentation

- [x] **6.1 `test_client.py`**
- [ ] **6.2 Unit tests** (wrap_text, hash key, safety mock, minio mock)
- [ ] **6.3 Integration** (cache hit, cancel đúng timing)
- [~] **6.4 Benchmarks** — **đang làm**
    - [x] Baseline Mac SDXL+LoRA+offload: **212s/panel** (log 2026-06-06)
    - [x] Cache hit: **0.01s**
    - [ ] Sau restart: sd-turbo profile
    - [ ] 4 panel tuần tự Mac vs 1 page task
    - [ ] Cloud CUDA cùng prompt (Giai đoạn C)
- [ ] **6.5 Docs**: `API.md`, `Runbook.md` (nhấn mạnh restart worker)

---

## Phase 7: GPU Cloud (Giai đoạn C — sau khi Phase B ổn trên Mac)

- [ ] **7.0 Fix Dockerfile protobuf path** *(mới, bug)*: `Dockerfile` hiện compile proto ra
    `./src/image_generation_pb2.py` (sed `from src import image_generation_pb2`), nhưng toàn bộ
    code (`server.py`, `image_service.py`) import từ `service.generated.*` — đúng path mà
    `scripts/generate_proto.sh` dùng (`src/service/generated/`). Phải sửa `Dockerfile` dùng
    `--python_out=./src/service/generated` (và sed tương ứng) **trước khi** bật lại
    `api-server`/`celery-worker` trong `docker-compose.yml`, nếu không container sẽ lỗi import.
- [ ] **7.1 Docker production** (multi-stage, non-root, HEALTHCHECK)
- [ ] **7.2 GPU runtime**
    - [ ] Bật `nvidia` trong `docker-compose.yml`
    - [ ] Worker CUDA: sd-turbo (nhanh) hoặc SDXL+LoRA (đẹp) — 2 profile
    - [ ] Network volume cache model
- [ ] **7.3 Auto-restart** (supervisor/systemd)
- [ ] **7.4 CI/CD** (pytest CPU-only, docker build)

---

## Phase 8: Deliverables luận văn

*   Sơ đồ: Orchestrator → gRPC → Celery → 4× Panel → Page Assembler → MinIO.
*   Bảng benchmark: Mac (A) vs Cloud (C); cache hit; cold vs warm.
*   Phần **Character Consistency**: IP-Adapter + reference chain panel 1→4.
*   Phần **UX truyện**: speech bubble đa nhân vật + layout 2×2.
*   Reliability: cancel, VRAM, timeout, thundering herd.
*   Safety + Observability.

---

## Thứ tự làm tiếp (cập nhật 2026-07)

1. ~~IP-Adapter + reference từ panel 1~~ — **xong**, tạm khoá do RAM Mac (mục 2.11).
2. Chuyển qua **orchestrator-ai**: dùng gRPC client gọi image-ai thật (thay mock story),
   parse schema thật của story-ai (`panel_number`/`image_prompt`/`dialogue`, không phải
   `index`/`prompt_en`/`caption_vi` như proto gốc — xem memory `comicsystem-story-ai-schema`).
3. **2.15 + 2.16** Ghép trang 2×2 — quyết định làm trong image-ai (API `GenerateComicPage`) hay
   để orchestrator tự ghép Pillow sau khi nhận đủ 4 URL (chưa chốt).
4. **2.12** Speech bubbles theo vị trí nhân vật (đã bàn hướng asset PNG bong bóng vẽ sẵn +
   Pillow composite, chưa code).
5. **7.2** GPU cloud + bật lại IP-Adapter + benchmark thesis (sau bảo vệ).

---

## Kiến trúc mục tiêu — Trang 2×2

```mermaid
flowchart TB
    O[Orchestrator / BE Comic] -->|GenerateComicPage| G[gRPC]
    G --> Q[Celery Queue]
    Q --> P1[Panel 1: text only]
    P1 -->|save ref image| R[Redis char cache]
    P1 --> P2[Panel 2: IP-Adapter + ref]
    R --> P2
    R --> P3[Panel 3]
    R --> P4[Panel 4]
    P2 --> B1[Bubbles speaker A/B]
    P3 --> B2[Bubbles]
    P4 --> B3[Bubbles]
    B1 --> A[Page Assembler 2x2]
    B2 --> A
    B3 --> A
    A --> M[MinIO page URL]
```

**Mục tiêu thời gian:**

| Môi trường | 4 panel + ghép trang |
|------------|----------------------|
| Mac (dreamshaper-8, 20 steps), không IP-Adapter | ~5-7 phút (đo thật, RAM rảnh) |
| Mac + IP-Adapter | **~50 phút** (đo thật `1 panel = ~745s`) — **không khả thi, đã tắt** |
| GPU cloud (chưa đo — kỳ vọng, cần benchmark thật khi có) | ~1–2.5 phút |
| Cache hit (cùng trang) | < 5 giây |
