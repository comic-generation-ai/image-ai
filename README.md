# Image AI — Comic System Generation Service

Dịch vụ sinh ảnh minh hoạ truyện tranh bằng Stable Diffusion. Giao tiếp qua gRPC, xử lý bất đồng bộ
qua Celery để tránh quá tải GPU/VRAM khi nhiều request cùng lúc.

**Nguyên tắc cốt lõi:** 1 lần gọi `GenerateImageAsync` = 1 khung tranh (panel) duy nhất. Orchestrator
gọi lặp lại (thường 4 lần) rồi tự ghép thành trang truyện — image-ai không tự vẽ nhiều khung trong 1 ảnh.

---

## Tóm tắt cấu hình đang chạy

| | |
|---|---|
| **Model** | `Lykon/dreamshaper-8` (SD 1.5, không phải turbo) — 512×512, 20 steps, DPM++ 2M Karras, CFG=7.0 |
| **Giao tiếp** | gRPC cho toàn bộ business logic; REST (FastAPI) chỉ phục vụ `/healthz` và `/metrics` |
| **Hàng đợi** | Celery + Redis, `concurrency=1` — xử lý GPU tuần tự, không OOM |
| **Lưu trữ** | MinIO (presigned URL) + Redis (cache kết quả theo hash prompt/seed/size/...) |
| **Nhất quán nhân vật** | Đã code (IP-Adapter) nhưng **đang tắt** — xem [mục riêng bên dưới](#tính-năng-đã-code-nhưng-đang-tắt) |
| **Phần cứng dev** | Mac Apple Silicon (MPS) — xem tối ưu riêng bên dưới |

Đổi model qua `IMAGE_AI_MODEL_ID` trong `.env` sang `stabilityai/sd-turbo` (nhẹ hơn, 4-8 steps) hoặc
`Lykon/dreamshaper-xl-v2-turbo` (SDXL, cần máy khoẻ hơn) — xem comment trong `.env`.

---

## Tính năng chính

### Sinh ảnh & phong cách
- **5 style preset dựng sẵn**: `storybook`, `anime`, `manga`, `retro`, `american_comic` — mỗi preset
  tự thêm suffix + negative prompt phù hợp. Chọn qua field `style` trong request, hoặc tag
  `[style:xxx]` ngay trong prompt.
- **Prompt engineering chống lỗi**: tự loại cú pháp Midjourney (`--ar 16:9`,...) vì CLIP không hiểu;
  ưu tiên giữ nguyên style suffix khi prompt đầu vào dài (cắt bớt phần mô tả cảnh thay vì làm rơi mất
  suffix) — quan trọng khi prompt đến từ story-ai (thường dài hơn giới hạn CLIP ~77 token).
- **Hậu kỳ Pillow**: chèn caption tiếng Việt có dấu vào khung ảnh (viền đen + hộp thoại mờ, tự xuống
  dòng), tăng nhẹ độ nét/màu (`enhance_comic_image`).
- **Output validation & safety**: chặn ảnh đen/hỏng trước khi upload; lọc NSFW bằng
  `Falconsai/nsfw_image_detection`.

### Vận hành & hạ tầng
- **gRPC async**: `GenerateImageAsync` trả `task_id` ngay lập tức, `GetTaskStatus` để poll,
  `CancelTask`, `CheckHealth`, `CheckGpuHealth`, `CheckCpuHealth`, `ClearGpuCache`.
- **Celery tuần tự hoá GPU**: `concurrency=1` + `worker_prefetch_multiplier=1` — loại bỏ hoàn toàn
  lỗi tràn VRAM khi nhiều request tới cùng lúc.
- **Cache chống trùng lặp**: băm MD5 toàn bộ tham số ảnh hưởng output (prompt, seed, size, steps,
  model, style, reference image,...); có lock chống thundering herd khi nhiều request cùng cache-miss
  một hash key.
- **MinIO trực tiếp từ RAM**: ảnh không ghi ra đĩa, upload thẳng từ `BytesIO` lên MinIO, trả về
  presigned URL.
- **Health check phản ánh đúng worker thật**: gRPC server và Celery worker là 2 process riêng biệt —
  `/healthz` đọc cờ `image_ai:worker_ready` qua Redis (worker tự ghi sau khi warmup xong), không check
  nhầm biến cục bộ của process server.

### Tối ưu Apple Silicon (Mac dev)
- Decode VAE trên CPU (float32) + generator trên CPU — tránh NaN/ảnh đen khi UNet chạy fp16 trên MPS.
- Sequential CPU offload có sẵn trong code nhưng **tắt mặc định** — Dreamshaper 8 (SD1.5) nhẹ hơn
  SDXL, không cần offload; chỉ bật lại nếu đổi sang model SDXL trên máy 8–16GB RAM.
- **Lưu ý vận hành quan trọng**: tốc độ sinh ảnh phụ thuộc nhiều vào RAM còn trống lúc chạy — đo thật
  cho thấy chậm gấp ~10 lần nếu máy gần cạn RAM (Chrome/IDE mở nhiều). Luôn đóng bớt app trước khi
  benchmark hoặc demo.

---

## Tính năng đã code nhưng đang tắt

### IP-Adapter — nhất quán nhân vật giữa các panel

Cơ chế: orchestrator truyền `reference_image_url` (ảnh panel đầu tiên) cho các panel sau; image-ai tải
ảnh đó và dùng [`h94/IP-Adapter`](https://huggingface.co/h94/IP-Adapter) để điều kiện hoá sinh ảnh,
giữ đặc điểm nhân vật (khuôn mặt, trang phục, màu sắc) xuyên suốt trang truyện.

Code đã hoàn thiện và test bằng ảnh thật (bao gồm cả trường hợp không có reference — dùng ảnh trắng
placeholder + scale=0 để tránh crash theo yêu cầu của `diffusers`). Bật qua 2 biến trong `.env`:

```bash
IMAGE_AI_IP_ADAPTER_ENABLED=true
IMAGE_AI_IP_ADAPTER_SCALE=0.6   # 0.5-0.7 hợp lý; cao quá thì panel sau gần như copy y hệt panel đầu
```

**Vì sao đang tắt:** đo thật trên Mac 8GB, IP-Adapter làm chậm gấp **~10 lần** (từ ~80-100s lên
~745s/panel) do CLIP vision encoder thêm ăn RAM, đẩy máy vào swap-thrashing. Không khả thi cho demo
trên máy hiện tại. Bật lại khi chuyển sang GPU cloud — chỉ cần đổi `.env`, không cần sửa code.

---

## Cấu trúc dự án
*   `proto/`: Định nghĩa API giao tiếp gRPC (`.proto`).
*   `scripts/`: Script tự động biên dịch Protobuf sang Python code (`generate_proto.sh`).
*   `src/config/`: Cấu hình tập trung bằng Pydantic Settings (`IMAGE_AI_*` trong `.env`).
*   `src/core/`: Trái tim AI — pipeline diffusion, IP-Adapter, LoRA loader, VRAM manager, NSFW filter.
*   `src/utils/`: Xử lý ảnh (chèn caption, hậu kỳ sharpen/color, validate ảnh đen/hỏng).
*   `src/worker/`: Celery App + Task chính (`generate_image_task`) chạy ngầm trên GPU.
*   `src/service/`: gRPC servicer (`ImageGenerationService`) + code sinh ra từ proto (`service/generated/`).
*   `src/storage/` & `src/cache/`: Xử lý MinIO và Redis Cache.
*   `src/logger/` & `src/metrics/`: Logging chuẩn hóa + Prometheus metrics.
*   `src/server.py`: Điểm khởi chạy song song gRPC và FastAPI Health Monitor.
*   `docs/`: [`production_guide.md`](docs/production_guide.md) (kiến trúc production) và [`TODO.md`](docs/TODO.md) (roadmap chi tiết + benchmark).
*   `tests/test_client.py`: Script test thủ công E2E (health, gRPC generate, cancel, cache hit).

---

## Cài đặt & chạy local

### 1. Chuẩn bị môi trường Python ảo
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Tạo file cấu hình `.env`
`.env` (không commit git) đang có sẵn, cấu hình chạy `Lykon/dreamshaper-8` như bảng tóm tắt ở trên.
Nếu cần tạo lại từ template:
```bash
cp .env.example .env
```
`.env.example` gợi ý mặc định khác (SDXL Turbo hoặc sd-turbo) — đổi `IMAGE_AI_MODEL_ID` tùy máy. Trên
Mac, giữ nguyên các biến `IMAGE_AI_MPS_*` để tránh ảnh đen/NaN khi decode.

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

### 6. Khởi chạy Celery Worker (cần GPU/MPS để chạy Stable Diffusion)
```bash
cd src && celery -A worker.celery_app worker --loglevel=info --concurrency=1
```
> Celery cần chạy từ thư mục `src/` để import `worker.*`/`config.*` đúng đường dẫn. Model/LoRA/
> IP-Adapter chỉ được nạp **1 lần lúc worker start** — sau khi sửa `.env` phải **restart worker**.

### 7. Test nhanh E2E (health + gRPC generate + cancel + cache hit)
```bash
python tests/test_client.py
```

---

## Production / Docker

`docker-compose.yml` hiện chỉ bật `redis` + `minio` (hạ tầng dev). Service `api-server` (gRPC) và
`celery-worker` (GPU) đang bị **comment** — chưa bật GPU runtime (`nvidia`), xem `docs/TODO.md` Phase 7.
Tới lúc đó, chạy gRPC server + Celery worker **local** theo các bước ở trên, chỉ dùng Docker cho hạ tầng:
```bash
docker-compose up -d redis minio
```
Khi Phase 7 hoàn tất (bật lại 2 service + sửa path proto trong `Dockerfile`), `docker-compose up --build`
sẽ dựng toàn bộ stack trong 1 lệnh.
