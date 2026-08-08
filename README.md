# image-ai — Dịch vụ sinh ảnh (Hướng dẫn tiếng Việt, chi tiết)

`image-ai` là microservice chịu trách nhiệm sinh ảnh cho ComicSystem. Nó
chạy một gRPC server để nhận job từ `orchestrator-ai` và có một worker Celery
để thực thi các tác vụ dùng mô hình (diffusers / PyTorch). File này mô tả
đầy đủ các bước cài đặt, cấu hình `.env`, biên dịch proto, chạy test và chạy
service trên local.

---

**Yêu cầu cơ bản**

- Python 3.10+ (khuyến nghị 3.12 trên mac nếu bạn có)
- CUDA/GPU khi chạy mô hình lớn (nếu dùng local GPU)
- Docker (để chạy Redis, MinIO) — Docker Compose khuyến nghị

**Lưu ý về phụ thuộc (requirements.txt)**

- Repo đã cố định một số phiên bản (ví dụ torch/diffusers) để tránh xung
  đột runtime. Khi cài, dùng file `requirements.txt` có sẵn; không ép lên bản
  mới hơn nếu không chắc vì có thể gây lỗi import (ví dụ `diffusers` mới
  có thể gọi tới API torch khác).

---

## 1. Cấu hình môi trường

Từ thư mục `image-ai`:

```bash
cd image-ai
cp .env.example .env
# Mở image-ai/.env và thêm HF_TOKEN=hf_xxx... nếu bạn dùng các model trên HF
# (không commit .env vào git)
```

Ghi chú quan trọng: phiên bản code hiện tại sẽ tự động đọc `.env` và đưa
`HF_TOKEN` vào `os.environ` khi module settings được import, vì vậy các thư
viện khác (SDK HF, etc.) có thể thấy token mà không cần `export HF_TOKEN` thủ
công trước khi chạy dịch vụ.

---

## 2. Cài đặt virtualenv và phụ thuộc

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Ghi chú: `pydantic-settings` đã có trong `requirements.txt`, nên bạn không cần
chạy `pip install pydantic-settings` riêng.

---

## 3. Chạy hạ tầng phụ trợ

```bash
docker compose up -d redis minio
```

---

## 4. Biên dịch protobuf và chạy test nhanh (sanity check)

Script `./scripts/generate_proto.sh` trong `image-ai` sẽ biên dịch các `.proto`
và sửa import path cho phù hợp với package. Script ưu tiên dùng `env/bin/python3`
nếu venv tên `env` tồn tại.

```bash
./scripts/generate_proto.sh
./env/bin/python -m pytest -q tests/test_settings_env.py
```

Test `tests/test_settings_env.py` xác nhận rằng `HF_TOKEN` (nếu có trong
`.env`) được đưa vào `os.environ` khi module settings được load/reload — giúp
đảm bảo thư viện gọi tới HF có thể dùng token ngay cả khi không `export` thủ
công.

---

## 5. Khởi chạy service

Khởi gRPC server (process 1):

```bash
# từ thư mục image-ai
source env/bin/activate
python src/server.py
```

Khởi Celery worker (một shell riêng):

```bash
cd image-ai
source env/bin/activate
cd src
celery -A worker.celery_app worker --loglevel=info --concurrency=1
```

Ghi chú: khi bạn chỉnh `.env` (ví dụ thay `HF_TOKEN`), **phải khởi lại Celery
worker** vì các mô hình và adapter thường được load vào bộ nhớ khi worker
khoi động lần đầu.

---

## 6. Các biến môi trường quan trọng

- `HF_TOKEN` — (nếu dùng model trên Hugging Face); để trong `image-ai/.env`.
- `IMAGE_AI_MODEL_ID` — id mô hình (mặc định trong `.env.example`).
- `IMAGE_AI_IP_ADAPTER_ENABLED` — bật/tắt IP-Adapter (tham chiếu ảnh để giữ
  tính nhất quán character)
- MinIO/Redis configs: `MINIO_*`, `REDIS_URL` (xem `.env.example`)

---

## 7. Gợi ý vận hành và tối ưu

- IP-Adapter giúp giữ nhân vật nhất quán nhưng tốn resource; bật khi có GPU
  cloud hoặc máy có bộ nhớ lớn.
- Trên mac dev không có GPU mạnh, tắt các chế độ GPU-intense hoặc dùng cloud
  để test model.

---

## 8. Troubleshooting

- Lỗi import torch/diffusers: kiểm tra `requirements.txt` và phiên bản torch.
- Nếu không thấy `HF_TOKEN` ở runtime: đảm bảo `.env` tồn tại và chạy test
  `tests/test_settings_env.py` để xác nhận. Không cần `export HF_TOKEN` nếu
  settings module đã load `.env` đúng cách.

---

## 9. Useful commands (tóm tắt)

```bash
# Tạo venv & cài
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Biên dịch proto
./scripts/generate_proto.sh

# Test settings (kiểm HF_TOKEN được nạp)
./env/bin/python -m pytest -q tests/test_settings_env.py

# Chạy server
python src/server.py

# Chạy celery worker (tách shell)
cd src
celery -A worker.celery_app worker --loglevel=info --concurrency=1
```

Nếu bạn muốn, tôi có thể thêm hướng dẫn cấu hình production (systemd, Docker
image) hoặc script khởi toàn bộ stack dev (start/stop cho be-comic, story-ai,
image-ai, orchestrator-ai). Bạn muốn tôi tiếp tục với phần nào?
