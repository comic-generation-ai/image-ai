import os

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
)

# api-server (gRPC/FastAPI, server.py) và celery-worker chạy ở 2 tiến trình
# tách biệt (xem docker-compose.yml) — mọi Counter/Histogram/Gauge dưới đây
# thực chất được inc()/observe() bên trong tiến trình Celery worker (worker/
# tasks.py), trong khi /metrics lại được FastAPI expose ở tiến trình api-server.
# Nếu không bật multiprocess mode, mỗi tiến trình giữ giá trị metric riêng
# trong bộ nhớ và /metrics sẽ luôn trả về 0 dù worker đã xử lý hàng nghìn tác
# vụ. PROMETHEUS_MULTIPROC_DIR phải trỏ tới một thư mục dùng chung giữa các
# tiến trình (volume chung nếu chạy nhiều container) để prometheus_client tự
# chuyển sang lưu giá trị dạng file thay vì bộ nhớ tiến trình.
_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

image_requests_total = Counter(
    "image_requests_total",
    "Total image generation requests",
    ["status"],
)

cache_hit_total = Counter(
    "cache_hit_total",
    "Number of Redis cache hits",
)

cache_miss_total = Counter(
    "cache_miss_total",
    "Number of Redis cache misses",
)

task_duration_seconds = Histogram(
    "task_duration_seconds",
    "Task duration in seconds",
    ["cached", "success"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120, 300),
)

minio_upload_errors_total = Counter(
    "minio_upload_errors_total",
    "Total MinIO upload failures",
)

safety_blocks_total = Counter(
    "safety_blocks_total",
    "Total blocked NSFW images",
)

# multiprocess_mode="livesum": cộng dồn giá trị từ các tiến trình worker con
# đang sống tại thời điểm scrape. Mode mặc định "all" phơi bày riêng từng pid
# (đúng cho counter) nhưng sai ngữ nghĩa cho một gauge đếm "đang chạy bao nhiêu
# tác vụ" — "livesum" mới cho ra đúng tổng hiện tại và tự loại pid đã chết.
active_gpu_tasks = Gauge(
    "active_gpu_tasks",
    "Current active GPU generation tasks",
    multiprocess_mode="livesum",
)


def prometheus_response():
    if _MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=_MULTIPROC_DIR)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
