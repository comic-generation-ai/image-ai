from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

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

active_gpu_tasks = Gauge(
    "active_gpu_tasks",
    "Current active GPU generation tasks",
)


def prometheus_response():
    return generate_latest(), CONTENT_TYPE_LATEST
