import os

from celery import Celery
from celery.signals import celeryd_init, worker_process_shutdown

from config.settings import get_settings

settings = get_settings()

# Khởi tạo Celery Application cho Worker chạy GPU sinh ảnh
celery_app = Celery(
    "image_ai_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["worker.tasks"] # Khai báo danh sách các file chứa task
)

# Cấu hình Celery tối ưu hóa
celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=settings.CELERY_ENABLE_UTC,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_track_started=True,

    # Rất quan trọng cho GPU Worker: Chỉ lấy 1 task tại một thời điểm
    # Tránh nạp đè nhiều tác vụ sinh ảnh làm sập GPU
    worker_prefetch_multiplier=1
)


@celeryd_init.connect
def _reset_prometheus_multiproc_dir(**kwargs):
    """Xoá sạch PROMETHEUS_MULTIPROC_DIR khi tiến trình worker cha khởi động,
    trước khi Celery fork các tiến trình con — tránh cộng dồn nhầm các file
    .db còn sót lại từ lần chạy trước (worker bị kill/crash không kịp dọn)."""
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_dir:
        return
    os.makedirs(multiproc_dir, exist_ok=True)
    for name in os.listdir(multiproc_dir):
        path = os.path.join(multiproc_dir, name)
        if os.path.isfile(path):
            os.remove(path)


@worker_process_shutdown.connect
def _mark_prometheus_process_dead(pid, **kwargs):
    """Dọn file metric của đúng tiến trình con vừa thoát — nếu không,
    MultiProcessCollector phía api-server vẫn đọc file .db của pid đã chết và
    cộng nhầm số liệu (đặc biệt sai với gauge multiprocess_mode="livesum")."""
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_dir:
        return
    from prometheus_client import multiprocess
    multiprocess.mark_process_dead(pid, path=multiproc_dir)
