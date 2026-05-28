from celery import Celery
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
