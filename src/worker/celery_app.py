import os
from celery import Celery

# Đọc cấu hình hàng đợi từ Env
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Khởi tạo Celery Application cho Worker chạy GPU sinh ảnh
celery_app = Celery(
    "image_ai_worker",
    broker=broker_url,
    backend=backend_url,
    include=["worker.tasks"] # Khai báo danh sách các file chứa task
)

# Cấu hình Celery tối ưu hóa
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    
    # Rất quan trọng cho GPU Worker: Chỉ lấy 1 task tại một thời điểm
    # Tránh nạp đè nhiều tác vụ sinh ảnh làm sập GPU
    worker_prefetch_multiplier=1
)
