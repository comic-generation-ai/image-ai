import uuid
import time
import secrets
from celery.signals import worker_process_init, task_revoked
from celery.exceptions import SoftTimeLimitExceeded
from worker.celery_app import celery_app
from core.pipeline_runner import pipeline_runner, ImageRequest
from core.vram_manager import vram_manager
from core.safety_checker import safety_checker
from utils.image_processing import add_caption_to_comic, enhance_comic_image
from utils.image_validation import validate_generated_image
from storage.minio_client import minio_storage_client
from cache.redis_cache import redis_cache_manager
from config.settings import get_settings
from metrics import (
    image_requests_total,
    cache_hit_total,
    cache_miss_total,
    task_duration_seconds,
    minio_upload_errors_total,
    safety_blocks_total,
    active_gpu_tasks,
)
from logger.config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Cờ báo worker đã nạp xong pipeline — lưu Redis vì gRPC/FastAPI server (server.py) và
# Celery worker là 2 process riêng, không share state Python trực tiếp được.
WORKER_READY_KEY = "image_ai:worker_ready"

@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Sự kiện kích hoạt khi một Celery Worker process được khởi tạo.
    Thực hiện nạp trước model (Singleton) và Warmup hệ thống để tối ưu hóa hiệu năng.
    """
    logger.info("--- KHỞI TẠO CELERY WORKER PROCESS ---")
    logger.info("Đang tiến hành nạp trước mô hình Stable Diffusion và chạy warmup...")
    redis_cache_manager.initialize_client()
    redis_cache_manager.client.delete(WORKER_READY_KEY)
    try:
        pipeline_runner.warmup()
        redis_cache_manager.client.set(WORKER_READY_KEY, "1")
        logger.info("--- KHỞI TẠO CELERY WORKER PROCESS HOÀN TẤT ---")
    except Exception as e:
        logger.error(f"✗ Lỗi khi khởi tạo mô hình trong worker process: {e}")

@task_revoked.connect
def on_task_revoked(sender=None, request=None, **kwargs):
    """
    Sự kiện kích hoạt khi một Task bị thu hồi (cancel/revoke) từ bên ngoài.
    Đảm bảo giải phóng VRAM tức thì để không bị treo GPU.
    """
    task_id = getattr(request, "id", "Unknown")
    logger.warning(f"⚠ Task {task_id} đã bị thu hồi (revoked/cancelled) từ bên ngoài! Đang giải phóng VRAM khẩn cấp...")
    vram_manager.clear_cache()

@celery_app.task(bind=True, name="worker.tasks.generate_image_task")

def generate_image_task(self, prompt: str, width: int, height: int, seed: int, steps: int, caption_text: str, style: str = "", reference_image_url: str = ""):
    """
    Nhiệm vụ Celery chạy ngầm chính:
    Thực hiện điều phối từ A-Z quy trình sinh ảnh và hậu kỳ.
    """
    task_id = self.request.id
    logger.info(f"--- BẮT ĐẦU XỬ LÝ TASK SINH ẢNH (ID: {task_id}) ---")
    self.update_state(state="STARTED", meta={"stage": "queued_or_running"})

    # Lấy thời gian bắt đầu đo hiệu năng
    start_time = time.time()
    actual_seed = seed if seed != -1 else secrets.randbelow(2**31 - 1)

    hash_key = redis_cache_manager.generate_hash_key(
        prompt=prompt,
        seed=actual_seed,
        caption_text=caption_text if settings.CAPTION_RENDER_ENABLED else "",
        width=width,
        height=height,
        steps=steps,
        model_id=pipeline_runner.cache_signature,
        guidance_scale=pipeline_runner.default_guidance_scale,
        lora_id=pipeline_runner.lora_signature,
        output_image_format=settings.OUTPUT_IMAGE_FORMAT.lower(),
        jpeg_quality=settings.JPEG_QUALITY,
        png_compress_level=settings.PNG_COMPRESS_LEVEL,
        style=style,
        reference_image_url=reference_image_url,
    )
    
    # Kiểm tra cache trong Redis
    cached_url = redis_cache_manager.get_cached_image_url(hash_key)
    if cached_url:
        cache_hit_total.inc()
        image_requests_total.labels(status="SUCCESS").inc()
        duration = round(time.time() - start_time, 2)
        task_duration_seconds.labels(cached="true", success="true").observe(duration)
        logger.info(f"Task {task_id} hoàn thành ngay lập tức nhờ Cache Hit!")
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "minio_url": cached_url,
            "cached": True,
            "duration_seconds": duration
        }
    cache_miss_total.inc()

    # Thundering herd protection: chỉ 1 worker render cho cùng 1 hash key
    lock_acquired = redis_cache_manager.acquire_generation_lock(hash_key)
    if not lock_acquired:
        for _ in range(20):
            time.sleep(0.25)
            cached_url = redis_cache_manager.get_cached_image_url(hash_key)
            if cached_url:
                cache_hit_total.inc()
                image_requests_total.labels(status="SUCCESS").inc()
                duration = round(time.time() - start_time, 2)
                task_duration_seconds.labels(cached="true", success="true").observe(duration)
                return {
                    "task_id": task_id,
                    "status": "SUCCESS",
                    "minio_url": cached_url,
                    "cached": True,
                    "duration_seconds": duration
                }
        lock_acquired = redis_cache_manager.acquire_generation_lock(hash_key)
        if not lock_acquired:
            raise RuntimeError("Ảnh tương tự đang được sinh; vui lòng thử lại sau vài giây")

    try:
        active_gpu_tasks.inc()
        # Sinh ảnh thô bằng GPU PyTorch
        # Note: Do Celery chạy đồng bộ bên trong Worker process, 
        # concurrrency=1 đảm bảo tuần tự hóa không cần asyncio.Lock vật lý.
        logger.info(f"Thông tin VRAM GPU trước khi chạy: {vram_manager.get_gpu_memory_info()}")

        response = pipeline_runner.generate(
            ImageRequest(
                prompt=prompt,
                width=width,
                height=height,
                seed=actual_seed,
                steps=steps,
                style=style,
                reference_image_url=reference_image_url,
            )
        )
        raw_image = response.image
        validate_generated_image(raw_image)

        logger.info(f"Thông tin VRAM GPU sau khi chạy: {vram_manager.get_gpu_memory_info()}")

        # Kiểm duyệt an toàn hình ảnh (NSFW Filter)
        safety_result = safety_checker.check_image(raw_image)
        if not safety_result.is_safe:
            safety_blocks_total.inc()
            raise ValueError(
                f"Bức ảnh bị chặn bởi Safety Checker (nsfw_score={safety_result.nsfw_score:.4f})"
            )

        # Hậu kỳ Pillow: mặc định trả ảnh sạch — caption do FE render bên dưới ảnh
        # (PanelResult.caption_vi đã có sẵn trong contract orchestrator).
        # Ảnh sạch cũng là reference sạch cho IP-Adapter (hết lỗi sinh chữ rác).
        raw_image = enhance_comic_image(raw_image)
        if settings.CAPTION_RENDER_ENABLED:
            processed_image = add_caption_to_comic(image=raw_image, text=caption_text)
        else:
            processed_image = raw_image

        # Tải ảnh lên MinIO Object Storage
        image_format = settings.OUTPUT_IMAGE_FORMAT.lower()
        if image_format not in {"jpeg", "jpg", "png"}:
            raise ValueError("OUTPUT_IMAGE_FORMAT phải là 'jpeg' hoặc 'png'")

        extension = "jpg" if image_format in {"jpeg", "jpg"} else "png"
        content_type = "image/jpeg" if extension == "jpg" else "image/png"
        filename = f"comic_{uuid.uuid4().hex}.{extension}"
        presigned_url = minio_storage_client.upload_image(
            image=processed_image,
            filename=filename,
            content_type=content_type,
            jpeg_quality=settings.JPEG_QUALITY,
            png_compress_level=settings.PNG_COMPRESS_LEVEL,
        )
        if not presigned_url:
            minio_upload_errors_total.inc()
            raise ValueError("Upload ảnh lên MinIO thất bại")

        # Lưu trữ vào Redis Cache để tái sử dụng
        redis_cache_manager.set_cached_image_url(hash_key, presigned_url)

        duration = round(time.time() - start_time, 2)
        image_requests_total.labels(status="SUCCESS").inc()
        task_duration_seconds.labels(cached="false", success="true").observe(duration)
        logger.info(f"--- HOÀN THÀNH TASK SINH ẢNH {task_id} TRONG {duration} GIÂY ---")

        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "minio_url": presigned_url,
            "cached": False,
            "seed": int(response.seed),
            "duration_seconds": duration
        }

    except SoftTimeLimitExceeded as e:
        logger.error(f"⚠ Task {task_id} bị dừng đột ngột (Soft Time Limit Exceeded)!")
        image_requests_total.labels(status="CANCELLED").inc()
        task_duration_seconds.labels(cached="false", success="false").observe(
            round(time.time() - start_time, 2)
        )
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": "Celery Task Soft Time Limit Exceeded",
            "duration_seconds": round(time.time() - start_time, 2)
        }

    except Exception as e:
        logger.error(f"Thực hiện Task {task_id} thất bại: {str(e)}")
        image_requests_total.labels(status="FAILED").inc()
        task_duration_seconds.labels(cached="false", success="false").observe(
            round(time.time() - start_time, 2)
        )
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": str(e),
            "duration_seconds": round(time.time() - start_time, 2)
        }
    finally:
        active_gpu_tasks.dec()
        # MPS: giữ cache giữa các task — clear xong task kế tiếp phải cấp phát lại
        # toàn bộ buffer (đo thực tế tốn ~15-20s/task trên Mac 8GB). CUDA vẫn clear
        # để trả VRAM. Cùng điều kiện với finally trong pipeline_runner.generate.
        if pipeline_runner.device != "mps" or settings.MPS_CLEAR_CACHE_AFTER_GENERATE:
            vram_manager.clear_cache()
        if lock_acquired:
            redis_cache_manager.release_generation_lock(hash_key)
