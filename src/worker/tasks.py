import uuid
import time
from worker.celery_app import celery_app
from core.pipeline_runner import pipeline_runner
from core.vram_manager import vram_manager
from core.safety_checker import safety_checker
from utils.image_processing import add_caption_to_comic
from storage.minio_client import minio_storage_client
from cache.redis_cache import redis_cache_manager
from logger.config import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, name="worker.tasks.generate_image_task")
def generate_image_task(self, prompt: str, width: int, height: int, seed: int, steps: int, caption_text: str):
    """
    Nhiệm vụ Celery chạy ngầm chính:
    Thực hiện điều phối từ A-Z quy trình sinh ảnh và hậu kỳ.
    """
    task_id = self.request.id
    logger.info(f"--- BẮT ĐẦU XỬ LÝ TASK SINH ẢNH (ID: {task_id}) ---")
    
    # Lấy thời gian bắt đầu đo hiệu năng
    start_time = time.time()

    # Bước 1: Tính mã khóa Hash Key để check cache
    hash_key = redis_cache_manager.generate_hash_key(
        prompt=prompt,
        seed=seed,
        caption_text=caption_text,
        width=width,
        height=height
    )
    
    # Bước 2: Kiểm tra cache trong Redis
    cached_url = redis_cache_manager.get_cached_image_url(hash_key)
    if cached_url:
        logger.info(f"Task {task_id} hoàn thành ngay lập tức nhờ Cache Hit!")
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "minio_url": cached_url,
            "cached": True,
            "duration_seconds": round(time.time() - start_time, 2)
        }

    try:
        # Bước 3: Sinh ảnh thô bằng GPU PyTorch
        # Note: Do Celery chạy đồng bộ bên trong Worker process, 
        # concurrrency=1 đảm bảo tuần tự hóa không cần asyncio.Lock vật lý.
        logger.info(f"Thông tin VRAM GPU trước khi chạy: {vram_manager.get_gpu_memory_info()}")
        
        raw_image = pipeline_runner.generate(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps
        )
        
        logger.info(f"Thông tin VRAM GPU sau khi chạy: {vram_manager.get_gpu_memory_info()}")

        # Bước 4: Kiểm duyệt an toàn hình ảnh (NSFW Filter)
        if not safety_checker.check_image(raw_image):
            raise ValueError("Bức ảnh không vượt qua bài kiểm duyệt nội dung an toàn (NSFW)!")

        # Bước 5: Hậu kỳ Pillow chèn khung truyện và lời thoại tiếng Việt
        processed_image = add_caption_to_comic(
            image=raw_image,
            text=caption_text
        )

        # Bước 6: Tải ảnh lên MinIO Object Storage
        filename = f"comic_{uuid.uuid4().hex}.jpg"
        presigned_url = minio_storage_client.upload_image(
            image=processed_image,
            filename=filename
        )

        # Bước 7: Lưu trữ vào Redis Cache để tái sử dụng
        redis_cache_manager.set_cached_image_url(hash_key, presigned_url)

        # Giải phóng CUDA Cache
        vram_manager.clear_cache()

        duration = round(time.time() - start_time, 2)
        logger.info(f"--- HOÀN THÀNH TASK SINH ẢNH {task_id} TRONG {duration} GIÂY ---")

        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "minio_url": presigned_url,
            "cached": False,
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Thực hiện Task {task_id} thất bại: {str(e)}")
        # Đảm bảo dọn dẹp GPU kể cả khi lỗi
        vram_manager.clear_cache()
        return {
            "task_id": task_id,
            "status": "FAILED",
            "error_message": str(e),
            "duration_seconds": round(time.time() - start_time, 2)
        }
