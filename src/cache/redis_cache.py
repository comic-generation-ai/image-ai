import hashlib
import json
import re
import time
import redis
from config.settings import get_settings
from logger.config import get_logger

logger = get_logger(__name__)

class RedisCacheManager:
    def __init__(self):
        self.settings = get_settings()
        self.redis_url = self.settings.REDIS_URL
        self.client = None

    def initialize_client(self):
        """Khởi tạo kết nối tới Redis."""
        if self.client is not None:
            return
        
        logger.info(f"Đang kết nối tới Redis cache tại: {self.redis_url}...")
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Kết nối thành công tới Redis!")
        except Exception as e:
            logger.error(f"Khởi tạo Redis Client thất bại: {str(e)}")
            raise e

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = re.sub(r"\s+", " ", (value or "").strip())
        return normalized

    def generate_hash_key(
        self,
        prompt: str,
        seed: int,
        caption_text: str,
        width: int,
        height: int,
        steps: int,
        model_id: str,
        guidance_scale: float = 2.0,
        lora_id: str = "",
        output_image_format: str = "",
        jpeg_quality: int = 0,
        png_compress_level: int = 0,
    ) -> str:
        """
        Tạo mã khóa MD5 từ tất cả các tham số ảnh để kiểm tra trùng lặp prompt đầu vào.
        """
        payload = {
            "v": self.settings.CACHE_KEY_VERSION,
            "prompt": self._normalize_text(prompt),
            "seed": seed,
            "caption_text": self._normalize_text(caption_text),
            "width": width,
            "height": height,
            "steps": steps,
            "model_id": model_id,
            "guidance_scale": guidance_scale,
            "lora_id": lora_id or "",
            "output_image_format": output_image_format,
            "jpeg_quality": jpeg_quality,
            "png_compress_level": png_compress_level,
        }
        raw_string = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return hashlib.md5(raw_string.encode("utf-8")).hexdigest()

    def get_cached_image_url(self, hash_key: str) -> str:
        """Kiểm tra xem prompt trùng lặp đã được vẽ chưa."""
        self.initialize_client()
        try:
            url = self.client.get(f"img_cache:{hash_key}")
            if url:
                logger.info(f"PHÁT HIỆN TRÙNG LẶP PROMPT (Cache Hit)! Đang lấy lại kết quả cũ trong Redis...")
                return url
            return None
        except Exception as e:
            logger.error(f"Đọc dữ liệu từ Redis thất bại: {str(e)}")
            return None

    def set_cached_image_url(self, hash_key: str, url: str):
        """Lưu đường dẫn ảnh vào cache để tái sử dụng."""
        self.initialize_client()
        try:
            self.client.setex(
                name=f"img_cache:{hash_key}",
                time=self.settings.REDIS_CACHE_TTL_SECONDS,
                value=url
            )
            logger.info(f"Đã lưu kết quả sinh ảnh vào Redis cache với key: img_cache:{hash_key}")
        except Exception as e:
            logger.error(f"Ghi dữ liệu vào Redis thất bại: {str(e)}")

    def acquire_generation_lock(self, hash_key: str, ttl_seconds: int = 120) -> bool:
        self.initialize_client()
        lock_key = f"img_lock:{hash_key}"
        try:
            return bool(self.client.set(lock_key, str(time.time()), nx=True, ex=ttl_seconds))
        except Exception as e:
            logger.error(f"Không thể lấy lock cache key {hash_key}: {str(e)}")
            return False

    def release_generation_lock(self, hash_key: str):
        self.initialize_client()
        lock_key = f"img_lock:{hash_key}"
        try:
            self.client.delete(lock_key)
        except Exception as e:
            logger.warning(f"Không thể release lock {lock_key}: {str(e)}")

redis_cache_manager = RedisCacheManager()
