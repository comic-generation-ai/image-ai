import hashlib
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

    def generate_hash_key(self, prompt: str, seed: int, caption_text: str, width: int, height: int) -> str:
        """
        Tạo mã khóa MD5 từ tất cả các tham số ảnh để kiểm tra trùng lặp prompt đầu vào.
        """
        raw_string = f"{prompt}_{seed}_{caption_text}_{width}_{height}"
        return hashlib.md5(raw_string.encode('utf-8')).hexdigest()

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

    def set_cached_image_url(self, hash_key: str, url: str, ttl_days: int = 14):
        """Lưu đường dẫn ảnh vào cache để tái sử dụng."""
        self.initialize_client()
        try:
            # Lưu trữ cache với thời gian hết hạn (mặc định 14 ngày)
            ttl_seconds = ttl_days * 24 * 60 * 60
            self.client.setex(
                name=f"img_cache:{hash_key}",
                time=ttl_seconds,
                value=url
            )
            logger.info(f"Đã lưu kết quả sinh ảnh vào Redis cache với key: img_cache:{hash_key}")
        except Exception as e:
            logger.error(f"Ghi dữ liệu vào Redis thất bại: {str(e)}")

redis_cache_manager = RedisCacheManager()
