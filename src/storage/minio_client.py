import os
import io
from minio import Minio
from datetime import timedelta
from PIL import Image
from logger.config import get_logger

logger = get_logger(__name__)

class MinioStorageClient:
    def __init__(self):
        # Đọc cấu hình từ Env
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = os.getenv("MINIO_SECURE", "False").lower() == "true"
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME", "comic-images")
        
        self.client = None

    def initialize_client(self):
        """Khởi tạo MinIO Connection và tự động tạo Bucket nếu chưa tồn tại."""
        if self.client is not None:
            return

        logger.info(f"Đang thiết lập kết nối tới MinIO tại: {self.endpoint}...")
        try:
            self.client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )

            # Tự động tạo Bucket chứa ảnh của truyện
            if not self.client.bucket_exists(self.bucket_name):
                logger.info(f"Bucket '{self.bucket_name}' chưa tồn tại. Đang tiến hành khởi tạo mới...")
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Đã khởi tạo Bucket '{self.bucket_name}' thành công!")
            
        except Exception as e:
            logger.error(f"Khởi tạo MinIO Client thất bại: {str(e)}")
            raise e

    def upload_image(self, image: Image.Image, filename: str) -> str:
        """
        Nghiệp vụ tải ảnh nhị phân trực tiếp từ RAM (io.BytesIO) lên MinIO, 
        giúp tăng tốc hệ thống và tránh đọc ghi ổ đĩa vật lý của Server.
        """
        self.initialize_client()
        logger.info(f"Đang chuẩn bị tải ảnh lên MinIO Bucket '{self.bucket_name}' với tên: {filename}...")

        try:
            # Chuyển đổi định dạng PIL Image sang byte stream nhị phân
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            # Gửi nhị phân lên MinIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=filename,
                data=img_byte_arr,
                length=len(img_byte_arr.getvalue()),
                content_type='image/jpeg'
            )
            logger.info(f"Đã upload thành công ảnh {filename} lên MinIO!")
            
            # Sinh link Presigned URL để client lấy hiển thị trực tiếp (hiệu lực trong 7 ngày)
            presigned_url = self.get_presigned_url(filename)
            return presigned_url
            
        except Exception as e:
            logger.error(f"Tải ảnh lên MinIO thất bại: {str(e)}")
            raise e

    def get_presigned_url(self, filename: str, expires_days: int = 7) -> str:
        """Sinh đường dẫn an toàn (Presigned URL) có giới hạn thời gian truy cập."""
        self.initialize_client()
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=filename,
                expires=timedelta(days=expires_days)
            )
            return url
        except Exception as e:
            logger.error(f"Lấy link Presigned URL thất bại cho file {filename}: {str(e)}")
            raise e

minio_storage_client = MinioStorageClient()
