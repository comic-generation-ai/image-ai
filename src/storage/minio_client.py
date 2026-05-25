import io
import logging
from datetime import timedelta
from typing import Optional, Union, Dict, Any
from PIL import Image
from minio import Minio
from minio.error import S3Error 

from config.settings import get_settings

logger = logging.getLogger(__name__)

class MinioStorageClient:
    """MinIO client for file storage operations operating entirely in RAM."""
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[Minio] = None

    @property
    def client(self) -> Minio:
        """Get or create MinIO client."""
        if self._client is None:
            self._client = Minio(
                endpoint=self.settings.MINIO_ENDPOINT,
                access_key=self.settings.MINIO_ROOT_USER,
                secret_key=self.settings.MINIO_ROOT_PASSWORD,
                secure=self.settings.MINIO_SECURE,
            )
        return self._client
    
    def health_check(self) -> dict:
        """Check status connection to MinIO."""
        try:
            buckets = self.client.list_buckets()
            return {
                "healthy": True,
                "buckets_count": len(buckets),
                "message": "MinIO connection successful",
            }
        except Exception as e:
            logger.error(f"MinIO health check failed: {e}")
            return {"healthy": False, "error": str(e)}
        
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if bucket exists."""
        try:
            return self.client.bucket_exists(bucket_name)
        except Exception as e:
            logger.error(f"Failed to check bucket {bucket_name}: {e}")
            return False
        
    def object_exists(self, bucket_name: str, object_path: str) -> bool:
        """Check if object exists in bucket."""
        try:
            self.client.stat_object(bucket_name, object_path)
            return True
        except S3Error:
            return False
        except Exception as e:
            logger.error(f"Error checking object existence: {e}")
            return False
        
    def get_object_info(self, bucket_name: str, object_path: str) -> Optional[dict]:
        """Get object metadata."""
        try:
            stat = self.client.stat_object(bucket_name, object_path)
            return {
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            logger.error(f"Failed to get object info for {object_path}: {e}")
            return None
    
    def upload_image(
        self,
        image: Image.Image,
        filename: Optional[str] = None,
        bucket_name: Optional[str] = None,
        object_path: Optional[str] = None,
        content_type: str = "image/png",
    ) -> Optional[str]:
        """
        Upload một đối tượng PIL Image từ RAM thẳng lên MinIO (Không ghi xuống ổ cứng)
        và trả về đường dẫn Presigned URL có thời hạn.

        Args:
            image: Đối tượng PIL.Image cần upload
            filename: Tên file (tương thích ngược)
            bucket_name: Tên bucket trong MinIO (mặc định lấy từ settings nếu không truyền)
            object_path: Đường dẫn lưu file trong bucket (mặc định bằng filename nếu không truyền)
            content_type: MIME type của ảnh (mặc định image/png)

        Returns:
            str: Đường dẫn Presigned URL nếu thành công, None nếu thất bại.
        """
        try:
            # 1. Resolve parameters dynamically for backward compatibility
            final_bucket = bucket_name or self.settings.MINIO_BUCKET_NAME
            final_path = object_path or filename
            
            if not final_path:
                raise ValueError("Phải cung cấp 'filename' hoặc 'object_path' để upload ảnh.")

            # Ensure bucket exists
            if not self.bucket_exists(final_bucket):
                logger.info(f"Creating bucket: {final_bucket}")
                self.client.make_bucket(final_bucket)
                
            # 2. Convert PIL Image to Byte Stream directly in RAM
            img_byte_arr = io.BytesIO()
            # Determine storage format based on content_type
            img_format = "JPEG" if content_type in ["image/jpeg", "image/jpg"] else "PNG"
            image.save(img_byte_arr, format=img_format)
            img_byte_arr.seek(0)  # Move stream pointer to start
            file_length = img_byte_arr.getbuffer().nbytes

            # 3. Use put_object to push byte stream from RAM to MinIO Server
            self.client.put_object(
                bucket_name=final_bucket,
                object_name=final_path,
                data=img_byte_arr,
                length=file_length,
                content_type=content_type,
            )
            logger.info(f"Successfully uploaded image to {final_bucket}/{final_path} (Size: {file_length} bytes)")
            
            # 4. Generate Presigned URL with 7-day validity to send back to Orchestrator
            presigned_url = self.get_presigned_url(final_bucket, final_path)
            return presigned_url
            
        except S3Error as e:
            logger.error(f"S3 Error uploading file: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            return None
    
    def list_objects(
        self, bucket_name: str, prefix: str = "", recursive: bool = True
    ) -> list[dict]:
        """List objects in bucket with optional prefix filter."""
        try:
            objects = self.client.list_objects(
                bucket_name, prefix=prefix, recursive=recursive
            )
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "is_dir": obj.is_dir,
                }
                for obj in objects
            ]
        except S3Error as e:
            logger.error(f"Failed to list objects in {bucket_name}: {e}")
            return []

    def get_presigned_url(self, bucket_name: str, object_path: str, expires_days: int = 7) -> str:
        """Generate Presigned URL for Client to access image directly."""
        try:
            url = self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_path,
                expires=timedelta(days=expires_days)
            )
            return url
        except Exception as e:
            logger.error(f"Lấy link Presigned URL thất bại cho file {object_path}: {str(e)}")
            raise e

    def close(self):
        """Close the MinIO client and release resources."""
        if self._client:
            self._client = None
            logger.info("MinIO client connection reference cleared")


# Singleton instance
_minio_storage_client: Optional[MinioStorageClient] = None

def get_minio_storage_client() -> MinioStorageClient:
    """Get or create MinIO storage client singleton."""
    global _minio_storage_client
    if _minio_storage_client is None:
        _minio_storage_client = MinioStorageClient()
    return _minio_storage_client

minio_storage_client = get_minio_storage_client()