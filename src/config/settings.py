""" Config Settings for Comic System (image-ai)."""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # gRPC & HTTP Server Settings
    HOST: str = "0.0.0.0"
    GRPC_PORT: int = 50051
    HTTP_PORT: int = 8000

    # MinIO (Object Storage) Settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "comics"

    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery Settings (Integration with Redis)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list[str] = ["json"]
    CELERY_TIMEZONE: str = "Asia/Ho_Chi_Minh"
    CELERY_ENABLE_UTC: bool = True
    
    # AI Model settings
    MODEL_ID: str = "Lykon/dreamshaper-xl-v2-turbo"
    LOW_VRAM_MODE: bool = False

    # Device Settings
    DEVICE: str = "auto"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_CACHE_DIR: Path = BASE_DIR / ".cache"
    
    # Configuration to load from .env file
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
    
@lru_cache
def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()