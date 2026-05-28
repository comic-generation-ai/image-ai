""" Config Settings for Comic System (image-ai)."""

from functools import lru_cache
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    ENV: str = Field(default="development", description="Environment mode (development/production)")
    
    # Cache Settings
    PRESIGNED_TTL_SECONDS: int = Field(default=3600, description="Presigned URL TTL in seconds")
    REDIS_CACHE_TTL_SECONDS: int = Field(default=3600, description="Redis cache TTL in seconds")
    CACHE_KEY_VERSION: str = Field(default="v3", description="Version prefix for image cache key")
    CELERY_TASK_TIME_LIMIT: int = Field(default=3600, description="Celery task time limit in seconds")
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=3300, description="Celery soft task time limit in seconds")
    MAX_STEPS: int = Field(default=20, description="Maximum number of steps")
    MIN_STEPS: int = Field(default=1, description="Minimum number of steps")
    MAX_WIDTH: int = Field(default=1024, description="Maximum width")
    MAX_HEIGHT: int = Field(default=1024, description="Maximum height")
    CAPTION_MAX_LENGTH: int = Field(default=500, description="Maximum caption length")

    # TLS Settings
    TLS_ENABLED: bool = Field(default=False, description="Enable TLS for gRPC")
    TLS_CERT_PATH: str = Field(default="certs/image-ai-server.crt", description="Server certificate path")
    TLS_KEY_PATH: str = Field(default="certs/image-ai-server.key", description="Server private key path")
    TLS_CA_PATH: str = Field(default="certs/ca.crt", description="CA certificate path")
    TLS_REQUIRE_CLIENT_CERT: bool = Field(default=False, description="Require client certificate (mTLS)")
    
    #Logging 
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        description="Logging format"
    )
    LOG_DIR: str = Field(default="logs", description="Log directory for JSON logs")
    SERVICE_NAME: str = Field(default="image-ai", description="Service name for logging")
    METRICS_PORT: int = Field(default=9107, description="Prometheus metrics port")

    # gRPC & HTTP Server Settings
    HOST: str = Field(default="0.0.0.0", description="Host address")
    GRPC_PORT: int = Field(default=50051, description="gRPC listen port")
    HTTP_PORT: int = Field(default=8000, description="HTTP listen port")
    MAX_WORKERS: int = Field(default=3, description="Thread pool size")

    # MinIO (Object Storage) Settings
    MINIO_ENDPOINT: str = Field(default="127.0.0.1", description="MinIO server endpoint")
    MINIO_PORT: int = Field(default=9000, description="MinIO server port (host port, docker-compose)")
    MINIO_ACCESS_KEY: str = Field(default="", description="MinIO access key")
    MINIO_SECRET_KEY: str = Field(default="", description="MinIO secret key")
    MINIO_USE_SSL: bool = Field(default=False, description="Use SSL for MinIO connection")
    MINIO_BUCKET_NAME: str = Field(default="lvtn", description="Default MinIO bucket name")
    OUTPUT_IMAGE_FORMAT: str = Field(default="jpeg", description="Output image format: jpeg or png")
    JPEG_QUALITY: int = Field(default=95, description="JPEG quality for uploaded images")
    PNG_COMPRESS_LEVEL: int = Field(default=4, description="PNG compression level for uploaded images")

    # Redis Settings
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis server URL")

    # Celery Settings (Integration with Redis)
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")         
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", description="Celery result backend URL") 
    CELERY_TASK_SERIALIZER: str = Field(default="json", description="Celery task serializer")
    CELERY_RESULT_SERIALIZER: str = Field(default="json", description="Celery result serializer")
    CELERY_ACCEPT_CONTENT: list[str] = ["json"]
    CELERY_TIMEZONE: str = Field(default="Asia/Ho_Chi_Minh")
    CELERY_ENABLE_UTC: bool = Field(default=True)
    
    # AI Model settings
    MODEL_ID: str = Field(default="Lykon/dreamshaper-xl-v2-turbo")
    LOW_VRAM_MODE: bool = Field(default=False)

    # Device Settings
    DEVICE: str = Field(default="auto")

    # Paths
    BASE_DIR: str = Field(default=str(Path(__file__).resolve().parent.parent.parent))
    MODEL_CACHE_DIR: str = Field(
        default=str(Path(__file__).resolve().parent.parent.parent / ".cache")
    )
    
    # Configuration to load from .env file (biến dạng IMAGE_AI_* trong .env)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="IMAGE_AI_",
        extra="ignore",
    )
    
    @property
    def minio_url(self) -> str:
        return f"http://{self.MINIO_ENDPOINT}:{self.MINIO_PORT}"
    @property
    def redis_url(self) -> str:
        return self.REDIS_URL
    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL
    @property
    def celery_result_backend_url(self) -> str:
        return self.CELERY_RESULT_BACKEND
    


@lru_cache
def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
