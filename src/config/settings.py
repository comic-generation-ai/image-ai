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
    DEFAULT_STEPS: int = Field(default=4, description="Default inference steps")
    DEFAULT_WIDTH: int = Field(default=768, description="Default generated image width (768 nhanh hơn 1024 trên turbo)")
    DEFAULT_HEIGHT: int = Field(default=768, description="Default generated image height")
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
    MODEL_MIN_FREE_DISK_GB: float = Field(
        default=10.0,
        description="Minimum free disk space required before downloading/loading model artifacts",
    )
    COMIC_STYLE_ENABLED: bool = Field(default=True, description="Append comic/cartoon style to prompts")
    COMIC_STYLE_PROMPT_SUFFIX: str = Field(
        default=(
            "storybook watercolor, cute chibi children, pastel colors, "
            "bright background, comic panel, sharp focus"
        ),
        description="Style suffix appended to user prompts (giữ ngắn cho CLIP 77 token)",
    )
    DEFAULT_NEGATIVE_PROMPT: str = Field(
        default=(
            "blurry, out of focus, low resolution, pixelated, noisy, jpeg artifacts, "
            "photorealistic, realistic photo, dull colors, muddy details, bad anatomy, "
            "deformed hands, extra fingers, distorted face, text, watermark, logo"
        ),
        description="Default negative prompt for non-zero CFG generation",
    )
    TURBO_GUIDANCE_SCALE: float = Field(
        default=1.0,
        description="CFG cho Turbo trên CUDA. Giá trị thấp = nhanh và ổn định hơn.",
    )
    MPS_TURBO_GUIDANCE_SCALE: float = Field(
        default=0.0,
        description="CFG cho Turbo trên MPS — SDXL turbo thường dùng 0.",
    )
    MPS_DECODE_ON_CPU: bool = Field(
        default=True,
        description="Decode VAE trên CPU float32 (tránh NaN khi UNet chạy fp16/MPS).",
    )
    MPS_USE_FP32: bool = Field(
        default=False,
        description="Toàn bộ model fp32 trên MPS — cần >16GB RAM; Mac 8GB nên để false.",
    )
    MPS_VAE_FP32: bool = Field(
        default=True,
        description="VAE fp32 + UNet fp16 — tránh NaN decode mà vẫn vừa RAM Mac.",
    )
    MPS_ENABLE_SEQUENTIAL_CPU_OFFLOAD: bool = Field(
        default=True,
        description="Chuyển layer sang CPU khi không dùng — bắt buộc cho SDXL trên Mac 8–16GB.",
    )
    MPS_USE_CPU_GENERATOR: bool = Field(
        default=True,
        description="Dùng torch.Generator trên CPU khi chạy MPS (tránh ảnh đen).",
    )
    MPS_CLEAR_CACHE_AFTER_GENERATE: bool = Field(
        default=False,
        description="empty_cache MPS sau mỗi ảnh — tắt để ổn định/tăng tốc trên Mac.",
    )
    MAX_PROMPT_CHARS: int = Field(
        default=280,
        description="Giới hạn độ dài prompt (CLIP ~77 token) để tránh truncate lỗi.",
    )
    DISABLE_DIFFUSERS_SAFETY_CHECKER: bool = Field(
        default=True,
        description="Tắt safety checker tích hợp diffusers (thường thay ảnh bằng khung đen).",
    )
    GUIDANCE_SCALE: float = Field(default=2.0, description="Default CFG guidance for non-Turbo models")
    OUTPUT_VALIDATION_ENABLED: bool = Field(
        default=True,
        description="Kiểm tra ảnh trước upload — từ chối ảnh đen/hỏng.",
    )
    OUTPUT_MIN_MEAN_BRIGHTNESS: float = Field(
        default=8.0,
        description="Độ sáng trung bình tối thiểu (0-255) để chấp nhận ảnh.",
    )
    MIN_UPLOAD_BYTES: int = Field(
        default=2048,
        description="Kích thước file tối thiểu sau encode (phát hiện upload rỗng).",
    )
    LORA_ENABLED: bool = Field(default=False, description="Enable default LoRA adapter")
    LORA_PATH: str = Field(default="", description="Path to default LoRA safetensors file or directory")
    LORA_ADAPTER_NAME: str = Field(default="comic_style", description="Adapter name for default LoRA")
    LORA_SCALE: float = Field(default=0.85, description="Default LoRA adapter weight")
    LORA_TRIGGER_WORDS: str = Field(default="", description="Trigger words prepended to prompts")
    LORA_STRICT_COMPATIBILITY: bool = Field(
        default=True,
        description="Fail startup/generation when the configured LoRA format does not match the pipeline",
    )

    # Post-processing Settings
    COMIC_POSTPROCESS_ENABLED: bool = Field(default=True, description="Enhance generated image sharpness and color")
    SHARPEN_RADIUS: float = Field(default=1.1, description="Unsharp mask radius")
    SHARPEN_PERCENT: int = Field(default=135, description="Unsharp mask strength")
    SHARPEN_THRESHOLD: int = Field(default=3, description="Unsharp mask threshold")
    COLOR_BOOST: float = Field(default=1.08, description="Color saturation multiplier")
    CONTRAST_BOOST: float = Field(default=1.05, description="Contrast multiplier")

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
