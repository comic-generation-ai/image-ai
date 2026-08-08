""" Config Settings for Comic System (image-ai)."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def load_environment_file(env_file: str | Path | None = None) -> Path:
    """Load variables from the project .env file into os.environ for local runs."""
    env_path = Path(env_file or ENV_FILE_PATH)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    return env_path


load_environment_file(ENV_FILE_PATH)


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
    DEFAULT_STEPS: int = Field(default=20, description="Default inference steps (SD1.5 non-turbo)")
    DEFAULT_WIDTH: int = Field(default=512, description="Default generated image width")
    DEFAULT_HEIGHT: int = Field(default=512, description="Default generated image height")
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
    
    # IP-Adapter Settings
    IP_ADAPTER_ENABLED: bool = Field(default=False, description="Enable IP-Adapter for image-to-image generation")
    IP_ADAPTER_SCALE: float = Field(default=0.6, description="Scale factor for IP-Adapter")
    
    # AI Model settings
    # Default khớp baseline thật đang chạy (Dreamshaper 8, SD1.5, Mac dev) — nếu .env
    # lỡ không load được (bug đã gặp — cwd sai khi chạy celery từ src/), fallback vẫn
    # đúng hướng thay vì âm thầm quay lại SDXL Turbo với steps/guidance sai lệch hẳn.
    MODEL_ID: str = Field(default="Lykon/dreamshaper-8")
    LOW_VRAM_MODE: bool = Field(default=False)
    MODEL_MIN_FREE_DISK_GB: float = Field(
        default=10.0,
        description="Minimum free disk space required before downloading/loading model artifacts",
    )
    COMIC_STYLE_ENABLED: bool = Field(default=True, description="Append comic/cartoon style to prompts")
    COMIC_STYLE_PROMPT_SUFFIX: str = Field(
        default=(
            "storybook watercolor, cute chibi children, pastel colors, "
            "bright background, single comic panel, one scene, sharp focus"
        ),
        description=(
            "Style suffix appended to user prompts (giữ ngắn cho CLIP 77 token). "
            "Dùng số ít 'panel'/'scene' — số nhiều dễ khiến model tự vẽ collage "
            "nhiều khung trong 1 ảnh, sai với mô hình 1 lần gọi = 1 khung."
        ),
    )
    DEFAULT_NEGATIVE_PROMPT: str = Field(
        default=(
            "blurry, out of focus, low resolution, pixelated, noisy, jpeg artifacts, "
            "photorealistic, realistic photo, dull colors, muddy details, bad anatomy, "
            "deformed hands, extra fingers, distorted face, text, watermark, logo, "
            "comic strip, panel grid, multiple panels, split screen, collage, montage"
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
        default=False,
        description="Chuyển layer sang CPU khi không dùng — chỉ bật khi chạy SDXL trên Mac 8–16GB.",
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
        default=380,
        description="Giới hạn độ dài prompt (CLIP ~77 token) để tránh truncate lỗi.",
    )
    DISABLE_DIFFUSERS_SAFETY_CHECKER: bool = Field(
        default=True,
        description="Tắt safety checker tích hợp diffusers (thường thay ảnh bằng khung đen).",
    )
    SAFETY_CHECKER_ENABLED: bool = Field(
        default=True,
        description=(
            "NSFW checker riêng (ViT ~500MB RAM, +1-3s/ảnh) — lớp lọc NSFW duy nhất "
            "của hệ thống. Dev Mac 8GB có thể tắt; production/demo phải bật."
        ),
    )
    GUIDANCE_SCALE: float = Field(default=7.0, description="Default CFG guidance for non-Turbo models")
    LCM_GUIDANCE_SCALE: float = Field(
        default=1.5,
        description="CFG cho model LCM (1.0-2.0; >1 thì negative prompt mới hoạt động)",
    )
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
    # Neo theo vị trí settings.py (không dùng cwd) — celery bắt buộc chạy từ src/ để
    # import được package worker.*, nhưng .env nằm ở project root; nếu để đường dẫn
    # tương đối thì Settings sẽ âm thầm bỏ qua .env và dùng default cứng trong code.
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
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
