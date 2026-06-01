"""Validate generated images before upload to catch black/corrupt outputs early."""

from PIL import Image, ImageStat

from config.settings import get_settings
from logger.config import get_logger

logger = get_logger(__name__)
settings = get_settings()


def validate_generated_image(image: Image.Image) -> None:
    """
    Raise ValueError if the image looks empty, fully black, or corrupt.
    Uses mean luminance on a downsampled copy for speed.
    """
    if not settings.OUTPUT_VALIDATION_ENABLED:
        return

    if image is None:
        raise ValueError("Ảnh sinh ra rỗng (None)")

    width, height = image.size
    if width < 8 or height < 8:
        raise ValueError(f"Kích thước ảnh không hợp lệ: {width}x{height}")

    rgb = image.convert("RGB")
    sample = rgb.resize((64, 64), Image.Resampling.BILINEAR)
    stat = ImageStat.Stat(sample)
    mean_brightness = sum(stat.mean) / len(stat.mean)

    min_brightness = settings.OUTPUT_MIN_MEAN_BRIGHTNESS
    if mean_brightness < min_brightness:
        raise ValueError(
            f"Ảnh sinh ra quá tối hoặc hỏng (độ sáng trung bình={mean_brightness:.1f}, "
            f"ngưỡng tối thiểu={min_brightness}). "
            "Thử giảm guidance trên MPS, tăng số bước, hoặc kiểm tra cấu hình GPU."
        )

    logger.debug(f"Ảnh hợp lệ: mean_brightness={mean_brightness:.1f}")
