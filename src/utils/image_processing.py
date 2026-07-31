import os

from PIL import Image, ImageEnhance, ImageFilter
from config.settings import get_settings
from logger.config import get_logger

logger = get_logger(__name__)
settings = get_settings()

def enhance_comic_image(image: Image.Image) -> Image.Image:
    """
    Tăng độ nét/màu nhẹ để ảnh đọc giống panel hoạt hình hơn mà không đổi bố cục.
    """
    if not settings.COMIC_POSTPROCESS_ENABLED:
        return image.convert("RGB")

    enhanced = image.convert("RGB")
    enhanced = ImageEnhance.Color(enhanced).enhance(settings.COLOR_BOOST)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(settings.CONTRAST_BOOST)
    enhanced = enhanced.filter(
        ImageFilter.UnsharpMask(
            radius=settings.SHARPEN_RADIUS,
            percent=settings.SHARPEN_PERCENT,
            threshold=settings.SHARPEN_THRESHOLD,
        )
    )
    return enhanced

