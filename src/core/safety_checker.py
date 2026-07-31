from dataclasses import dataclass
import torch
from transformers import pipeline
from config.settings import get_settings
from logger.config import get_logger

logger = get_logger(__name__)
settings = get_settings()

@dataclass(slots=True)
class SafetyResult:
    is_safe: bool
    nsfw_score: float


class SafetyChecker:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.classifier = None
        self.enabled = settings.SAFETY_CHECKER_ENABLED
        if not self.enabled:
            # Dev Mac 8GB: tiết kiệm ~500MB RAM + 1-3s/ảnh. Production phải bật —
            # đây là lớp lọc NSFW duy nhất (checker của diffusers đã tắt).
            logger.info(
                "Safety checker TẮT qua IMAGE_AI_SAFETY_CHECKER_ENABLED=false — không load model NSFW"
            )
            return
        try:
            self.classifier = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Không thể khởi tạo safety checker model: {str(e)}")

    @property
    def device_name(self) -> str:
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def check_image(self, image) -> SafetyResult:
        if not self.enabled:
            # Tắt chủ động qua config — pass im lặng, không spam log mỗi ảnh
            return SafetyResult(is_safe=True, nsfw_score=0.0)
        if self.classifier is None:
            logger.warning("Safety checker unavailable, fallback is_safe=True")
            return SafetyResult(is_safe=True, nsfw_score=0.0)
        try:
            results = self.classifier(image)
            nsfw_score = 0.0

            for item in results:
                if item["label"].lower() == "nsfw":
                    nsfw_score = item["score"]

            return SafetyResult(
                is_safe=nsfw_score < self.threshold,
                nsfw_score=nsfw_score
            )

        except Exception as e:
            logger.error(f"Safety checker failed: {str(e)}")
            return SafetyResult(
                is_safe=True,
                nsfw_score=0.0
            )

safety_checker = SafetyChecker()