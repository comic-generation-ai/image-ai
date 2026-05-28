import os
import shutil
import threading
from pathlib import Path

import torch

from PIL import Image
from dataclasses import dataclass
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
)

from config.settings import get_settings
from logger.config import get_logger
from core.vram_manager import vram_manager
from typing import Optional

logger = get_logger(__name__)


@dataclass
class ImageRequest:
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 8
    guidance_scale: float | None = None
    negative_prompt: str = ""
    lora_path: str | None = None


@dataclass
class ImageResponse:
    image: Image.Image
    seed: int


class PipelineRunner:
    def __init__(self):
        self.settings = get_settings()

        self.model_id = self.settings.MODEL_ID
        self.low_vram_mode = self.settings.LOW_VRAM_MODE

        self.pipeline = None

        self._lock = threading.Lock()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16

        elif torch.backends.mps.is_available():
            self.device = "mps"
            # fp16 UNet + fp32 VAE → lỗi "Input type (Half) and bias type (float)" khi decode
            self.dtype = torch.float32

        else:
            self.device = "cpu"
            self.dtype = torch.float32

        logger.info(f"Running on device: {self.device}")
        self._configure_hf_cache()

    @property
    def is_sdxl(self) -> bool:
        return "xl" in self.model_id.lower()

    @property
    def is_turbo(self) -> bool:
        return "turbo" in self.model_id.lower()

    def _default_guidance_scale(self) -> float:
        # SD-Turbo được distill với CFG=0; giá trị >0 dễ gây artefact/NaN trên MPS
        return 0.0 if self.is_turbo else 2.0

    def _configure_hf_cache(self) -> None:
        """Dùng MODEL_CACHE_DIR từ settings thay vì ~/.cache mặc định."""
        cache_root = Path(self.settings.MODEL_CACHE_DIR)
        cache_root.mkdir(parents=True, exist_ok=True)
        hub_cache = cache_root / "hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))

    def _hf_cache_path(self) -> str:
        return os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            str(Path(self.settings.MODEL_CACHE_DIR) / "hub"),
        )

    def _required_free_disk_gb(self) -> float:
        if self.is_sdxl:
            return 16.0
        return 6.0

    def _check_disk_space(self) -> None:
        path = self._hf_cache_path()
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        required_gb = self._required_free_disk_gb()
        if free_gb < required_gb:
            lighter = "stabilityai/sd-turbo" if self.is_sdxl else None
            hint = (
                f" Gợi ý: đặt MODEL_ID={lighter} trong .env (cần ~6GB trống)."
                if lighter
                else ""
            )
            raise RuntimeError(
                f"Không đủ dung lượng ổ đĩa để tải model '{self.model_id}': "
                f"cần ~{required_gb:.0f}GB, còn {free_gb:.1f}GB tại {path}.{hint} "
                "Hoặc giải phóng dung lượng / xóa cache HuggingFace cũ."
            )

    def initialize_pipeline(self):
        if self.pipeline is not None:
            return

        self._check_disk_space()
        logger.info(f"Loading model: {self.model_id} (cache: {self._hf_cache_path()})")

        kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "cache_dir": self._hf_cache_path(),
        }

        if self.device == "cuda":
            kwargs["variant"] = "fp16"

        pipeline_cls = (
            StableDiffusionXLPipeline if self.is_sdxl else StableDiffusionPipeline
        )
        self.pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)

        if not self.is_turbo:
            self.pipeline.scheduler = (
                EulerAncestralDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            )

        self.pipeline.to(self.device)

        self._optimize_pipeline()

        logger.info("Pipeline initialized successfully")

    def _optimize_pipeline(self):

        if self.device != "cpu":
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()
            self.pipeline.enable_attention_slicing()

        if self.device == "cuda":

            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xFormers enabled")
            except Exception:
                logger.warning("xFormers unavailable")

        if self.low_vram_mode and self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()

    def _validate_inputs(self, width, height, steps):

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width/Height must be divisible by 8")

        if steps < 1 or steps > 20:
            raise ValueError("Steps must be between 1 and 20")

    def generate(self, request: ImageRequest) -> ImageResponse:

        self.initialize_pipeline()

        self._validate_inputs(
            request.width,
            request.height,
            request.steps
        )
        with self._lock:
            try:
                seed = request.seed
                if seed == -1:
                    seed = torch.seed()
                # Generator trên MPS không ổn định; dùng CPU là workaround phổ biến
                gen_device = "cpu" if self.device == "mps" else self.device
                generator = torch.Generator(device=gen_device).manual_seed(seed)

                guidance_scale = (
                    request.guidance_scale
                    if request.guidance_scale is not None
                    else self._default_guidance_scale()
                )

                logger.info(
                    f"Generating image | "
                    f"Seed={seed} | "
                    f"Steps={request.steps} | "
                    f"Guidance={guidance_scale}"
                )

                with torch.inference_mode():

                    result = self.pipeline(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        width=request.width,
                        height=request.height,
                        num_inference_steps=request.steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )

                image = result.images[0]

                return ImageResponse(
                    image=image,
                    seed=seed
                )

            except Exception as e:
                logger.error(f"Inference failed: {str(e)}")
                raise e

            finally:
                vram_manager.clear_cache()


pipeline_runner = PipelineRunner()