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
from core.lora_loader import lora_loader
from core.vram_manager import vram_manager

logger = get_logger(__name__)


@dataclass
class ImageRequest:
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    steps: int = 4
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
            # fp16 trên MPS gây NaN khi VAE decode → ảnh đen (invalid value in cast)
            self.dtype = (
                torch.float32
                if self.settings.MPS_USE_FP32
                else torch.float16
            )

        else:
            self.device = "cpu"
            self.dtype = torch.float32

        logger.info(
            f"Running on device: {self.device} | dtype={self.dtype} | "
            f"mps_fp32={self.settings.MPS_USE_FP32}"
        )
        self._configure_hf_cache()

    @property
    def is_sdxl(self) -> bool:
        return "xl" in self.model_id.lower()

    @property
    def is_turbo(self) -> bool:
        return "turbo" in self.model_id.lower()

    def _default_guidance_scale(self) -> float:
        # SD-Turbo: CFG thấp/0; trên MPS fp16 thường cần 0 để tránh ảnh đen/artifact.
        if self.is_turbo:
            if self.device == "mps":
                return self.settings.MPS_TURBO_GUIDANCE_SCALE
            return self.settings.TURBO_GUIDANCE_SCALE
        return self.settings.GUIDANCE_SCALE

    def _generator_device(self) -> str:
        """MPS Generator trên CPU ổn định hơn — tránh ảnh đen trên Apple Silicon."""
        if self.settings.MPS_USE_CPU_GENERATOR and self.device == "mps":
            return "cpu"
        return self.device

    @property
    def default_guidance_scale(self) -> float:
        return self._default_guidance_scale()

    @property
    def cache_signature(self) -> str:
        style_suffix = (
            self.settings.COMIC_STYLE_PROMPT_SUFFIX
            if self.settings.COMIC_STYLE_ENABLED
            else ""
        )
        return (
            f"{self.model_id}|comic_style={self.settings.COMIC_STYLE_ENABLED}|"
            f"style={style_suffix}|lora={self.lora_signature}"
        )

    @property
    def lora_signature(self) -> str:
        if not self.settings.LORA_ENABLED:
            return "disabled"
        return (
            f"path={self._resolve_lora_path()}|adapter={self.settings.LORA_ADAPTER_NAME}|"
            f"scale={self.settings.LORA_SCALE}|trigger={self.settings.LORA_TRIGGER_WORDS}"
        )

    def _resolve_lora_path(self, lora_path: str | None = None) -> str:
        raw_path = (lora_path or self.settings.LORA_PATH or "").strip()
        if not raw_path:
            return ""
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(self.settings.BASE_DIR) / path
        return str(path)

    def _ensure_lora_loaded(self, lora_path: str | None = None) -> None:
        if not self.settings.LORA_ENABLED and not lora_path:
            return

        resolved_path = self._resolve_lora_path(lora_path)
        if not resolved_path:
            raise ValueError("LORA_ENABLED=true nhưng IMAGE_AI_LORA_PATH đang rỗng")

        self.pipeline = lora_loader.load_lora(
            pipeline=self.pipeline,
            lora_path=resolved_path,
            adapter_name=self.settings.LORA_ADAPTER_NAME,
            adapter_weight=self.settings.LORA_SCALE,
            strict_compatibility=self.settings.LORA_STRICT_COMPATIBILITY,
        )

    def _build_prompt(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        trigger_words = self.settings.LORA_TRIGGER_WORDS.strip()
        if trigger_words:
            prompt = f"{trigger_words}, {prompt}" if prompt else trigger_words
        if not self.settings.COMIC_STYLE_ENABLED:
            return self._truncate_for_clip(prompt)
        style_suffix = self.settings.COMIC_STYLE_PROMPT_SUFFIX.strip()
        if not style_suffix:
            return self._truncate_for_clip(prompt)
        if not prompt:
            return self._truncate_for_clip(style_suffix)
        return self._truncate_for_clip(f"{prompt}, {style_suffix}")

    def _truncate_for_clip(self, text: str) -> str:
        """CLIP giới hạn ~77 token; cắt sớm để tránh truncate làm hỏng prompt."""
        limit = self.settings.MAX_PROMPT_CHARS
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        trimmed = text[:limit].rsplit(",", 1)[0].strip()
        logger.warning(
            f"Prompt bị rút gọn {len(text)} → {len(trimmed)} ký tự (CLIP limit)"
        )
        return trimmed

    def _build_negative_prompt(self, negative_prompt: str) -> str:
        parts = [
            value.strip()
            for value in [negative_prompt, self.settings.DEFAULT_NEGATIVE_PROMPT]
            if value and value.strip()
        ]
        return ", ".join(dict.fromkeys(parts))

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
        if self.settings.MODEL_MIN_FREE_DISK_GB > 0:
            return self.settings.MODEL_MIN_FREE_DISK_GB
        if self.is_sdxl:
            return 10.0
        return 4.0

    def _check_disk_space(self) -> None:
        path = self._hf_cache_path()
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        required_gb = self._required_free_disk_gb()
        if free_gb < required_gb:
            lighter = "stabilityai/sd-turbo" if self.is_sdxl else None
            hint = (
                f" Gợi ý: đặt MODEL_ID={lighter} trong .env (cần ~4GB trống), "
                "hoặc giảm IMAGE_AI_MODEL_MIN_FREE_DISK_GB nếu model đã có cache."
                if lighter
                else ""
            )
            raise RuntimeError(
                f"Không đủ dung lượng ổ đĩa để tải model '{self.model_id}': "
                f"cần ~{required_gb:.1f}GB, còn {free_gb:.1f}GB tại {path}.{hint} "
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

        # SDXL pipeline không nhận safety_checker kwargs (chỉ SD 1.x)
        if (
            not self.is_sdxl
            and self.settings.DISABLE_DIFFUSERS_SAFETY_CHECKER
        ):
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

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

        if self._uses_mps_offload():
            self.pipeline.enable_sequential_cpu_offload()
            logger.info("MPS: sequential CPU offload enabled")
        elif self.low_vram_mode and self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
            logger.info("CUDA: model CPU offload enabled")
        else:
            self.pipeline.to(self.device)

        self._optimize_pipeline()
        self._configure_mps_precision()
        with torch.inference_mode():
            self._ensure_lora_loaded()

        logger.info("Pipeline initialized successfully")

    def _uses_mps_offload(self) -> bool:
        return (
            self.device == "mps"
            and self.settings.MPS_ENABLE_SEQUENTIAL_CPU_OFFLOAD
        )

    def _uses_mps_cpu_decode(self) -> bool:
        return self.device == "mps" and self.settings.MPS_DECODE_ON_CPU

    def _configure_mps_precision(self) -> None:
        """Cấu hình precision cho MPS — tránh NaN khi decode VAE."""
        if self.device != "mps" or self.pipeline is None:
            return
        # sequential CPU offload đặt hook lên module — gọi .to() gây meta tensor error
        if self._uses_mps_offload():
            if self._uses_mps_cpu_decode():
                logger.info(
                    "MPS: decode latent trên CPU (sequential offload — không gọi vae.to)"
                )
            return
        if self._uses_mps_cpu_decode():
            self.pipeline.vae.to(device="cpu", dtype=torch.float32)
            logger.info("MPS: VAE pinned to CPU float32 for decode")
            return
        if self.settings.MPS_VAE_FP32 and not self.settings.MPS_USE_FP32:
            self.pipeline.vae.to(dtype=torch.float32)
            logger.info("MPS: VAE float32 (UNet giữ fp16)")
        elif self.settings.MPS_USE_FP32:
            self.pipeline.vae.to(dtype=torch.float32)
            logger.info("MPS: full float32 mode")

    def _vae_weight_dtype(self) -> torch.dtype:
        try:
            return next(self.pipeline.vae.parameters()).dtype
        except StopIteration:
            return self.dtype

    def _decode_latents_on_cpu(self, latents: torch.Tensor) -> Image.Image:
        """Decode latent trên CPU — dtype latent khớp VAE (fp16 khi pipeline fp16)."""
        vae = self.pipeline.vae
        vae_dtype = self._vae_weight_dtype()
        latents = latents.detach().cpu().to(dtype=vae_dtype)
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            raise ValueError(
                "Latent chứa NaN/Inf sau bước diffusion (UNet fp16 trên MPS). "
                "Thử IMAGE_AI_MPS_TURBO_GUIDANCE_SCALE=0, tắt LoRA, hoặc model nhẹ hơn."
            )
        scaling = getattr(vae.config, "scaling_factor", 0.13025)
        with torch.inference_mode():
            decoded = vae.decode(latents / scaling, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(decoded, output_type="pil")
        return images[0]

    def _run_inference(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator: torch.Generator,
    ) -> Image.Image:
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if self._uses_mps_cpu_decode():
            pipe_kwargs["output_type"] = "latent"
        result = self.pipeline(**pipe_kwargs)
        if self._uses_mps_cpu_decode():
            return self._decode_latents_on_cpu(result.images)
        return result.images[0]

    def warmup(self):
        """
        Chạy thử một lượt sinh ảnh nháp (Dummy Inference) với prompt rỗng ("") và 1 step
        để khởi động (warmup) GPU/MPS kernel trước khi nhận request thực tế.
        """
        self.initialize_pipeline()
        logger.info("Bắt đầu khởi động (warmup) mô hình Stable Diffusion...")
        try:
            # Chạy thử 1 step cực nhẹ với prompt trống và kích thước tối thiểu
            width = min(384, self.settings.DEFAULT_WIDTH)
            height = min(384, self.settings.DEFAULT_HEIGHT)
            steps = 1
            
            generator = torch.Generator(
                device=self._generator_device()
            ).manual_seed(42)
            
            guidance_scale = self._default_guidance_scale()
            
            with torch.inference_mode():
                self._run_inference(
                    prompt="warmup",
                    negative_prompt="",
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            logger.info("Khởi động (warmup) mô hình thành công!")
        except Exception as e:
            logger.warning(f"Khởi động (warmup) mô hình thất bại: {str(e)}")
        finally:
            vram_manager.clear_cache()


    def _optimize_pipeline(self):
        if self.device == "cuda":
            self.pipeline.vae.enable_slicing()
            if self.low_vram_mode:
                self.pipeline.vae.enable_tiling()
                self.pipeline.enable_attention_slicing()
        elif self.device == "mps" and not self._uses_mps_cpu_decode():
            self.pipeline.vae.enable_slicing()

        if self.device == "cuda":
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xFormers enabled")
            except Exception:
                logger.warning("xFormers unavailable")

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
            request.steps,
        )
        with self._lock:
            try:
                with torch.inference_mode():
                    self._ensure_lora_loaded(request.lora_path)

                    seed = request.seed
                    if seed == -1:
                        seed = int(torch.randint(0, 2147483647, (1,)).item())
                    generator = torch.Generator(
                        device=self._generator_device()
                    ).manual_seed(seed)

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
                    prompt = self._build_prompt(request.prompt)
                    negative_prompt = self._build_negative_prompt(
                        request.negative_prompt
                    )

                    image = self._run_inference(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")

                return ImageResponse(
                    image=image,
                    seed=seed
                )

            except Exception as e:
                logger.error(f"Inference failed: {str(e)}")
                raise e

            finally:
                if (
                    self.device != "mps"
                    or self.settings.MPS_CLEAR_CACHE_AFTER_GENERATE
                ):
                    vram_manager.clear_cache()


pipeline_runner = PipelineRunner()
