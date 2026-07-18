import io
import os
import re
import shutil
import threading
from pathlib import Path

import requests
import torch

from PIL import Image
from dataclasses import dataclass, replace
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    LCMScheduler,
)

try:
    from compel import Compel
except ImportError:
    Compel = None

from config.settings import get_settings
from logger.config import get_logger
from core.lora_loader import lora_loader
from core.vram_manager import vram_manager

logger = get_logger(__name__)

@dataclass
class ImageRequest:
    """
    This class is used to request an image from the pipeline.
    Attributes:
        prompt: The prompt to use for the image generation.
        width: The width of the image.
        height: The height of the image.
        seed: The seed to use for the image generation.
        steps: The number of steps to use for the image generation.
        guidance_scale: The guidance scale to use for the image generation.
        negative_prompt: The negative prompt to use for the image generation.
        lora_path: The path to the lora to use for the image generation.
    """
    prompt: str
    width: int = 1024
    height: int = 1024
    reference_image_url: str = ""
    seed: int = -1
    steps: int = 4
    guidance_scale: float | None = None
    negative_prompt: str = ""
    lora_path: str | None = None
    style: str = ""


@dataclass
class ImageResponse:
    image: Image.Image
    seed: int


STYLE_PRESETS = {
    "storybook": {
        "suffix": "whimsical fable book illustration, highly detailed, vibrant colors, fantasy art style",
        "negative": "ugly, blurry, low quality, photorealistic, 3d render, dark, scary, deformed hands, bad anatomy, text, watermark, logo, nsfw",
    },
    "anime": {
        "suffix": "anime key visual style, highly detailed digital illustration, vibrant colors, clean lineart, anime background, comic book panel",
        "negative": "monochrome, black and white, sketch, draft, ugly, blurry, low quality, photorealistic, 3d render, deformed hands, bad anatomy, text, watermark, logo, nsfw",
    },
    "manga": {
        "suffix": "manga art style, black and white, screen tones, hand-drawn ink sketch, highly detailed lineart, dramatic shading, comic book panel",
        "negative": "colored, vibrant colors, painting, watercolor, ugly, blurry, low quality, photorealistic, 3d render, deformed hands, bad anatomy, text, watermark, logo, nsfw",
    },
    "retro": {
        "suffix": "retro classic comic book style, pop art, dot shading halftone print, vintage colors, bold ink outlines, 1980s print quality",
        "negative": "modern digital painting, smooth gradients, watercolor, anime, manga, photorealistic, 3d render, ugly, blurry, low quality, text, watermark, logo, nsfw",
    },
    "american_comic": {
        "suffix": "classic american comic book style, dramatic lighting, bold lines, rich colors, superhero comic art, action panel",
        "negative": "anime, manga, watercolor, soft illustration, photorealistic, 3d render, ugly, blurry, low quality, text, watermark, logo, nsfw",
    }
}


class PipelineRunner:
    def _parse_style_from_prompt(self, prompt: str) -> tuple[str, str]:
        prompt = (prompt or "").strip()
        if prompt.startswith("[style:"):
            end_idx = prompt.find("]")
            if end_idx != -1:
                style_name = prompt[7:end_idx].strip().lower()
                clean_prompt = prompt[end_idx + 1:].strip()
                return clean_prompt, style_name
        return prompt, ""

    def __init__(self):
        self.settings = get_settings()

        self.model_id = self.settings.MODEL_ID
        self.low_vram_mode = self.settings.LOW_VRAM_MODE
        self.compel = None
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

    @property
    def is_lcm(self) -> bool:
        return "lcm" in self.model_id.lower()

    def _default_guidance_scale(self) -> float:
        if self.is_turbo:
            if self.device == "mps":
                return self.settings.MPS_TURBO_GUIDANCE_SCALE
            return self.settings.TURBO_GUIDANCE_SCALE
        # LCM distill: CFG 1.0-2.0; >1 thì negative prompt mới có tác dụng
        if self.is_lcm:
            return self.settings.LCM_GUIDANCE_SCALE
        return self.settings.GUIDANCE_SCALE

    def _generator_device(self) -> str:
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
            f"style={style_suffix}|lora={self.lora_signature}|"
            f"ip_adapter={self.settings.IP_ADAPTER_ENABLED}:{self.settings.IP_ADAPTER_SCALE}"
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

    _MJ_FLAG_PATTERN = re.compile(r"--\w+\s+\S+")

    def _strip_midjourney_flags(self, text: str) -> str:
        """Loại cú pháp riêng của Midjourney (--ar 16:9, --v 5, --style raw,...) —
        CLIP/SD không hiểu, chỉ tokenize thành rác và phí ngân sách 77 token."""
        cleaned = self._MJ_FLAG_PATTERN.sub("", text)
        return re.sub(r"\s{2,}", " ", cleaned).strip().rstrip(",").strip()

    def _build_prompt(self, prompt: str, style: str = "") -> str:
        prompt = (prompt or "").strip()
        clean_prompt, parsed_style = self._parse_style_from_prompt(prompt)
        clean_prompt = self._strip_midjourney_flags(clean_prompt)
        style_to_use = (style or parsed_style or "").strip().lower()

        # Chỉ chèn trigger words khi LoRA thực sự bật — nếu không sẽ phí ngân sách
        # ký tự cho từ khoá vô nghĩa với base model (vd "EldritchComicsXL").
        trigger_words = self.settings.LORA_TRIGGER_WORDS.strip() if self.settings.LORA_ENABLED else ""
        if trigger_words:
            clean_prompt = f"{trigger_words}, {clean_prompt}" if clean_prompt else trigger_words
        if not self.settings.COMIC_STYLE_ENABLED:
            return self._truncate_for_clip(clean_prompt)

        if style_to_use in STYLE_PRESETS:
            style_suffix = STYLE_PRESETS[style_to_use]["suffix"]
        else:
            style_suffix = self.settings.COMIC_STYLE_PROMPT_SUFFIX.strip()

        if not style_suffix:
            return self._truncate_for_clip(clean_prompt)
        if not clean_prompt:
            return self._truncate_for_clip(style_suffix)

        # Ưu tiên giữ nguyên style_suffix (chặn multi-panel + định hướng phong cách) —
        # cắt bớt phần mô tả cảnh (thường từ story-ai, có thể rất dài) để nhường chỗ,
        # tránh suffix bị rơi mất hoàn toàn khi prompt đầu vào đã gần/vượt giới hạn CLIP.
        limit = self.settings.MAX_PROMPT_CHARS
        reserved = len(style_suffix) + 2  # ", "
        prompt_budget = max(limit - reserved, 0)
        if len(clean_prompt) > prompt_budget:
            trimmed_prompt = self._truncate_scene_prompt(clean_prompt, prompt_budget)
            logger.warning(
                f"Prompt mô tả cảnh bị rút gọn {len(clean_prompt)} → {len(trimmed_prompt)} ký tự "
                f"để dành {len(style_suffix)} ký tự cho style suffix (giới hạn tổng {limit} ký tự)."
            )
            clean_prompt = trimmed_prompt

        return self._truncate_for_clip(f"{clean_prompt}, {style_suffix}")

    def _truncate_scene_prompt(self, clean_prompt: str, prompt_budget: int) -> str:
        two_char_match = self._TWO_CHARACTER_PATTERN.match(clean_prompt)
        if two_char_match:
            protected = f"on the left, {two_char_match.group('left').strip()}; on the right, "
            droppable_source = two_char_match.group("right").strip()
        else:
            protected = ""
            droppable_source = clean_prompt

        segments = [s.strip() for s in droppable_source.split(",") if s.strip()]
        # Chỉ 2 tag ĐẦU (identity, action) được coi là bảo vệ tuyệt đối — mọi
        # tag sau đó (setting, camera, và các tag khí quyển phụ LLM hay thêm dư
        # như "dust kicking up") đều droppable, bỏ dần từ cuối tới khi vừa ngân
        # sách. Không giới hạn cứng "chỉ bỏ tối đa 2 tag" vì thực tế story-ai có
        # lúc viết 3-4 tag setting/camera thay vì đúng 2 như rule đề ra.
        min_segments = min(2, len(segments))

        candidate = protected + ", ".join(segments)
        while len(segments) > min_segments and len(candidate) > prompt_budget:
            segments = segments[:-1]
            candidate = protected + ", ".join(segments)

        if len(candidate) <= prompt_budget:
            return candidate

        logger.warning(
            "Prompt vẫn vượt ngân sách sau khi đã bỏ hết setting/camera droppable "
            "— fallback cắt thô, có thể ảnh hưởng phần nhân vật/hành động. "
            "Cân nhắc tăng IMAGE_AI_MAX_PROMPT_CHARS nếu lỗi này lặp lại nhiều."
        )
        return candidate[:prompt_budget].strip()

    def _truncate_for_clip(self, text: str) -> str:
        limit = self.settings.MAX_PROMPT_CHARS
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        trimmed = text[:limit].rsplit(",", 1)[0].strip()
        logger.warning(
            f"Prompt bị rút gọn {len(text)} → {len(trimmed)} ký tự (CLIP limit)"
        )
        return trimmed

    def _build_negative_prompt(self, negative_prompt: str, style: str = "") -> str:
        style_to_use = (style or "").strip().lower()
        if style_to_use in STYLE_PRESETS:
            default_neg = STYLE_PRESETS[style_to_use]["negative"]
        else:
            default_neg = self.settings.DEFAULT_NEGATIVE_PROMPT

        parts = [
            value.strip()
            for value in [negative_prompt, default_neg]
            if value and value.strip()
        ]
        # Negative prompt cũng bị giới hạn CLIP 77 token như prompt dương —
        # không cắt thì transformers truncate câm lặng phần đuôi.
        return self._truncate_for_clip(", ".join(dict.fromkeys(parts)))

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
        try:
            self.pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
        except Exception as e:
            # CUDA yêu cầu variant fp16 nhưng không phải repo nào cũng ship file fp16
            # (vd một số checkpoint cộng đồng) — tải bản thường rồi cast theo torch_dtype.
            if kwargs.get("variant") == "fp16":
                logger.warning(
                    f"Model không có variant fp16 ({e}) — tải bản thường, cast fp16 sau."
                )
                kwargs.pop("variant", None)
                self.pipeline = pipeline_cls.from_pretrained(self.model_id, **kwargs)
            else:
                raise

        if self.is_lcm:
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config
            )
            logger.info("LCM model: scheduler = LCMScheduler")
        elif not self.is_turbo:
            # DPM++ 2M Karras: giữ nguyên config gốc của checkpoint (beta_schedule,
            # prediction_type,...) qua from_config — chỉ override phần Karras/dpmsolver++.
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )

        if self.settings.IP_ADAPTER_ENABLED:
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models" if self.is_sdxl else "models",
                weight_name="ip-adapter_sdxl.bin" if self.is_sdxl else "ip-adapter_sd15.bin",
            )
            self.pipeline.set_ip_adapter_scale(self.settings.IP_ADAPTER_SCALE)
            logger.info(f"IP-Adapter enabled (scale={self.settings.IP_ADAPTER_SCALE})")

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
            
        self._init_compel()

        logger.info("Pipeline initialized successfully")
        
        
    def _init_compel(self):
        if Compel is None:
            logger.warning("Compel not installed, weighted prompts will not work")
            return
        if self.is_sdxl:
            logger.warning("Compel chưa hỗ trợ SDXL dual-encoder — dùng prompt dạng text thuần")
            return

        try:
            self.compel = Compel(
                tokenizer=self.pipeline.tokenizer,
                text_encoder=self.pipeline.text_encoder,
                truncate_long_prompts=False
            )
            logger.info("Compel initialized successfully")
        except Exception as e:
            self.compel = None
            logger.error(f"Failed to initialize Compel: {e}")

    # Compel coi ( ) + và - cuối từ là cú pháp weighting — prompt tự nhiên chứa
    # ngoặc đơn hoặc từ có gạch nối (hand-drawn, close-up) sẽ bị parse sai nghĩa.
    _COMPEL_UNSAFE_CHARS = str.maketrans({"(": " ", ")": " ", "+": " ", "-": " "})

    def _encode_with_compel(self, prompt: str, negative_prompt: str) -> dict | None:
        """
        Encode prompt bằng Compel: chia chunk 77 token, encode từng chunk rồi nối
        embeddings — bỏ được giới hạn CLIP với prompt dài từ story-ai.
        Trả về dict prompt_embeds/negative_prompt_embeds, hoặc None để caller
        fallback về đường truyền prompt dạng text (bị truncate 77 token).
        """   
        if self.compel is None:
            return None
        try:
            clean = prompt.translate(self._COMPEL_UNSAFE_CHARS)
            clean_neg = (negative_prompt or "").translate(self._COMPEL_UNSAFE_CHARS)
            conditioning = self.compel(clean)
            negative_conditioning = self.compel(clean_neg)
            # Bắt buộc pad về cùng số chunk: prompt dài 2-3 chunk còn negative
            # thường 1 chunk — lệch shape là pipeline crash ngay bước CFG.
            [conditioning, negative_conditioning] = (
                self.compel.pad_conditioning_tensors_to_same_length(
                    [conditioning, negative_conditioning]
                )
            )
            return {
                "prompt_embeds": conditioning,
                "negative_prompt_embeds": negative_conditioning,
            }
        except Exception as e:
            logger.warning(f"Compel encode thất bại ({e}) — fallback prompt dạng text")
            return None
    
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

    def _load_reference_image(self, url: str) -> Image.Image | None:
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.warning(
                f"Không tải được ảnh tham chiếu ({url}): {e} — sinh ảnh không dùng reference."
            )
            return None

    _BLANK_IP_ADAPTER_IMAGE: Image.Image | None = None

    def _blank_ip_adapter_image(self) -> Image.Image:
        """Ảnh giả (placeholder) cho IP-Adapter khi không có reference thật.

        Diffusers yêu cầu luôn phải truyền ip_adapter_image một khi pipeline đã
        load_ip_adapter() — không truyền sẽ crash 'NoneType is not iterable'. Dùng
        ảnh trắng trơn + set_ip_adapter_scale(0) để triệt tiêu ảnh hưởng thị giác.
        """
        if self._BLANK_IP_ADAPTER_IMAGE is None:
            self._BLANK_IP_ADAPTER_IMAGE = Image.new("RGB", (224, 224), color="white")
        return self._BLANK_IP_ADAPTER_IMAGE

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
        reference_image: Image.Image | None = None,
    ) -> Image.Image:
        pipe_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        
        embeds_kwargs = self._encode_with_compel(prompt, negative_prompt)
        if embeds_kwargs is not None:
            pipe_kwargs.update(embeds_kwargs)
        else:
            pipe_kwargs["prompt"] = prompt
            pipe_kwargs["negative_prompt"] = negative_prompt
        if reference_image is not None:
            pipe_kwargs["ip_adapter_image"] = reference_image
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
            # Warmup đúng resolution mặc định — kernel MPS biên dịch theo kích thước,
            # warmup 384 thì request 512 đầu tiên vẫn tốn ~17-19s biên dịch lại.
            width = self.settings.DEFAULT_WIDTH
            height = self.settings.DEFAULT_HEIGHT
            
            steps = 1
            
            generator = torch.Generator(
                device=self._generator_device()
            ).manual_seed(42)
            
            guidance_scale = self._default_guidance_scale()

            if self.settings.IP_ADAPTER_ENABLED:
                self.pipeline.set_ip_adapter_scale(0.0)

            with torch.inference_mode():
                self._run_inference(
                    prompt="warmup",
                    negative_prompt="",
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    reference_image=(
                        self._blank_ip_adapter_image()
                        if self.settings.IP_ADAPTER_ENABLED
                        else None
                    ),
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

        if steps < self.settings.MIN_STEPS or steps > self.settings.MAX_STEPS:
            raise ValueError(f"Steps must be between {self.settings.MIN_STEPS} and {self.settings.MAX_STEPS}")

    _TWO_CHARACTER_PATTERN = re.compile(
        r"^on the left,\s*(?P<left>.+?);\s*on the right,\s*(?P<right>.+)$",
        re.IGNORECASE | re.DOTALL,
    )

    def _split_two_character_prompt(self, prompt: str) -> tuple[str, str] | None:
        clean_prompt, _ = self._parse_style_from_prompt(prompt)
        match = self._TWO_CHARACTER_PATTERN.match(clean_prompt.strip())
        if not match:
            return None

        left = match.group("left").strip().rstrip(",").strip()
        right = match.group("right").strip().rstrip(",").strip()
        if not left or not right:
            return None
        segments = [s.strip() for s in right.split(",") if s.strip()]
        tail = ", ".join(segments[-2:]) if len(segments) >= 3 else ""
        left_prompt = f"{left}, {tail}" if tail else left

        return left_prompt, right

    def _generate_two_character_composite(
        self, request: ImageRequest, split_prompts: tuple[str, str]
    ) -> ImageResponse:
        left_prompt, right_prompt = split_prompts
        half_width = max(8, (request.width // 2 // 8) * 8)

        seed = request.seed
        if seed == -1:
            seed = int(torch.randint(0, 2147483647, (1,)).item())
            
        left_response = self._generate_single(
            replace(request, prompt=left_prompt, width=half_width, seed=seed, reference_image_url="")
        )
        right_response = self._generate_single(
            replace(
                request,
                prompt=right_prompt,
                width=request.width - half_width,
                seed=seed + 1,
                reference_image_url="",
            )
        )

        composite = Image.new("RGB", (request.width, request.height))
        composite.paste(left_response.image, (0, 0))
        composite.paste(right_response.image, (half_width, 0))
        return ImageResponse(image=composite, seed=seed)

    def generate(self, request: ImageRequest) -> ImageResponse:
        split_prompts = self._split_two_character_prompt(request.prompt)
        if split_prompts is not None:
            return self._generate_two_character_composite(request, split_prompts)
        return self._generate_single(request)

    def _generate_single(self, request: ImageRequest) -> ImageResponse:

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

                    # Determine style
                    clean_prompt, parsed_style = self._parse_style_from_prompt(request.prompt)
                    style_to_use = (request.style or parsed_style or "").strip().lower()

                    logger.info(
                        f"Generating image | "
                        f"Seed={seed} | "
                        f"Steps={request.steps} | "
                        f"Guidance={guidance_scale} | "
                        f"Style={style_to_use or 'default'}"
                    )
                    prompt = self._build_prompt(request.prompt, style_to_use)
                    negative_prompt = self._build_negative_prompt(
                        request.negative_prompt, style_to_use
                    )

                    reference_image = None
                    if self.settings.IP_ADAPTER_ENABLED:
                        real_reference = None
                        if request.reference_image_url:
                            real_reference = self._load_reference_image(request.reference_image_url)

                        if real_reference is not None:
                            reference_image = real_reference
                            self.pipeline.set_ip_adapter_scale(self.settings.IP_ADAPTER_SCALE)
                        else:
                            # Không có reference (panel đầu) hoặc tải lỗi — vẫn phải truyền
                            # ip_adapter_image (diffusers bắt buộc một khi đã load_ip_adapter),
                            # dùng ảnh trắng trơn + scale=0 để không ảnh hưởng thị giác.
                            reference_image = self._blank_ip_adapter_image()
                            self.pipeline.set_ip_adapter_scale(0.0)

                    image = self._run_inference(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        reference_image=reference_image,
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
