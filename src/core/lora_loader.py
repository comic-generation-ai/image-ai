from dataclasses import dataclass
from pathlib import Path

import torch

from logger.config import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LoadedLora:
    adapter_name: str
    path: str


class LoraLoader:
    def __init__(self):
        self.loaded_adapters: dict[str, LoadedLora] = {}

    @property
    def total_loaded(self) -> int:
        return len(self.loaded_adapters)

    def is_loaded(self, adapter_name: str) -> bool:
        return adapter_name in self.loaded_adapters

    def inspect_format(self, lora_path: str) -> str:
        """
        Nhận diện nhanh format LoRA theo key trong safetensors để tránh load nhầm base model.
        """
        path = Path(lora_path)
        if path.is_dir():
            candidates = list(path.glob("*.safetensors"))
            path = candidates[0] if candidates else path

        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy LoRA: {lora_path}")

        if path.suffix != ".safetensors":
            return "unknown"

        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as f:
                first_keys = list(f.keys())[:50]
        except Exception as e:
            logger.warning(f"Không thể đọc metadata LoRA {lora_path}: {e}")
            return "unknown"

        if any(
            key.startswith(("lora_unet", "unet.", "text_encoder", "lora_te"))
            for key in first_keys
        ):
            return "sdxl"

        if any(key.startswith("diffusion_model.layers.") for key in first_keys):
            return "dit"

        if any("transformer_blocks" in key or "double_blocks" in key for key in first_keys):
            return "flux_or_sd3"

        return "unknown"

    def validate_compatibility(
        self,
        pipeline,
        lora_path: str,
        strict: bool = True,
    ) -> str:
        lora_format = self.inspect_format(lora_path)
        pipeline_name = pipeline.__class__.__name__.lower().replace("_", "")

        is_sdxl_pipeline = "stablediffusionxl" in pipeline_name
        # "stablediffusion" cũng là substring của "stablediffusionxlpipeline" — phải loại
        # trừ XL riêng, nếu không is_sd_pipeline sẽ luôn True kể cả khi pipeline là SDXL.
        is_sd_pipeline = "stablediffusion" in pipeline_name and not is_sdxl_pipeline

        incompatible = (
            lora_format in {"dit", "flux_or_sd3"} and (is_sdxl_pipeline or is_sd_pipeline)
        ) or (
            lora_format == "sdxl" and is_sd_pipeline
        )

        if incompatible:
            message = (
                f"LoRA '{lora_path}' có format '{lora_format}' nhưng pipeline hiện tại là "
                f"{pipeline.__class__.__name__}. LoRA này không tương thích với pipeline hiện tại; "
                "hãy dùng đúng LoRA cho kiến trúc model, hoặc đổi model cho khớp LoRA."
            )
            if strict:
                raise RuntimeError(message)
            logger.warning(message)

        return lora_format

    def load_lora(
        self,
        pipeline,
        lora_path: str,
        adapter_name: str,
        adapter_weight: float = 1.0,
        strict_compatibility: bool = True,
    ):

        with torch.inference_mode():
            if self.is_loaded(adapter_name):
                self._activate_adapter(pipeline, adapter_name, adapter_weight)
                logger.info(f"LoRA already loaded: {adapter_name}")
                return pipeline

            lora_format = self.validate_compatibility(
                pipeline=pipeline,
                lora_path=lora_path,
                strict=strict_compatibility,
            )

            pipeline.load_lora_weights(
                lora_path,
                adapter_name=adapter_name,
            )
            self._activate_adapter(pipeline, adapter_name, adapter_weight)
            self.loaded_adapters[adapter_name] = LoadedLora(
                adapter_name=adapter_name,
                path=lora_path,
            )

        logger.info(
            f"Loaded LoRA: {adapter_name} | path={lora_path} | "
            f"format={lora_format} | weight={adapter_weight}"
        )
        return pipeline

    @staticmethod
    def _activate_adapter(pipeline, adapter_name: str, adapter_weight: float) -> None:
        """Kích hoạt adapter trong inference_mode — tránh lỗi requires_grad sau warmup."""
        try:
            pipeline.set_adapters(adapter_name, adapter_weights=adapter_weight)
        except TypeError:
            pipeline.set_adapters(adapter_name)

    def unload_lora(
        self,
        pipeline,
        adapter_name: str
    ):

        if not self.is_loaded(adapter_name):
            return pipeline

        pipeline.delete_adapters(adapter_name)
        del self.loaded_adapters[adapter_name]
        logger.info(f"Unloaded LoRA: {adapter_name}")

        return pipeline

lora_loader = LoraLoader()
