from dataclasses import dataclass
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

    def load_lora(
        self,
        pipeline,
        lora_path: str,
        adapter_name: str
    ):

        if self.is_loaded(adapter_name):
            logger.info(f"LoRA already loaded")
            return pipeline

        pipeline.load_lora_weights(
            lora_path,
            adapter_name=adapter_name
        )

        pipeline.set_adapters(adapter_name)
        self.loaded_adapters[adapter_name] = LoadedLora(
            adapter_name=adapter_name,
            path=lora_path
        )

        logger.info(f"Loaded LoRA: {adapter_name}")
        return pipeline

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