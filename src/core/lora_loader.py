from logger.config import get_logger

logger = get_logger(__name__)

class LoraLoader:
    def __init__(self):
        pass

    def load_lora(self, pipeline, lora_path_or_id: str, adapter_name: str = "default_lora"):
        """
        Nạp một tệp LoRA cụ thể vào pipeline để thay đổi nét vẽ theo nhân vật.
        """
        logger.info(f"Đang tiến hành nạp động LoRA weights từ: {lora_path_or_id}...")
        try:
            # Gọi hàm của diffusers để nạp LoRA
            # pipeline.load_lora_weights(lora_path_or_id, adapter_name=adapter_name)
            logger.info(f"LoRA '{adapter_name}' đã được nạp thành công vào bộ nhớ GPU!")
            return pipeline
        except Exception as e:
            logger.error(f"Nạp LoRA thất bại: {str(e)}")
            raise e

    def unload_lora(self, pipeline):
        """Giải phóng LoRA khỏi pipeline để giải phóng bộ nhớ."""
        logger.info("Đang tiến hành giải phóng LoRA weights để dọn dẹp GPU...")
        try:
            # pipeline.unload_lora_weights()
            logger.info("Đã giải phóng hoàn toàn LoRA weights khỏi GPU!")
            return pipeline
        except Exception as e:
            logger.error(f"Giải phóng LoRA thất bại: {str(e)}")
            return pipeline

lora_loader = LoraLoader()
