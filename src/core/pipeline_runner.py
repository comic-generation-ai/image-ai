import torch
from diffusers import StableDiffusionXLPipeline
from config.settings import get_settings
from logger.config import get_logger

logger = get_logger(__name__)

class PipelineRunner:
    def __init__(self):
        self.settings = get_settings()
        self.model_id = self.settings.MODEL_ID
        self.low_vram_mode = self.settings.LOW_VRAM_MODE
        self.pipeline = None
        
        # Tự động nhận diện thiết bị phần cứng để tối ưu hóa hiệu năng
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
            logger.info("Phát hiện GPU NVIDIA CUDA. Chạy ở chế độ High-Performance.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16  # Apple Silicon hỗ trợ tốt float16 cho đa số mô hình
            logger.info("Phát hiện Apple Silicon GPU (MPS). Chạy tối ưu hóa nội bộ trên Mac.")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            logger.warning("Không tìm thấy GPU. Chạy trên CPU (Hiệu năng sẽ cực kỳ chậm!).")

    def initialize_pipeline(self):
        """Khởi tạo Stable Diffusion XL Pipeline với các cấu hình tối ưu bộ nhớ."""
        if self.pipeline is not None:
            return

        logger.info(f"Đang khởi tạo pipeline SDXL từ weights: {self.model_id}...")
        try:
            # Khởi tạo mô hình ở định dạng float16 hoặc float32 tùy thiết bị
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None
            )

            # Chuyển mô hình lên thiết bị phần cứng tối ưu nhất
            self.pipeline.to(self.device)

            # Áp dụng các tối ưu hóa phần cứng được thiết lập trong đồ án
            logger.info("Đang áp dụng các kỹ thuật tối ưu hóa bộ nhớ GPU...")
            
            # Tối ưu hóa VAE (Chỉ bật khi không chạy trên CPU)
            if self.device != "cpu":
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_vae_tiling()

            # Bật CPU Offloading nếu ở chế độ VRAM thấp (Chỉ hỗ trợ trên CUDA)
            if self.low_vram_mode and self.device == "cuda":
                logger.info("Chế độ LOW_VRAM_MODE được kích hoạt. Đang bật Model CPU Offloading...")
                self.pipeline.enable_model_cpu_offload()

            logger.info(f"Pipeline SDXL đã được khởi tạo và tối ưu hóa thành công trên [{self.device.upper()}]!")
        except Exception as e:
            logger.error(f"Khởi tạo Pipeline thất bại: {str(e)}")
            raise e

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = -1, steps: int = 8):
        """
        Thực hiện sinh ảnh từ prompt.
        Sử dụng Lykon/dreamshaper-xl-v2-turbo tối ưu tốc độ sinh ảnh (chỉ cần 4-8 steps).
        """
        self.initialize_pipeline()
        
        logger.info(f"Đang sinh ảnh với prompt: '{prompt}' | Steps: {steps} | Size: {width}x{height} | Device: {self.device}")
        
        # Thiết lập seed để kiểm soát tính nhất quán
        generator = None
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            # Chạy inference
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=2.0,  # SDXL Turbo yêu cầu guidance_scale thấp (1.0 - 2.0)
                    generator=generator
                )
            
            image = result.images[0]
            logger.info(f"Sinh ảnh thành công trên {self.device.upper()}!")
            return image
        except Exception as e:
            logger.error(f"Lỗi xảy ra trong quá trình sinh ảnh trên {self.device.upper()}: {str(e)}")
            raise e

# Khởi tạo instance singleton của runner
pipeline_runner = PipelineRunner()
