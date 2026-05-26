import gc
import torch
import asyncio
from logger.config import get_logger
from dataclasses import dataclass

logger = get_logger(__name__)


@dataclass
class VramInfo:
    allocated_mb: float
    reserved_mb: float
    device_name: str
    status: str = "GPU is running"
    is_gpu_available: bool = True
    free_gpu_available: float = 0
    
@dataclass
class VramLock:
    allocated_mb: float
    free_gpu_available: float = 0
    
@dataclass
class VramStatus:
    total_gpu_available: float
    free_gpu_available: float
    allocated_mb: float
    reserved_mb: float
    device_name: str
    status: str
    is_gpu_available: bool
    allocated_locks: list[VramLock] = []

@dataclass
class VramManagerConfig:
    max_gpu_locks: int = 1
    max_gpu_memory_usage: float = 0.9
    max_gpu_memory_usage_reserved: float = 0.9
    max_gpu_memory_usage_allocated: float = 0.9
    max_gpu_memory_usage_free: float = 0.9
    max_gpu_memory_usage_total: float = 0.9
    max_gpu_memory_usage_used: float = 0.9



class VramManager:
    def __init__(self):
        # Khóa Lock để đảm bảo chỉ có tối đa 1 task sinh ảnh được dùng GPU tại một thời điểm
        self._gpu_lock = asyncio.Lock()

    def get_lock(self) -> asyncio.Lock:
        """Trả về khóa Lock của GPU để dùng trong các khối lệnh async with"""
        return self._gpu_lock

    def get_gpu_memory_info(self):
        """Lấy thông tin sử dụng bộ nhớ VRAM hiện tại"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
            cached = torch.cuda.memory_reserved(device) / (1024 ** 2)      # MB
            return {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(cached, 2),
                "device_name": torch.cuda.get_device_name(device)
            }
        elif torch.backends.mps.is_available():
            return {
                "status": "Apple Silicon (MPS) hoạt động",
                "device_name": "Apple M-Series GPU"
            }
        else:
            return {"status": "Chạy bằng CPU"}

    def clear_cache(self):
        """Giải phóng bộ nhớ VRAM rác của PyTorch"""
        logger.info("Đang bắt đầu dọn dẹp bộ nhớ RAM & VRAM...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Đã dọn dẹp CUDA Cache thành công!")
            logger.info(f"VRAM sau dọn dẹp: {self.get_gpu_memory_info()}")
        elif torch.backends.mps.is_available():
            import torch.mps
            torch.mps.empty_cache()
            logger.info("Đã dọn dẹp MPS Cache thành công!")


# Singleton VramManager
vram_manager = VramManager()
