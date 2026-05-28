import gc
import asyncio
from dataclasses import dataclass, field
import torch
from logger.config import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class VramInfo:
    allocated_mb: float
    reserved_mb: float
    device_name: str

    status: str = "GPU is running"
    is_gpu_available: bool = True
    free_gpu_available: float = 0

@dataclass(slots=True)
class VramLock:
    allocated_mb: float
    free_gpu_available: float = 0


@dataclass(slots=True)
class VramStatus:
    total_gpu_available: float
    free_gpu_available: float
    allocated_mb: float
    reserved_mb: float
    device_name: str
    status: str
    is_gpu_available: bool
    allocated_locks: list[VramLock] = field(
        default_factory=list
    )


@dataclass(slots=True)
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
        self._gpu_lock = asyncio.Lock()
        self.config = VramManagerConfig()

    def get_lock(self) -> asyncio.Lock:
        return self._gpu_lock

    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def is_mps_available(self) -> bool:
        return torch.backends.mps.is_available()

    def is_gpu_available(self) -> bool:
        return (
            self.is_cuda_available()
            or self.is_mps_available()
        )

    def get_gpu_memory_info(self) -> VramInfo:
        if self.is_cuda_available():
            device = torch.cuda.current_device()
            allocated = (
                torch.cuda.memory_allocated(device)
                / (1024 ** 2)
            )

            reserved = (
                torch.cuda.memory_reserved(device)
                / (1024 ** 2)
            )

            total = (
                torch.cuda.get_device_properties(device)
                .total_memory
                / (1024 ** 2)
            )

            free = total - reserved

            return VramInfo(
                allocated_mb=round(allocated, 2),
                reserved_mb=round(reserved, 2),
                free_gpu_available=round(free, 2),
                device_name=torch.cuda.get_device_name(device),
                is_gpu_available=True
            )

        if self.is_mps_available():
            return VramInfo(
                allocated_mb=0,
                reserved_mb=0,
                free_gpu_available=0,
                device_name="Apple M-Series GPU",
                status="Apple Silicon (MPS) active",
                is_gpu_available=True
            )

        return VramInfo(
            allocated_mb=0,
            reserved_mb=0,
            free_gpu_available=0,
            device_name="CPU",
            status="Running on CPU",
            is_gpu_available=False
        )

    def get_vram_status(self) -> VramStatus:
        info = self.get_gpu_memory_info()
        total_gpu = (
            info.allocated_mb
            + info.free_gpu_available
        )

        return VramStatus(
            total_gpu_available=round(total_gpu, 2),
            free_gpu_available=info.free_gpu_available,
            allocated_mb=info.allocated_mb,
            reserved_mb=info.reserved_mb,
            device_name=info.device_name,
            status=info.status,
            is_gpu_available=info.is_gpu_available
        )

    def clear_cache(self):
        logger.info(
            "Starting RAM & VRAM cleanup..."
        )
        gc.collect()
        if self.is_cuda_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(
                "CUDA cache cleared successfully"
            )

        elif self.is_mps_available():
            torch.mps.empty_cache()
            logger.info(
                "MPS cache cleared successfully"
            )

vram_manager = VramManager()