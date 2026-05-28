import time
import os
import asyncio
from concurrent import futures
import grpc
from fastapi import FastAPI, Response
import uvicorn
import threading

from config.settings import get_settings
from logger.config import get_logger
from service.image_service import ImageGenerationService
from storage import minio_storage_client
from cache.redis_cache import redis_cache_manager
from core.vram_manager import vram_manager
from core.pipeline_runner import pipeline_runner
from metrics import prometheus_response

# Import gRPC generated classes
try:
    from service.generated import image_generation_pb2_grpc
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'service', 'generated'))
    import image_generation_pb2_grpc

logger = get_logger(__name__)

# ----------------- FASTAPI SERVER (Health Check & Metrics) -----------------
app = FastAPI(title="Image AI Health Service", version="1.0.0")

@app.get("/healthz")
def health_check():
    """Endpoint giám sát trạng thái sức khỏe (Health Check) hệ thống."""
    # Check MinIO status
    minio_status = minio_storage_client.health_check()
    
    # Check Redis status
    redis_status = {"healthy": False}
    try:
        redis_cache_manager.initialize_client()
        if redis_cache_manager.client and redis_cache_manager.client.ping():
            redis_status = {"healthy": True, "message": "Redis connection successful"}
        else:
            redis_status = {"healthy": False, "error": "Redis ping failed"}
    except Exception as e:
        redis_status = {"healthy": False, "error": str(e)}

    overall_healthy = minio_status.get("healthy", False) and redis_status.get("healthy", False)
    gpu_status = vram_manager.get_gpu_memory_info()
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "service": "image-ai",
        "timestamp": time.time(),
        "dependencies": {
            "minio": minio_status,
            "redis": redis_status
        },
        "gpu": {
            "device": gpu_status.device_name,
            "is_gpu_available": gpu_status.is_gpu_available,
            "allocated_mb": gpu_status.allocated_mb,
            "reserved_mb": gpu_status.reserved_mb,
            "free_gpu_available": gpu_status.free_gpu_available,
        },
        "pipeline_ready": pipeline_runner.pipeline is not None,
    }

@app.get("/metrics")
def get_metrics():
    payload, content_type = prometheus_response()
    return Response(content=payload, media_type=content_type)

def start_fastapi_server(host: str, port: int):
    logger.info(f"Đang khởi động FastAPI Health Server tại http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="warning")

# ----------------- GRPC SERVER (Main Business Endpoint) -----------------
def start_grpc_server(host: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Đăng ký dịch vụ Image Generation vào gRPC Server
    image_generation_pb2_grpc.add_ImageGenerationServiceServicer_to_server(
        ImageGenerationService(), server
    )
    
    bind_address = f"{host}:{port}"
    server.add_insecure_port(bind_address)
    logger.info(f"Đang khởi động gRPC Server tại địa chỉ: {bind_address}...")
    
    server.start()
    logger.info("gRPC Server đã sẵn sàng nhận kết nối từ Orchestrator!")
    
    # Giữ luồng chính chạy liên tục để gRPC không bị ngắt
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Đang dừng gRPC Server...")
        server.stop(0)

# ----------------- MAIN INITIALIZATION -----------------
if __name__ == "__main__":
    settings = get_settings()
    host = settings.HOST
    grpc_port = settings.GRPC_PORT
    http_port = settings.HTTP_PORT

    # Chạy FastAPI Health Server trên một luồng nền phụ (Daemon Thread)
    fastapi_thread = threading.Thread(
        target=start_fastapi_server, 
        args=(host, http_port), 
        daemon=True
    )
    fastapi_thread.start()

    # Khởi chạy gRPC Server trên luồng chính
    start_grpc_server(host, grpc_port)
