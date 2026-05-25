import os
import time
import asyncio
from concurrent import futures
import grpc
from fastapi import FastAPI
import uvicorn
import threading

from logger.config import get_logger
from service.image_service import ImageGenerationService

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
    return {"status": "healthy", "service": "image-ai", "timestamp": time.time()}

@app.get("/metrics")
def get_metrics():
    """Metrics phục vụ cho Prometheus giám sát hiệu năng."""
    # Bạn có thể đọc chỉ số GPU thực tế tại đây
    return {
        "active_gpu_tasks": 0,
        "service_status": "online"
    }

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
    host = os.getenv("HOST", "0.0.0.0")
    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    http_port = int(os.getenv("HTTP_PORT", "8000"))

    # Chạy FastAPI Health Server trên một luồng nền phụ (Daemon Thread)
    fastapi_thread = threading.Thread(
        target=start_fastapi_server, 
        args=(host, http_port), 
        daemon=True
    )
    fastapi_thread.start()

    # Khởi chạy gRPC Server trên luồng chính
    start_grpc_server(host, grpc_port)
