import grpc
from celery.result import AsyncResult
from worker.celery_app import celery_app
from worker.tasks import generate_image_task
from core.vram_manager import vram_manager
from config.settings import get_settings
from logger.config import get_logger

try:
    from .generated import image_generation_pb2
    from .generated import image_generation_pb2_grpc
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
    import image_generation_pb2
    import image_generation_pb2_grpc

logger = get_logger(__name__)
settings = get_settings()

class ImageGenerationService(image_generation_pb2_grpc.ImageGenerationServiceServicer):
    
    def GenerateImageAsync(self, request, context):
        """
        Nhận request sinh ảnh từ gRPC -> Đẩy vào Celery Queue -> Phản hồi lập tức Task ID
        """
        logger.info(f"Đã nhận yêu cầu sinh ảnh gRPC từ Orchestrator | Prompt: '{request.prompt[:30]}...'")
        
        try:
            width = request.width if request.width > 0 else settings.DEFAULT_WIDTH
            height = request.height if request.height > 0 else settings.DEFAULT_HEIGHT
            
            # Tự động tối ưu số bước lặp (steps) cho các model thường (như dreamshaper-8)
            # Nếu client yêu cầu steps quá nhỏ (< 10) mà model không phải loại ít-step
            # (turbo/LCM vốn chạy chuẩn ở 4-6 steps), ghi đè bằng DEFAULT_STEPS để
            # đảm bảo chất lượng ảnh không bị xấu/nhiễu.
            model_id_lower = settings.MODEL_ID.lower()
            is_fast_model = "turbo" in model_id_lower or "lcm" in model_id_lower
            requested_steps = request.num_inference_steps
            if requested_steps > 0:
                if not is_fast_model and requested_steps < 10:
                    logger.warning(
                        f"Model {settings.MODEL_ID} không phải loại ít-step (turbo/LCM) nhưng nhận yêu cầu steps={requested_steps} quá thấp. "
                        f"Tự động điều chỉnh lên default steps={settings.DEFAULT_STEPS} để đảm bảo chất lượng."
                    )
                    steps = settings.DEFAULT_STEPS
                else:
                    steps = requested_steps
            else:
                steps = settings.DEFAULT_STEPS

            if width <= 0 or width > settings.MAX_WIDTH or width % 8 != 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"width phải trong khoảng 8..{settings.MAX_WIDTH} và chia hết cho 8")
                return image_generation_pb2.GenerateImageResponse()
            
            if height <= 0 or height > settings.MAX_HEIGHT or height % 8 != 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"height phải trong khoảng 8..{settings.MAX_HEIGHT} và chia hết cho 8")
                return image_generation_pb2.GenerateImageResponse()
            
            if steps < settings.MIN_STEPS or steps > settings.MAX_STEPS:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"num_inference_steps phải trong khoảng {settings.MIN_STEPS}..{settings.MAX_STEPS}")
                return image_generation_pb2.GenerateImageResponse()
            
            if len(request.caption_text or "") > settings.CAPTION_MAX_LENGTH:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"caption_text vượt quá {settings.CAPTION_MAX_LENGTH} ký tự")
                return image_generation_pb2.GenerateImageResponse()

            style = (request.style or "").strip()

            # Đẩy công việc vào Celery Worker xử lý bất đồng bộ
            # Concurrency=1 trên Worker đảm bảo GPU xử lý tuần tự không bị sập
            async_result = generate_image_task.delay(
                prompt=request.prompt,
                width=width,
                height=height,
                seed=request.seed,
                steps=steps,
                caption_text=request.caption_text,
                style=style,
                reference_image_url=(request.reference_image_url or "").strip(),
            )
            
            logger.info(f"Đã đẩy task vào queue hàng đợi thành công | Celery Task ID: {async_result.id}")
            
            # Trả kết quả gRPC ngay lập tức
            return image_generation_pb2.GenerateImageResponse(
                task_id=async_result.id,
                status=image_generation_pb2.PENDING
            )
            
        except Exception as e:
            logger.error(f"Lỗi khi đẩy task vào Celery: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Lỗi hệ thống hàng đợi: {str(e)}")
            return image_generation_pb2.GenerateImageResponse()

    def GetTaskStatus(self, request, context):
        """
        Kiểm tra trạng thái xử lý của Celery Task dựa trên Task ID
        """
        task_id = request.task_id
        logger.info(f"gRPC Kiểm tra trạng thái Task ID: {task_id}")
        
        try:
            async_result = AsyncResult(task_id, app=celery_app)
            status = async_result.status  # PENDING, STARTED, SUCCESS, FAILURE
            
            response = image_generation_pb2.TaskStatusResponse(task_id=task_id)
            
            if status == "SUCCESS":
                result = async_result.result 
                if result.get("status") == "SUCCESS":
                    response.status = image_generation_pb2.SUCCESS
                    response.minio_url = result.get("minio_url", "")
                    response.seed = int(result.get("seed", -1) or -1)
                else:
                    response.status = image_generation_pb2.FAILED
                    response.error_message = result.get("error_message", "Sinh ảnh lỗi trên GPU")
                    
            elif status == "FAILURE":
                response.status = image_generation_pb2.FAILED
                response.error_message = str(async_result.result)
                
            elif status == "PENDING" or status == "RECEIVED" or status == "RETRY":
                response.status = image_generation_pb2.PENDING
                
            elif status == "STARTED":
                response.status = image_generation_pb2.PROCESSING
                
            elif status == "REVOKED":
                response.status = image_generation_pb2.FAILED
                response.error_message = "Task đã bị hủy"
                
            else:
                response.status = image_generation_pb2.FAILED
                response.error_message = f"Trạng thái không xác định: {status}"
                
            return response
            
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn Celery Task {task_id}: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return image_generation_pb2.TaskStatusResponse()

    def CancelTask(self, request, context):
        """
        Hủy một Task sinh ảnh đang chờ hoặc đang chạy
        """
        task_id = request.task_id
        logger.info(f"Yêu cầu hủy Task ID: {task_id}")
        try:
            # Soft revoke trước để task chưa chạy sẽ bị chặn ngay.
            celery_app.control.revoke(task_id, terminate=False)
            # Nếu task đã bắt đầu chạy thì terminate để dừng sớm.
            celery_app.control.revoke(task_id, terminate=True)
            vram_manager.clear_cache()
            logger.info(f"Đã gửi lệnh hủy Task ID: {task_id} thành công")
            return image_generation_pb2.CancelResponse(
                task_id=task_id,
                status=image_generation_pb2.FAILED
            )
        except Exception as e:
            logger.error(f"Lỗi khi hủy Task ID {task_id}: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return image_generation_pb2.CancelResponse(
                task_id=task_id,
                status=image_generation_pb2.FAILED
            )

    def CheckHealth(self, request, context):
        """
        Kiểm tra sức khỏe dịch vụ gRPC
        """
        logger.info("gRPC Health check requested")
        return image_generation_pb2.CheckHealthResponse(
            is_alive=True,
            versions={"service": "image-ai", "version": "1.0.0"}
        )

    def CheckGpuHealth(self, request, context):
        info = vram_manager.get_gpu_memory_info()
        return image_generation_pb2.CheckGpuHealthResponse(
            is_healthy=info.is_gpu_available,
            gpu_name=info.device_name,
            gpu_memory_usage=str(info.allocated_mb),
            gpu_memory_free=str(info.free_gpu_available),
            gpu_memory_total=str(info.allocated_mb + info.free_gpu_available),
            gpu_memory_used=str(info.reserved_mb),
        )

    def ClearGpuCache(self, request, context):
        vram_manager.clear_cache()
        return image_generation_pb2.ClearGpuCacheResponse(is_cleared=True)
