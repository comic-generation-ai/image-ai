import grpc
from celery.result import AsyncResult
from worker.celery_app import celery_app
from worker.tasks import generate_image_task
from logger.config import get_logger

# Import các module được sinh ra tự động từ file .proto
try:
    from .generated import image_generation_pb2
    from .generated import image_generation_pb2_grpc
except ImportError:
    # Fallback cho chạy local khi chưa chạy scripts compile
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'generated'))
    import image_generation_pb2
    import image_generation_pb2_grpc

logger = get_logger(__name__)

class ImageGenerationService(image_generation_pb2_grpc.ImageGenerationServiceServicer):
    
    def GenerateImageAsync(self, request, context):
        """
        Nhận request sinh ảnh từ gRPC -> Đẩy vào Celery Queue -> Phản hồi lập tức Task ID
        """
        logger.info(f"Đã nhận yêu cầu sinh ảnh gRPC từ Orchestrator | Prompt: '{request.prompt[:30]}...'")
        
        try:
            # Đẩy công việc vào Celery Worker xử lý bất đồng bộ
            # Concurrency=1 trên Worker đảm bảo GPU xử lý tuần tự không bị sập
            async_result = generate_image_task.delay(
                prompt=request.prompt,
                width=request.width if request.width > 0 else 1024,
                height=request.height if request.height > 0 else 1024,
                seed=request.seed,
                steps=request.num_inference_steps if request.num_inference_steps > 0 else 8,
                caption_text=request.caption_text
            )
            
            logger.info(f"Đã đẩy task vào queue hàng đợi thành công | Celery Task ID: {async_result.id}")
            
            # Trả kết quả gRPC ngay lập tức
            return image_generation_pb2.GenerateImageResponse(
                task_id=async_result.id,
                status="PENDING"
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
            # Lấy thông tin Celery Task từ Redis Backend
            async_result = AsyncResult(task_id, app=celery_app)
            status = async_result.status  # PENDING, STARTED, SUCCESS, FAILURE
            
            response = image_generation_pb2.TaskStatusResponse(task_id=task_id)
            
            if status == "SUCCESS":
                result = async_result.result  # Kết quả trả về của task
                if result.get("status") == "SUCCESS":
                    response.status = "SUCCESS"
                    response.minio_url = result.get("minio_url", "")
                else:
                    response.status = "FAILED"
                    response.error_message = result.get("error_message", "Sinh ảnh lỗi trên GPU")
                    
            elif status == "FAILURE":
                response.status = "FAILED"
                response.error_message = str(async_result.result)
                
            elif status == "PENDING" or status == "RECEIVED" or status == "RETRY":
                response.status = "PENDING"
                
            elif status == "STARTED":
                response.status = "PROCESSING"
                
            elif status == "REVOKED":
                response.status = "CANCELLED"
                
            else:
                response.status = status
                
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
            # Thu hồi task (revoke) trong Celery
            # terminate=True để ngắt tiến trình nếu task đang chạy
            celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Đã gửi lệnh hủy Task ID: {task_id} thành công")
            return image_generation_pb2.CancelResponse(
                task_id=task_id,
                status="CANCELLED"
            )
        except Exception as e:
            logger.error(f"Lỗi khi hủy Task ID {task_id}: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return image_generation_pb2.CancelResponse(
                task_id=task_id,
                status="FAILED"
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

