import sys
import os
import time
import grpc
import requests

# Thêm thư mục gốc của dự án vào sys.path để tìm thấy package 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

try:
    from src.service.generated import image_generation_pb2
    from src.service.generated import image_generation_pb2_grpc
except ImportError as e:
    print(f"Lỗi: Không import được protobuf classes. Hãy chắc chắn bạn đã biên dịch proto bằng generate_proto.sh. Chi tiết: {e}")
    sys.exit(1)

def test_http_health():
    print("\n=== 1. KIỂM TRA FASTAPI HEALTH SERVER ===")
    try:
        response = requests.get("http://localhost:8000/healthz", timeout=3)
        if response.status_code == 200:
            print("✓ FastAPI Health Check: OK!")
            print(f"   Kết quả: {response.json()}")
        else:
            print(f"FastAPI Health Check: Lỗi (Status Code: {response.status_code})")
    except Exception as e:
        print(f"✗ FastAPI Health Check: Thất bại! Không kết nối được. Lỗi: {e}")

def test_grpc_service():
    print("\n=== 2. KIỂM TRA DỊCH VỤ GRPC ===")
    channel = grpc.insecure_channel("localhost:50051")
    stub = image_generation_pb2_grpc.ImageGenerationServiceStub(channel)
    
    # 2.1 Check Health gRPC
    try:
        print("Đang gửi yêu cầu gRPC CheckHealth...")
        health_resp = stub.CheckHealth(image_generation_pb2.CheckHealthRequest())
        print(f"✓ gRPC CheckHealth: OK!")
        print(f"   is_alive: {health_resp.is_alive}")
        print(f"   versions: {dict(health_resp.versions)}")
    except grpc.RpcError as e:
        print(f"✗ gRPC CheckHealth: Thất bại! (Code: {e.code()}, Details: {e.details()})")
        print("Hãy chắc chắn rằng server.py đang chạy.")
        return
        
    # 2.2 Generate Image
    try:
        prompt = "[style:retro] The family bird is singing a song on a tree branch"
        caption = "Gia đình nhà chim đang hót líu lo trên cành cây"
        print(f"\n=== 3. YÊU CẦU SINH ẢNH QUA GRPC ===")
        print(f"   Prompt: '{prompt}'")
        print(f"   Caption: '{caption}'")
        
        req = image_generation_pb2.GenerateImageRequest(
            prompt=prompt,
            width=512,          # Dùng kích thước nhỏ để sinh ảnh nhanh khi test
            height=512,
            seed=42,
            num_inference_steps=15,
            caption_text=caption,
            style="retro"       # Sử dụng trường style mới trong gRPC
        )
        
        resp = stub.GenerateImageAsync(req)
        task_id = resp.task_id
        print(f"✓ Đã gửi request thành công. Celery Task ID: {task_id}")
        
        # Poll status
        print("\nĐang theo dõi trạng thái task sinh ảnh (polling từ Redis qua gRPC)...")
        print("Hãy đảm bảo Celery Worker đang chạy để xử lý tác vụ này.")
        for i in range(180):  
            status_req = image_generation_pb2.TaskStatusRequest(task_id=task_id)
            status_resp = stub.GetTaskStatus(status_req)
            status_name = image_generation_pb2.ImageGenerationStatus.Name(status_resp.status)
            print(f"   [{i*2}s] Trạng thái: {status_name}")
            
            if status_resp.status == image_generation_pb2.PROCESSING:
                continue
            if status_resp.status == image_generation_pb2.SUCCESS:
                print(f"\nSINH ẢNH THÀNH CÔNG!")
                print(f"Đường dẫn MinIO Presigned URL:\n{status_resp.minio_url}")
                print("\nBạn hãy copy link trên và mở trong trình duyệt để xem kết quả nhé!")
                break
            elif status_resp.status == image_generation_pb2.FAILED:
                print(f"\n✗ SINH ẢNH THẤT BẠI! Lỗi từ worker: {status_resp.error_message}")
                break
                
            time.sleep(2)
        else:
            print("\n✗ Quá thời gian chờ (Timeout)!")
            
    except grpc.RpcError as e:
        print(f"✗ Lỗi kết nối gRPC khi sinh ảnh: {e.code()} - {e.details()}")
    finally:
        channel.close()

def test_cancel_task():
    print("\n=== 4. KIỂM TRA HỦY TASK (CANCELTASK) ===")
    channel = grpc.insecure_channel("localhost:50051")
    stub = image_generation_pb2_grpc.ImageGenerationServiceStub(channel)
    try:
        # Gửi request sinh ảnh
        print("Đang gửi yêu cầu sinh ảnh để hủy...")
        req = image_generation_pb2.GenerateImageRequest(
            prompt="[style:retro] A cool cat sitting on a futuristic motorcycle in cyberpunk city at night" ,
            width=512,
            height=512,
            seed=12,
            num_inference_steps=15,
            caption_text="Một con mèo ngồi trên chiếc xe máy tương lai trong cảnh thành phố cyberpunk về đêm"
        )
        resp = stub.GenerateImageAsync(req)
        task_id = resp.task_id
        print(f"✓ Đã gửi request thành công. Celery Task ID: {task_id}")
        
        # Hủy task lập tức
        print(f"Đang gửi lệnh hủy Task ID: {task_id}...")
        cancel_req = image_generation_pb2.CancelRequest(task_id=task_id)
        cancel_resp = stub.CancelTask(cancel_req)
        cancel_status_name = image_generation_pb2.ImageGenerationStatus.Name(cancel_resp.status)
        print(f"✓ gRPC CancelTask Response: Task ID: {cancel_resp.task_id}, Status: {cancel_status_name}")
        
        # Kiểm tra lại trạng thái xem task có bị hủy không
        time.sleep(1)
        status_req = image_generation_pb2.TaskStatusRequest(task_id=task_id)
        status_resp = stub.GetTaskStatus(status_req)
        status_name = image_generation_pb2.ImageGenerationStatus.Name(status_resp.status)
        print(f"   Trạng thái sau khi hủy: {status_name}")
        
    except grpc.RpcError as e:
        print(f"✗ Lỗi gRPC khi hủy task: {e.code()} - {e.details()}")
    finally:
        channel.close()

if __name__ == "__main__":
    test_http_health()
    test_grpc_service()
    test_cancel_task()
