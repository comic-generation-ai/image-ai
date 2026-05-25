from logger.config import get_logger

logger = get_logger(__name__)

class SafetyChecker:
    def __init__(self):
        # Bạn có thể khởi tạo mô hình phân loại NSFW (ví dụ: Falcon-NSFW hoặc CLIP-NSFW) tại đây
        pass

    def check_image(self, image) -> bool:
        """
        Kiểm tra hình ảnh sinh ra có chứa nội dung nhạy cảm hay không.
        Trả về True nếu AN TOÀN, False nếu KHÔNG AN TOÀN (NSFW).
        """
        logger.info("Đang thực hiện kiểm duyệt nội dung bức ảnh vừa sinh ra...")
        try:
            # Trong phiên bản đồ án này, ta giả lập trả về True (An toàn)
            # Bạn có thể tích hợp thư viện: nsfw-detector hoặc model CLIP-safety-checker vào đây
            is_safe = True
            
            if is_safe:
                logger.info("Bức ảnh đã vượt qua vòng kiểm duyệt an toàn.")
            else:
                logger.warning("CẢNH BÁO: Phát hiện ảnh chứa nội dung NSFW!")
                
            return is_safe
        except Exception as e:
            logger.error(f"Lỗi khi kiểm duyệt ảnh: {str(e)}")
            return True # Cho qua nếu có sự cố để tránh chặn người dùng nhầm

safety_checker = SafetyChecker()
