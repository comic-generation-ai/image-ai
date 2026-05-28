import logging
import sys

def get_logger(name: str):
    """Cấu hình Logging chuẩn hóa dễ dàng theo dõi hiệu năng hệ thống."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format ghi log có chi tiết thời gian chạy
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Ghi log ra màn hình Console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        
    return logger
