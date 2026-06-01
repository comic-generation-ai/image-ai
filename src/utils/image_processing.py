import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from config.settings import get_settings
from logger.config import get_logger

logger = get_logger(__name__)
settings = get_settings()

DEFAULT_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


@lru_cache(maxsize=16)
def _load_font(font_path: str | None, font_size: int) -> ImageFont.ImageFont:
    candidates = [font_path] if font_path else []
    candidates.extend(DEFAULT_FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, font_size)
            except Exception as e:
                logger.warning(f"Không thể load font {candidate}: {e}")
    return ImageFont.load_default()


def enhance_comic_image(image: Image.Image) -> Image.Image:
    """
    Tăng độ nét/màu nhẹ để ảnh đọc giống panel hoạt hình hơn mà không đổi bố cục.
    """
    if not settings.COMIC_POSTPROCESS_ENABLED:
        return image.convert("RGB")

    enhanced = image.convert("RGB")
    enhanced = ImageEnhance.Color(enhanced).enhance(settings.COLOR_BOOST)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(settings.CONTRAST_BOOST)
    enhanced = enhanced.filter(
        ImageFilter.UnsharpMask(
            radius=settings.SHARPEN_RADIUS,
            percent=settings.SHARPEN_PERCENT,
            threshold=settings.SHARPEN_THRESHOLD,
        )
    )
    return enhanced


def wrap_text(text: str, draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """
    Thuật toán tự động xuống dòng câu thoại dựa trên độ rộng thực tế của font chữ.
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = " ".join(current_line)
        # Sử dụng getbbox để tính toán độ rộng thực tế của chuỗi ký tự
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width > max_width:
            current_line.pop()
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines

def add_caption_to_comic(image: Image.Image, text: str, font_path: str = None) -> Image.Image:
    """
    Quy trình xử lý hậu kỳ Pillow vẽ khung truyện tranh và chèn thoại:
    1. Tạo viền đen cho tranh truyện.
    2. Vẽ hộp thoại màu trắng đục (trong suốt alpha).
    3. Render văn bản tiếng Việt Unicode (Roboto).
    """
    if not text:
        return image.convert("RGB") if image.mode != "RGB" else image

    logger.info(f"Đang tiến hành chèn lời thoại tiếng Việt: '{text}' vào ảnh...")

    try:
        # Bước 1: Tạo viền đen dày 15px cho bức ảnh bằng ImageOps
        image_with_border = ImageOps.expand(image, border=15, fill="black")
        width, height = image_with_border.size

        # Khởi tạo vẽ trên hệ màu RGBA để xử lý được kênh màu Alpha (độ trong suốt)
        overlay = Image.new("RGBA", image_with_border.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # Thiết lập font chữ (cache để tránh load lại mỗi task)
        font_size = 28
        font = _load_font(font_path, font_size)

        # Thiết lập kích thước tối đa của hộp thoại ở đáy ảnh
        padding = 20
        box_max_width = width - (padding * 4)

        # Gọi hàm tự động xuống dòng
        lines = wrap_text(text, draw_overlay, font, box_max_width)

        # Tính toán chiều cao cần thiết cho hộp thoại dựa trên số dòng chữ
        line_heights = []
        for line in lines:
            line_bbox = draw_overlay.textbbox((0, 0), line, font=font)
            line_heights.append(line_bbox[3] - line_bbox[1])
            
        total_text_height = sum(line_heights) + (len(lines) - 1) * 6
        box_height = total_text_height + (padding * 2)

        # Xác định tọa độ vẽ Hộp thoại ở đáy tranh
        box_left = padding * 2
        box_right = width - (padding * 2)
        box_bottom = height - padding - 15  # 15px viền đen
        box_top = box_bottom - box_height

        # Bước 2: Vẽ hộp thoại màu trắng đục có góc bo tròn (Alpha = 200/255)
        # Sử dụng rounded_rectangle có sẵn từ PIL>=9.0.0
        draw_overlay.rounded_rectangle(
            [box_left, box_top, box_right, box_bottom],
            radius=15,
            fill=(255, 255, 255, 200),  # Màu trắng đục
            outline=(0, 0, 0, 255),       # Viền đen mảnh bao quanh hộp thoại
            width=2
        )

        # Bước 3: Render chữ lên trên hộp thoại đục
        # Tạo bút vẽ trên ảnh chính để đè chữ sắc nét
        final_image = image_with_border.convert("RGBA")
        
        # Hợp nhất lớp phủ trong suốt (RGBA) lên ảnh chính
        combined_image = Image.alpha_composite(final_image, overlay)

        # Vẽ chữ trực tiếp
        draw_final = ImageDraw.Draw(combined_image)
        
        # Vẽ căn lề giữa cho từng dòng thoại
        current_y = box_top + padding
        for line, h in zip(lines, line_heights):
            line_bbox = draw_final.textbbox((0, 0), line, font=font)
            line_w = line_bbox[2] - line_bbox[0]
            # Tọa độ X để dòng chữ nằm chính giữa hộp thoại
            x_pos = box_left + (box_right - box_left - line_w) // 2
            
            # Vẽ chữ màu đen tuyền sắc nét
            draw_final.text((x_pos, current_y), line, font=font, fill=(0, 0, 0, 255))
            current_y += h + 6

        logger.info("Chèn lời thoại và viền truyện thành công!")
        return combined_image.convert("RGB")

    except Exception as e:
        logger.error(f"Lỗi hậu kỳ Pillow: {str(e)}")
        # Giữ ảnh gốc (không chỉ viền đen) để tránh MinIO nhận panel gần như toàn đen
        fallback = image.convert("RGB") if image.mode != "RGB" else image
        return fallback
