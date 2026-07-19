"""Sinh 1 ảnh chứa 2 nhân vật trong CÙNG 1 khung bằng kỹ thuật regional
diffusion (MultiDiffusion kiểu tiling, đơn giản hoá cho đúng 2 vùng trái/phải).

LỊCH SỬ 4 BẢN, ĐÃ TEST THẬT TRÊN GPU (SDXL, dreamshaper-xl, steps=25) — đọc kỹ
trước khi sửa tiếp, tránh lặp lại 3 hướng đã chứng minh tệ hơn bản này:

1) Full-shared-latent (2 lần gọi UNet trên CHUNG 1 latent, chỉ cắt noise
   OUTPUT theo nửa) → HỢP THỂ HOÀN TOÀN (mai rùa dính liền thân thỏ). Nguyên
   nhân: SDXL UNet dùng self-attention TOÀN CỤC ở các resolution thấp trong
   bottleneck — dù noise output bị cắt theo nửa, mỗi lần gọi UNet vẫn "nhìn
   thấy" (qua attention) toàn bộ latent kể cả nửa nhân vật kia, rò rỉ identity
   qua các bước denoise.

2) Bản HIỆN TẠI (đang dùng) — cắt latent thành 2 tensor CON VẬT LÝ RIÊNG BIỆT
   (trái/phải, overlap ~16% + feather blend tuyến tính ở giữa) SUỐT TOÀN BỘ
   quá trình, không có giai đoạn share latent nào. Tensor vật lý tách biệt tự
   động đảm bảo mỗi nhánh CHỈ có thể vẽ trong đúng nửa của mình (positional
   grounding miễn phí, không cần cơ chế gì thêm) — đã test cho kết quả 2 nhân
   vật tách biệt đúng giải phẫu, đúng vị trí trái/phải, cùng bối cảnh (khi
   prompt nêu rõ setting/action chung ở cả 2 vế). Giới hạn còn lại (đã biết,
   KHÔNG cố sửa tiếp): 2 nhánh độc lập hoàn toàn nên đôi khi tự chọn góc
   máy/độ cao mặt đất hơi khác nhau (vd 1 nhân vật lơ lửng cao hơn nhân vật
   kia) — đánh đổi chấp nhận được, xem 2 hướng đã thử và bỏ bên dưới.

3) Đã thử hybrid 2 pha (vài step đầu chạy full-shared-latent để "thoả thuận"
   bố cục trước khi tách crop) — ĐÃ TEST, TỆ HƠN: chỉ 30% step đầu share
   latent đã đủ để 2 hình dạng dính thành 1 khối (như bản 1), pha tách-crop
   sau đó chỉ tô chi tiết lên trên cái khung đã sai. Không có giai đoạn nào
   của quá trình denoise là "an toàn" để share NGUYÊN latent giữa 2 danh tính
   khác nhau. ĐÃ BỎ, không thử lại.

4) Đã thử self-attention masking (bỏ crop vật lý, quay lại full-frame cho cả
   2 nhánh nhưng patch attn1 của UNet để chặn attend xuyên nửa trái/phải,
   giữ nguyên convolution chạy xuyên suốt để có continuity bố cục) — ĐÃ TEST,
   RA LỖI KHÁC: hết hợp thể (đúng như lý thuyết) nhưng 2 nhân vật CHỒNG LẤN
   lên nhau (thỏ chạy đè lên mai rùa) thay vì đứng đúng trái/phải — vì mask
   chỉ chặn rò rỉ danh tính qua attention, KHÔNG neo vị trí không gian (thứ
   mà crop vật lý ở bản 2 tự động có được miễn phí). Convolution không bị
   chặn nên tự do "kéo" cả 2 chủ thể về gần nhau. ĐÃ BỎ, không thử lại — bản
   2 (crop vật lý) vẫn là bản duy nhất trong 4 lần thử cho kết quả dùng được
   thật sự, KHÔNG đầu tư thêm vào việc sửa lệch góc máy nữa.

Mỗi crop được gán đúng crops_coords_top_left/target_size (cơ chế
micro-conditioning có sẵn của SDXL, vốn sinh ra để dạy UNet "đây là 1 crop từ
ảnh lớn hơn, ở toạ độ nào") để biết đúng vị trí không gian của mình thay vì
tưởng đang vẽ toàn khung.

Rủi ro/giới hạn đã biết:
- Chỉ dùng được với SDXL (cần text_encoder_2, pooled embeddings, add_time_ids
  — SD1.5 không có các thành phần này). Caller (pipeline_runner.py) phải tự
  kiểm tra self.is_sdxl trước khi gọi vào đây.
- IP-Adapter đã cắm sẵn vào attention của UNet ngay lúc pipeline khởi tạo
  (load_ip_adapter() chạy 1 lần, không gỡ ra được) — UNet bắt buộc phải nhận
  đủ added_cond_kwargs["image_embeds"] mỗi lần gọi, kể cả khi không dùng
  reference thật. Dùng lại đúng cơ chế ảnh trắng + scale=0 mà
  pipeline_runner._generate_single() đã dùng cho panel không có reference.
  Embedding này không mang thông tin không gian nên dùng chung được cho cả 2
  crop, không cần tách theo vùng.
- VAE của SDXL tràn số (NaN) khi decode trực tiếp ở fp16 — đã xác nhận thật
  trên GPU (RuntimeWarning "invalid value encountered in cast" khi chưa có
  đoạn upcast bên dưới). Bắt buộc phải nâng VAE lên fp32 đúng lúc decode rồi
  hạ lại, y hệt StableDiffusionXLPipeline.__call__() làm — không được bỏ qua
  bước này dù chỉ viết vòng lặp tự tay. Nhớ hạ lại fp16 sau decode vì
  pipe.vae dùng chung với _generate_single() — không hạ lại sẽ làm mọi lần
  sinh ảnh 1-nhân-vật sau đó âm thầm chạy VAE ở fp32 (chậm hơn, tốn VRAM hơn).
- Chữ ký chính xác của pipe.encode_prompt() / pipe._get_add_time_ids() /
  pipe.prepare_ip_adapter_image_embeds() có thể lệch nhẹ giữa các version
  diffusers — đây là phần rủi ro nhất, dễ gãy nhất nếu diffusers version trên
  máy GPU khác với lúc viết file này.
"""

import torch
from PIL import Image, ImageDraw, ImageFilter


def _round_down_to_multiple_of_8(value: int) -> int:
    return max(8, (value // 8) * 8)


def generate_two_characters_same_frame(
    pipeline_runner,
    left_prompt: str,
    right_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[Image.Image, int]:
    """Trả về (ảnh PIL, seed đã dùng). Raise Exception nếu có bất kỳ bước nào
    thất bại — caller chịu trách nhiệm bắt lỗi và fallback, hàm này không tự
    fallback để giữ logic đơn giản, dễ debug khi có lỗi thật trên GPU.
    """
    pipe = pipeline_runner.pipeline
    device = pipeline_runner.device

    if seed == -1:
        seed = int(torch.randint(0, 2147483647, (1,)).item())

    with torch.inference_mode():
        left_embeds, left_neg_embeds, left_pooled, left_neg_pooled = pipe.encode_prompt(
            prompt=left_prompt,
            negative_prompt=negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        right_embeds, right_neg_embeds, right_pooled, right_neg_pooled = pipe.encode_prompt(
            prompt=right_prompt,
            negative_prompt=negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        generator = torch.Generator(device=pipeline_runner._generator_device()).manual_seed(seed)
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            left_embeds.dtype,
            device,
            generator,
        )
        latent_width = latents.shape[-1]
        half_width = latent_width // 2

        # Overlap giữa 2 crop để có vùng chuyển tiếp mượt, ~16% bề rộng latent,
        # làm tròn về bội số 8 vì UNet downsample latent thêm 1 lần nữa bên
        # trong (stride tổng /8) nên crop lệch bội số 8 dễ vỡ shape ở
        # skip-connection.
        overlap = _round_down_to_multiple_of_8(int(latent_width * 0.16))
        overlap = min(overlap, half_width - 8) if half_width > 8 else 0

        left_slice = slice(0, half_width + overlap)
        right_slice = slice(half_width - overlap, latent_width)

        # Feather weight tuyến tính thay vì average phẳng: w_left đi từ 1 (rìa
        # trái ngoài overlap) xuống 0 (rìa phải ngoài overlap), w_right là
        # phần bù (1 - w_left) — 2 nhánh chỉ thực sự "trộn 50/50" đúng tại tâm
        # dải overlap, càng ra rìa overlap càng thuộc hẳn về 1 nhánh.
        # Khởi tạo hard-split tại đúng tâm trước (an toàn cho trường hợp
        # overlap=0 — ảnh quá nhỏ) — rồi mới ghi đè dải overlap bằng ramp nếu
        # overlap > 0. Nếu khởi tạo bằng torch.ones() rồi lấy phần bù, trường
        # hợp overlap=0 sẽ khiến w_right_full = 0 TOÀN BỘ (nhánh phải mất
        # trắng), vì vòng if bị bỏ qua nên không có gì trừ bớt phía phải.
        w_left_full = torch.zeros(latent_width, device=device, dtype=latents.dtype)
        w_left_full[:half_width] = 1.0
        if overlap > 0:
            band_start = half_width - overlap
            band_end = half_width + overlap
            ramp = torch.linspace(1.0, 0.0, steps=2 * overlap, device=device, dtype=latents.dtype)
            w_left_full[band_start:band_end] = ramp
            w_left_full[band_end:] = 0.0
        w_right_full = 1.0 - w_left_full

        vae_scale_factor = pipe.vae_scale_factor  # 8 cho SDXL

        def _crop_add_time_ids(crop_slice: slice) -> torch.Tensor:
            crop_width_px = (crop_slice.stop - crop_slice.start) * vae_scale_factor
            crop_left_px = crop_slice.start * vae_scale_factor
            add_time_ids = pipe._get_add_time_ids(
                (height, width),          # original_size: canvas đầy đủ
                (0, crop_left_px),         # crops_coords_top_left: vị trí crop trong canvas
                (height, crop_width_px),   # target_size: kích thước thật của crop này
                dtype=left_embeds.dtype,
                text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
            ).to(device)
            return torch.cat([add_time_ids, add_time_ids])

        left_add_time_ids = _crop_add_time_ids(left_slice)
        right_add_time_ids = _crop_add_time_ids(right_slice)

        # Ảnh trắng + scale=0: giữ đúng hành vi "không có reference" hiện tại
        # của _generate_single() — IP-Adapter bắt buộc phải có image_embeds
        # một khi đã load_ip_adapter(), dù ta không dùng ảnh tham chiếu thật.
        pipe.set_ip_adapter_scale(0.0)
        ip_adapter_image_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=pipeline_runner._blank_ip_adapter_image(),
            ip_adapter_image_embeds=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        def _predict_noise_for_crop(t, crop_slice, add_time_ids, prompt_embeds, neg_embeds, pooled, neg_pooled):
            crop_latents = latents[..., crop_slice]
            latent_input = torch.cat([crop_latents] * 2)
            latent_input = pipe.scheduler.scale_model_input(latent_input, t)
            embeds = torch.cat([neg_embeds, prompt_embeds])
            add_text_embeds = torch.cat([neg_pooled, pooled])
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
                "image_embeds": ip_adapter_image_embeds,
            }
            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            return noise_uncond + guidance_scale * (noise_text - noise_uncond)

        for t in timesteps:
            noise_pred = torch.zeros_like(latents)

            noise_left = _predict_noise_for_crop(
                t, left_slice, left_add_time_ids,
                left_embeds, left_neg_embeds, left_pooled, left_neg_pooled,
            )
            noise_pred[..., left_slice] += noise_left * w_left_full[left_slice].view(1, 1, 1, -1)

            noise_right = _predict_noise_for_crop(
                t, right_slice, right_add_time_ids,
                right_embeds, right_neg_embeds, right_pooled, right_neg_pooled,
            )
            noise_pred[..., right_slice] += noise_right * w_right_full[right_slice].view(1, 1, 1, -1)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # SDXL VAE (fp16) tràn số khi decode trực tiếp ở fp16 — pipeline chuẩn
        # của diffusers luôn nâng VAE lên fp32 đúng lúc decode rồi hạ lại
        # (xem StableDiffusionXLPipeline.__call__), vòng lặp tự viết tay ở đây
        # phải tự làm lại đúng bước đó, nếu không ảnh ra sẽ toàn NaN → đen/rác
        # (đã xác nhận thật trên GPU trước khi thêm đoạn này).
        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float32)
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image_tensor = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False
        )[0]

        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)

        image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

    return image, seed


def generate_two_characters_via_inpaint(
    pipeline_runner,
    background_prompt: str,
    left_prompt: str,
    right_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[Image.Image, int]:
    """HƯỚNG THỬ NGHIỆM THỨ 5 — CHƯA TEST TRÊN GPU. Khác hẳn 4 bản trên (không
    tự viết vòng lặp denoise): dùng StableDiffusionXLInpaintPipeline có sẵn
    của diffusers (from_pipe() — dùng lại đúng unet/vae/text_encoder đã load,
    không tải lại model), 1 pipeline được HUẤN LUYỆN CHUYÊN cho việc "nhìn
    phần ảnh xung quanh mask, vẽ nội dung mới khớp góc máy/ánh sáng/mặt đất
    với phần đó" — đúng năng lực mà 4 bản tự viết tay ở trên đang thiếu.

    Quy trình 3 bước, TUẦN TỰ trên CÙNG 1 ảnh đang lớn dần (không phải 2 nhánh
    độc lập như 4 bản trên):
    1) Sinh nền bằng txt2img bình thường (không cần đúng chi tiết nhân vật,
       vùng 2 nhân vật sẽ bị vẽ đè hoàn toàn ở bước 2-3 nên nội dung nền ở
       đúng 2 vùng đó không quan trọng — chỉ ảnh hưởng tông màu/ánh sáng ban
       đầu).
    2) Inpaint rùa vào NỬA TRÁI — model nhìn thấy toàn bộ ảnh nền (kể cả nửa
       phải) làm ngữ cảnh.
    3) Inpaint thỏ vào NỬA PHẢI — model nhìn thấy ẢNH ĐÃ CÓ RÙA THẬT (không
       phải placeholder) làm ngữ cảnh, nên có cơ sở khớp góc máy/mặt đất với
       rùa tốt hơn nhiều so với 4 bản trên (nơi 2 nhánh không hề thấy kết quả
       thật của nhau, chỉ thấy prompt text).

    Rủi ro/giới hạn:
    - Chậm hơn cả 4 bản trên: 3 lần generate tuần tự (nền + 2 lần inpaint)
      thay vì 1 lần.
    - ĐÃ TEST TRÊN GPU 16GB, ĐÃ TÌM RA VÀ FIX 1 BUG THẬT: lần test đầu OOM
      ngay tại StableDiffusionXLInpaintPipeline.from_pipe(pipe) — tưởng là do
      GPU 16GB quá nhỏ, nhưng đọc source diffusers (pipeline_utils.py:2119)
      phát hiện from_pipe() MẶC ĐỊNH torch_dtype=torch.float32 nếu không
      truyền tường minh, rồi tự ép TOÀN BỘ model (UNet, VAE, 2 text encoder)
      lên fp32 ngay tại dòng đó — tăng gấp đôi VRAM cần dùng, không liên quan
      gì đến việc "wrap lại pipeline" như docstring của from_pipe() hứa hẹn
      ("without reallocating additional memory" — chỉ đúng nếu bạn tự truyền
      đúng torch_dtype). Đã fix bằng cách truyền torch_dtype=pipe.dtype tường
      minh — re-test sau fix: HẾT OOM, chạy xong cả 3 bước.
    - KẾT QUẢ THỊ GIÁC (đã test): hết hợp thể, bối cảnh/ánh sáng/mặt đất khớp
      tốt (tốt hơn bản crop-tiling ở khoản này, đúng như kỳ vọng vì inpaint
      thấy được ảnh thật của nhánh trước làm ngữ cảnh) — NHƯNG lệch TỶ LỆ
      nghiêm trọng: thỏ bị vẽ quá to (gần như tràn hết khung, đầu/tai chạm mép
      trên), rùa bị đẩy dồn về sát mép trái, gần như bị cắt hình. Mask 50/50
      diện tích không đồng nghĩa 2 nhân vật được vẽ với tỷ lệ cơ thể ngang
      nhau — mỗi lần inpaint tự quyết định "nhân vật to bao nhiêu trong vùng
      của nó" độc lập, không có ràng buộc tỷ lệ giữa 2 lần gọi.
    - ĐÁNH GIÁ SAU 6 LẦN THỬ TỔNG CỘNG (bản này + 4 bản đầu + hybrid): bản
      crop-tiling (generate_two_characters_same_frame, đang được
      pipeline_runner.py sử dụng) vẫn là bản duy nhất có chất lượng tổng thể
      tốt nhất, dù không hoàn hảo. Hàm generate_two_characters_via_inpaint
      này GIỮ LẠI trong file để tham khảo/thử tiếp sau này (vd thêm ràng buộc
      tỷ lệ qua prompt hoặc ControlNet), nhưng KHÔNG được wire vào
      pipeline_runner.py cho tới khi vấn đề tỷ lệ được giải quyết.
    - IP-Adapter (nếu bật) vẫn cần ip_adapter_image ở CẢ 3 lần gọi — dùng lại
      đúng cơ chế ảnh trắng + scale=0 như pipeline_runner._generate_single().
    - Mask được feather bằng Gaussian blur nhẹ ở biên (không hard-cut) — nếu
      vẫn thấy seam rõ, thử tăng feather_px hoặc strength.
    """
    from diffusers import StableDiffusionXLInpaintPipeline
    from core.vram_manager import vram_manager

    pipe = pipeline_runner.pipeline
    settings = pipeline_runner.settings

    if seed == -1:
        seed = int(torch.randint(0, 2147483647, (1,)).item())
    generator = torch.Generator(device=pipeline_runner._generator_device()).manual_seed(seed)

    ip_adapter_kwargs = {}
    if settings.IP_ADAPTER_ENABLED:
        pipe.set_ip_adapter_scale(0.0)
        ip_adapter_kwargs["ip_adapter_image"] = pipeline_runner._blank_ip_adapter_image()

    background = pipe(
        prompt=background_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **ip_adapter_kwargs,
    ).images[0]

    # Giải phóng cache PyTorch trước khi from_pipe() cần cấp phát thêm — GPU
    # 16GB đã hết OOM đúng ở bước này khi chưa có dòng dọn cache này.
    vram_manager.clear_cache()

    # QUAN TRỌNG: PHẢI truyền torch_dtype tường minh — from_pipe() mặc định
    # torch_dtype=torch.float32 nếu không truyền (xem diffusers
    # pipeline_utils.py:2119: "torch_dtype = kwargs.pop('torch_dtype',
    # torch.float32)"), rồi tự gọi new_pipeline.to(dtype=torch_dtype) ngay
    # sau đó — nếu pipe gốc đang fp16 mà không truyền dtype, nó sẽ ÉP TOÀN BỘ
    # MODEL (UNet, VAE, 2 text encoder) LÊN FP32, tức tăng gấp đôi VRAM cần
    # dùng ngay tại dòng này. Đã tự test và xác nhận đây chính là nguyên nhân
    # OOM trên GPU 16GB (không phải do GPU quá nhỏ) — thêm dòng dưới là đủ
    # fix, không cần giảm resolution hay tắt IP-Adapter.
    inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pipe(pipe, torch_dtype=pipe.dtype)

    half_width_px = _round_down_to_multiple_of_8(width // 2)
    feather_px = max(8, int(width * 0.04))

    def _make_mask(is_left: bool) -> Image.Image:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        if is_left:
            draw.rectangle([0, 0, half_width_px, height], fill=255)
        else:
            draw.rectangle([half_width_px, 0, width, height], fill=255)
        return mask.filter(ImageFilter.GaussianBlur(feather_px))

    after_left = inpaint_pipe(
        prompt=left_prompt,
        negative_prompt=negative_prompt,
        image=background,
        mask_image=_make_mask(is_left=True),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **ip_adapter_kwargs,
    ).images[0]

    vram_manager.clear_cache()

    final = inpaint_pipe(
        prompt=right_prompt,
        negative_prompt=negative_prompt,
        image=after_left,
        mask_image=_make_mask(is_left=False),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **ip_adapter_kwargs,
    ).images[0]

    return final, seed
