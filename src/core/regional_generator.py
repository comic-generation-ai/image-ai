"""Sinh 1 ảnh chứa 2 nhân vật trong CÙNG 1 khung bằng kỹ thuật regional
diffusion (giống MultiDiffusion, đơn giản hoá cho đúng 2 vùng trái/phải tĩnh).

Khác với cách ghép-2-ảnh-đã-sinh-xong (đã bỏ vì luôn lộ đường ranh giới cứng):
ở đây CHỈ CÓ 1 latent duy nhất, tiến hoá xuyên suốt toàn bộ quá trình khử
nhiễu. Mỗi bước, UNet được hỏi 2 lần (1 lần theo prompt nhân vật trái, 1 lần
theo prompt nhân vật phải) trên CÙNG latent hiện tại, rồi noise dự đoán được
lấy đúng nửa trái từ lần hỏi trái, nửa phải từ lần hỏi phải, trước khi bước
scheduler. Vì latent luôn chung và VAE/UNet là mạng tích chập (receptive field
tràn qua ranh giới), ảnh cuối không có đường nối cứng như cách dán ảnh.

Rủi ro/giới hạn đã biết (chưa test được trên GPU thật khi viết file này):
- Chỉ dùng được với SDXL (cần text_encoder_2, pooled embeddings, add_time_ids
  — SD1.5 không có các thành phần này). Caller (pipeline_runner.py) phải tự
  kiểm tra self.is_sdxl trước khi gọi vào đây.
- IP-Adapter đã cắm sẵn vào attention của UNet ngay lúc pipeline khởi tạo
  (load_ip_adapter() chạy 1 lần, không gỡ ra được) — UNet bắt buộc phải nhận
  đủ added_cond_kwargs["image_embeds"] mỗi lần gọi, kể cả khi không dùng
  reference thật. Dùng lại đúng cơ chế ảnh trắng + scale=0 mà
  pipeline_runner._generate_single() đã dùng cho panel không có reference.
- Chữ ký chính xác của pipe.encode_prompt() / pipe._get_add_time_ids() /
  pipe.prepare_ip_adapter_image_embeds() có thể lệch nhẹ giữa các version
  diffusers — đây là phần rủi ro nhất, dễ gãy nhất nếu diffusers version trên
  máy GPU khác với lúc viết file này.
"""

import torch
from PIL import Image


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
        half_width_latent = latents.shape[-1] // 2

        add_time_ids = pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=left_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device)
        add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids])

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

        def _predict_noise(t, prompt_embeds, neg_embeds, pooled, neg_pooled):
            latent_input = torch.cat([latents] * 2)
            latent_input = pipe.scheduler.scale_model_input(latent_input, t)
            embeds = torch.cat([neg_embeds, prompt_embeds])
            add_text_embeds = torch.cat([neg_pooled, pooled])
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids_cfg,
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
            noise_left = _predict_noise(t, left_embeds, left_neg_embeds, left_pooled, left_neg_pooled)
            noise_right = _predict_noise(t, right_embeds, right_neg_embeds, right_pooled, right_neg_pooled)

            noise_pred = torch.zeros_like(noise_left)
            noise_pred[..., :half_width_latent] = noise_left[..., :half_width_latent]
            noise_pred[..., half_width_latent:] = noise_right[..., half_width_latent:]

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        image_tensor = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

    return image, seed
