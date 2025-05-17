from configs import DatasetAugmentationConfig
import tyro
import os
import cv2
import torch.nn.functional as F
import torch
import tqdm
import torchvision.transforms as transforms
import copy
from StableSR.scripts.sr_val_ddpm_text_T_vqganfin_oldcanvas import (
    load_model_from_config,
    space_timesteps,
    adaptive_instance_normalization,
)
import numpy as np
from einops import repeat
from PIL import Image


def main(cfg: DatasetAugmentationConfig) -> None:
    input_dir = f"{cfg.data_dir}/images_{cfg.data_factor}"

    resulting_factor = int(cfg.data_factor // cfg.scale_factor)
    if resulting_factor > 1:
        output_dir = f"{cfg.data_dir}/images_{resulting_factor}_{cfg.upscale_type}"
    else:
        output_dir = f"{cfg.data_dir}/images_{cfg.upscale_type}"
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(input_dir)
    device = "cuda:0"
    transform = transforms.ToTensor()

    if cfg.upscale_type == "deepfloyd":
        from diffusers import DiffusionPipeline
        from diffusers.utils import pt_to_pil

        # load stage 2
        stage_2_path = "/home/nskochetkov/.cache/huggingface/hub/models--DeepFloyd--IF-II-L-v1.0/snapshots/609476ce702b2d94aff7d1f944dcc54d4f972901"
        stage_2 = DiffusionPipeline.from_pretrained(
            stage_2_path, variant="fp16", torch_dtype=torch.float16
        ).to(device)
        stage_2.enable_model_cpu_offload()
        prompt_embeds, negative_embeds = stage_2.encode_prompt(cfg.prompt)
        generator = torch.manual_seed(0)
    elif cfg.upscale_type == "stablesr":
        # basically copied from StableSR script
        from omegaconf import OmegaConf

        model_config = OmegaConf.load(cfg.stable_sr_config_path)
        model = load_model_from_config(model_config, cfg.stable_sr_checkpoint_path)
        model.configs = model_config
        # default stable diffusion schedule
        model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        model.num_timesteps = 1000
        sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = copy.deepcopy(
            model.sqrt_one_minus_alphas_cumprod
        )
        use_timesteps = set(space_timesteps(1000, [cfg.denoising_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(model.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        model.register_schedule(
            given_betas=np.array(new_betas), timesteps=len(new_betas)
        )
        model.num_timesteps = 1000
        model.ori_timesteps = list(use_timesteps)
        model.ori_timesteps.sort()
        model = model.to(device)
        vqgan_config = OmegaConf.load(cfg.encoder_config_path)
        encoder = load_model_from_config(vqgan_config, cfg.encoder_checkpoint_path)
        encoder = encoder.to(device)
        encoder.decoder.fusion_w = 0.5
        text_init = [""]
        semantic_c = model.cond_stage_model(text_init)

    with torch.no_grad():
        for image in tqdm.tqdm(images):
            image_input_full_path = os.path.join(input_dir, image)
            image_loaded = cv2.imread(image_input_full_path)
            image_loaded = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)
            output_image_path = os.path.join(output_dir, image)
            if cfg.upscale_type in ("bicubic", "bilinear"):
                image_interpolated = (
                    F.interpolate(
                        torch.from_numpy(image_loaded).permute(2, 0, 1).unsqueeze(0),
                        scale_factor=cfg.scale_factor,
                        mode=cfg.upscale_type,
                    )
                    .squeeze(0)
                    .clamp(0.0, 255.0)
                )
                image_interpolated = (
                    image_interpolated.permute(1, 2, 0).detach().cpu().numpy()
                )
                image_interpolated = image_interpolated.astype(np.uint8)

                image_interpolated = cv2.cvtColor(image_interpolated, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_image_path, image_interpolated)
            elif cfg.upscale_type == "deepfloyd":
                image_loaded = transform(image_loaded) * 2 - 1
                image_loaded = F.interpolate(
                    image_loaded.unsqueeze(0), (64, 64), mode="bicubic"
                ).clamp(-1.0, 1.0)
                image_interpolated = stage_2(
                    image=image_loaded,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt",
                    noise_level=cfg.condtion_noise_level_steps,
                ).images
                pil_image = pt_to_pil(image_interpolated)
                pil_image[0].save(output_image_path)
            elif cfg.upscale_type == "stablesr":
                batch_size = 1
                tile_size = 64
                tile_overlap = 32
                image_loaded = (transform(image_loaded) * 2 - 1).unsqueeze(0).to(device)
                # need to be a multiple of tile_overlap
                height, width = (
                    image_loaded.size(2) * cfg.scale_factor,
                    image_loaded.size(3) * cfg.scale_factor,
                )
                height, width = (
                    height - height % tile_overlap,
                    width - width % tile_overlap,
                )
                struct_cond = F.interpolate(
                    image_loaded,
                    (height, width),
                    mode="bicubic",
                ).clamp(-1.0, 1.0)
                struct_cond_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(struct_cond)
                )
                noise = torch.randn_like(struct_cond_latent)
                t = repeat(torch.tensor([999]), "1 -> b", b=struct_cond_latent.size(0))
                t = t.to(device).long()
                x_T = model.q_sample_respace(
                    x_start=struct_cond_latent,
                    t=t,
                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                    noise=noise,
                )

                samples, _ = model.sample_canvas(
                    cond=semantic_c,
                    struct_cond=struct_cond_latent,
                    batch_size=batch_size,
                    timesteps=cfg.denoising_steps,
                    time_replace=cfg.denoising_steps,
                    x_T=x_T,
                    return_intermediates=True,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    batch_size_sample=batch_size,
                )
                _, enc_fea_lq = encoder.encode(struct_cond)
                x_samples = encoder.decode(
                    samples * 1.0 / model.scale_factor, enc_fea_lq
                )
                x_samples = adaptive_instance_normalization(x_samples, struct_cond)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples:
                    x_sample = (
                        255.0 * x_sample.permute(1, 2, 0).detach().cpu().numpy()
                    ).astype(np.uint8)
                    Image.fromarray(x_sample).save(output_image_path)


if __name__ == "__main__":
    cfg = tyro.cli(DatasetAugmentationConfig)
    main(cfg)
