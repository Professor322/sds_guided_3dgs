import torch
import torch.nn.functional as F
from torch import nn

from diffusers import DiffusionPipeline
from omegaconf import OmegaConf
from StableSR.scripts.sr_val_ddpm_text_T_vqganfin_oldcanvas import (
    load_model_from_config,
)
from StableSR.ldm.models.diffusion.ddpm import LatentDiffusionSRTextWT
from StableSR.ldm.models.autoencoder import AutoencoderKLResi
from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import Literal


@dataclass
class SDSLoss3DGS_StableSR(nn.Module):
    model: LatentDiffusionSRTextWT
    encoder: AutoencoderKLResi
    decode_images_every: int = 0
    render_dir: str = ""
    interpolation_type: Literal["bicubic", "bilinear"] = "bilinear"

    def __init__(
        self,
        interpolation_type: Literal["bicubic", "bilinear"],
        model_config_path: str,
        model_checkpoint_path: str,
        encoder_config_path: str = "",
        encoder_checkpoint_path: str = "",
        render_dir: str = "",
    ):
        super().__init__()
        self.interpolation_type = interpolation_type
        self.device = torch.device("cuda:0")
        model_config = OmegaConf.load(model_config_path)
        self.model = load_model_from_config(model_config, model_checkpoint_path)

        self.model.configs = model_config

        self.model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        self.model = self.model.to(self.device)
        self.tile_size = 64
        self.tile_overlap = 32
        # this is for debugging
        if (
            encoder_checkpoint_path != ""
            and encoder_config_path != ""
            and render_dir != ""
        ):
            vqgan_config = OmegaConf.load(encoder_config_path)
            self.encoder = load_model_from_config(vqgan_config, encoder_checkpoint_path)
            self.encoder = self.encoder.to(self.device)
            self.encoder.decoder.fusion_w = 0.5
            self.decode_images_every = 50
            self.render_dir = render_dir
            print("Warning: decoding of the samples enabled")

    @torch.autocast(device_type="cuda")
    def forward(
        self,
        render: torch.Tensor,
        condition: torch.Tensor,
        min_noise_step: int,
        max_noise_step: int,
        iteration: int,
    ):

        batch_size = render.size(0)

        # model works in [-1, 1] instead of [0, 1]
        render = 2.0 * render - 1.0
        condition = 2.0 * condition - 1.0

        assert render.requires_grad, "render has no gradient enabled"

        # resolution has to be a multiple of tile_overlap
        height, width = render.size(2), render.size(3)
        height, width = (
            height - height % self.tile_overlap,
            width - width % self.tile_overlap,
        )
        render = F.interpolate(
            render,
            (height, width),
            mode=self.interpolation_type,
        ).clamp(-1.0, 1.0)

        render_latent = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(render)
        )

        assert render_latent.requires_grad, "render_latent has no gradient enabled"

        text_init = [""] * batch_size
        semantic_c = self.model.cond_stage_model(text_init)

        noise = torch.randn_like(render_latent, device=self.device)

        t = torch.randint(
            min_noise_step,
            max_noise_step,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        # apply noise to the render, this is our X_t here
        noised_render_latent = self.model.q_sample(
            x_start=render_latent, t=t, noise=noise
        )

        assert (
            noised_render_latent.requires_grad
        ), "noised_render_latent has no gradient enabled"

        # upscale condition and encode it as well
        condition_upscaled = F.interpolate(
            condition,
            (height, width),
            mode=self.interpolation_type,
        ).clamp(-1.0, 1.0)
        condition_upscaled_latent = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(condition_upscaled)
        )

        # p_sample_canvas has torch.no_grad() inside, no need to have it here
        tile_weights = self.model._gaussian_weights(self.tile_size, self.tile_size, 1)
        _, denoised_render_latent = self.model.p_sample_canvas(
            x=noised_render_latent,
            c=semantic_c,
            struct_cond=condition_upscaled_latent,
            t=t,
            clip_denoised=True,
            return_x0=True,
            tile_size=self.tile_size,
            tile_weights=tile_weights,
            tile_overlap=self.tile_overlap,
            batch_size=batch_size,
        )
        # this is for debugging
        if self.decode_images_every > 0 and iteration % self.decode_images_every == 0:
            with torch.no_grad():
                _, enc_fea_lq = self.encoder.encode(condition_upscaled)
                x_samples = self.encoder.decode(
                    denoised_render_latent * 1.0 / self.model.scale_factor, enc_fea_lq
                )
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples:
                    x_sample = (
                        255.0 * x_sample.permute(1, 2, 0).detach().cpu().numpy()
                    ).astype(np.uint8)
                    Image.fromarray(x_sample).save(
                        f"{self.render_dir}/decoded_image_{iteration}.png"
                    )

        w = self.model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        loss_sds = (
            0.5
            * w
            * F.mse_loss(render_latent, denoised_render_latent, reduction="sum")
            / batch_size
        )
        return loss_sds


class SDSLoss3DGS(torch.nn.Module):
    def __init__(self, prompt=""):
        super().__init__()

        # load the model
        self.device = torch.device("cuda:0")

        # this is to make it work on the cluster as it does not have internet access to download the model
        deepfloyd_sr_model_path = "/home/nskochetkov/.cache/huggingface/hub/models--DeepFloyd--IF-II-L-v1.0/snapshots/609476ce702b2d94aff7d1f944dcc54d4f972901"
        pipe = DiffusionPipeline.from_pretrained(
            deepfloyd_sr_model_path,
            # text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)

        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.prompt_embeddings = pipe.encode_prompt(prompt)

    @torch.amp.autocast("cuda", enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states, **kwargs):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(torch.float16),
            t.to(torch.float16),
            encoder_hidden_states=encoder_hidden_states.to(torch.float16),
            **kwargs,
        ).sample.to(input_dtype)

    def prepare_latents(self, images):
        images = F.interpolate(
            images, (256, 256), mode="bilinear", align_corners=False, antialias=True
        )
        return 2.0 * images - 1.0

    def prepare_downscaled_latents(self, images, lowres_noise_level, downscale=False):
        if downscale:
            images = F.interpolate(
                images, (64, 64), mode="bilinear", align_corners=False, antialias=True
            )
        upscaled = F.interpolate(
            images, (256, 256), mode="bilinear", align_corners=False, antialias=True
        )
        upscaled = 2.0 * upscaled - 1.0
        upscaled = self.scheduler.add_noise(
            upscaled,
            torch.randn_like(upscaled),
            torch.tensor(int(self.num_train_timesteps * lowres_noise_level)),
        )
        return upscaled

    def construct_gradient(self, noise_pred, noise, t, guidance_scale):
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, _ = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        # w = w * (1 - w) ** 0.5
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        return grad

    def forward(
        self,
        images,
        original,
        min_step=20,
        max_step=980,
        guidance_scale=10.0,
        lowres_noise_level=0.75,
        scheduler_timestep=None,
        downscale_condition=False,
    ):
        # prepare images
        batch_size = images.shape[0]
        # positive and negative embeddings
        batch_embeddings = [
            self.prompt_embeddings[0].repeat(batch_size, 1, 1),
            self.prompt_embeddings[1].repeat(batch_size, 1, 1),
        ]

        latents = self.prepare_latents(images)

        condition = original
        condition = self.prepare_downscaled_latents(
            condition, lowres_noise_level, downscale=downscale_condition
        )
        noise_level = torch.full(
            [2 * condition.shape[0]],
            torch.tensor(int(self.num_train_timesteps * lowres_noise_level)),
            device=condition.device,
        )

        if scheduler_timestep is not None:
            t = scheduler_timestep * torch.ones(
                batch_size, dtype=torch.long, device=self.device
            )
        else:
            # sample ts
            t = torch.randint(
                min_step, max_step, [batch_size], dtype=torch.long, device=self.device
            )

        with torch.no_grad():
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            latents_noisy = torch.cat([latents_noisy, condition], dim=1)

            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)
            noise_pred = self.forward_unet(
                torch.cat(2 * [latents_noisy]),
                torch.cat(2 * [t]),
                torch.cat(batch_embeddings),
                class_labels=noise_level,
            )
        # convert noise prediction into gradient
        grad = self.construct_gradient(noise_pred, noise, t, guidance_scale)
        # compute surrogate loss
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return loss_sds


class SDILoss3DGS(SDSLoss3DGS):
    def forward(
        self,
        images,
        original=None,
        min_step=20,
        max_step=980,
        guidance_scale=10.0,
        lowres_noise_level=0.75,
        scheduler_timestep=None,
        stochastic_inversion=True,
        clip_x0=True,
        downscale_condition=False,
    ):
        # prepare images
        batch_size = images.shape[0]
        # positive and negative embeddings
        batch_embeddings = [
            self.prompt_embeddings[0].repeat(batch_size, 1, 1),
            self.prompt_embeddings[1].repeat(batch_size, 1, 1),
        ]

        self.lowres_noise_level = lowres_noise_level
        self.stochastic_inversion = stochastic_inversion
        self.clip_x0 = clip_x0
        latents = self.prepare_latents(images)

        if scheduler_timestep is not None:
            t = scheduler_timestep * torch.ones(1, dtype=torch.long, device=self.device)
        else:
            # sample ts
            t = torch.randint(
                min_step,
                max_step,
                [
                    1,
                ],
                dtype=torch.long,
                device=self.device,
            )

        # predict noise
        with torch.no_grad():
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.predict_noise(
                latents_noisy,
                t,
                batch_embeddings,
                original=original,
                guidance_scale=guidance_scale,
                lowres_noise_level=lowres_noise_level,
                downscale_condition=downscale_condition,
            )
            latents_denoised = self.get_x0(latents_noisy, noise_pred, t)

        w = ((1 - self.alphas[t]) * self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        self.debugging_stuff = {
            "latents": latents.detach(),
            "noise": noise.detach(),
            "latents_noisy": latents_noisy.detach(),
            "latents_denoised": latents_denoised.detach(),
            "t": t,
        }
        # TODO: This w leads to vanilla SDS. We can try other weighting strategies to improve the results.
        loss_sds = (
            0.5
            * w
            * F.mse_loss(latents, latents_denoised, reduction="sum")
            / batch_size
        )
        return loss_sds

    def get_x0(self, original_samples, noise_pred, t):
        alpha_prod_t = self.alphas[t]
        beta_prod_t = 1 - alpha_prod_t
        x0 = (original_samples - noise_pred * beta_prod_t ** (0.5)) / (
            alpha_prod_t ** (0.5)
        )
        if self.clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def predict_noise(
        self,
        latents_noisy,
        current_t,
        prompt_embeddings,
        original,
        guidance_scale,
        lowres_noise_level,
        downscale_condition,
    ):
        # Expand the latents if we are doing classifier free guidance
        batch_size = latents_noisy.shape[0]
        # TODO: hardcoded stuff
        condition = original
        condition = self.prepare_downscaled_latents(
            condition, lowres_noise_level, downscale=downscale_condition
        )
        condition = self.scheduler.scale_model_input(
            condition, current_t
        )  # here i changed from next_t to current_t
        noise_level = torch.full(
            [2 * condition.shape[0]],
            torch.tensor(int(self.num_train_timesteps * lowres_noise_level)),
            device=condition.device,
        )
        latents_noisy = torch.cat([latents_noisy, condition], dim=1)
        latent_model_input = torch.cat([latents_noisy] * 2)
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, current_t
        )  # here i changed from next_t to current_t
        noise_pred = self.forward_unet(
            latent_model_input,
            current_t
            * torch.ones(2 * batch_size, dtype=torch.long, device=self.device),
            torch.cat(prompt_embeddings),
            class_labels=noise_level,
        )
        # classifier guidance:
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred
