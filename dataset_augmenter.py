from configs import DatasetAugmentationConfig
import tyro
import os
import cv2
import torch.nn.functional as F
import torch
import tqdm


def main(cfg: DatasetAugmentationConfig) -> None:
    input_dir = f"{cfg.data_dir}/images_{cfg.data_factor}"

    resulting_factor = int(cfg.data_factor // cfg.scale_factor)
    if resulting_factor > 1:
        output_dir = f"{cfg.data_dir}/images_{resulting_factor}_{cfg.upscale_type}"
    else:
        output_dir = f"{cfg.data_dir}/images_{cfg.upscale_type}"
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(input_dir)

    if cfg.upscale_type == "sr":
        from diffusers import DiffusionPipeline
        from diffusers.utils import pt_to_pil
        import torchvision.transforms as transforms

        # load stage 2
        stage_2_path = "/home/nskochetkov/.cache/huggingface/hub/models--DeepFloyd--IF-II-L-v1.0/snapshots/609476ce702b2d94aff7d1f944dcc54d4f972901"
        stage_2 = DiffusionPipeline.from_pretrained(
            stage_2_path, variant="fp16", torch_dtype=torch.float16
        ).to("cuda")
        stage_2.enable_model_cpu_offload()
        prompt_embeds, negative_embeds = stage_2.encode_prompt(cfg.prompt)
        generator = torch.manual_seed(0)
        transform = transforms.ToTensor()

    for image in tqdm.tqdm(images):
        image_input_full_path = os.path.join(input_dir, image)
        image_loaded = cv2.imread(image_input_full_path)
        image_loaded = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)
        output_image_path = os.path.join(output_dir, image)
        if cfg.upscale_type != "sr":
            image_interpolated = F.interpolate(
                torch.from_numpy(image_loaded).permute(2, 0, 1).unsqueeze(0),
                scale_factor=cfg.scale_factor,
                mode=cfg.upscale_type,
            ).squeeze(0)
            image_interpolated = (
                image_interpolated.permute(1, 2, 0).detach().cpu().numpy()
            )

            image_interpolated = cv2.cvtColor(image_interpolated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, image_interpolated)
        else:
            image_loaded = transform(image_loaded) * 2 - 1
            image_loaded = F.interpolate(
                image_loaded.unsqueeze(0), (64, 64), mode="bilinear"
            )
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


if __name__ == "__main__":
    cfg = tyro.cli(DatasetAugmentationConfig)
    main(cfg)
