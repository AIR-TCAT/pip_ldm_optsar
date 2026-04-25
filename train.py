import os
import sys
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import transformers.modeling_utils
import transformers.utils.import_utils

def clean_check(*args, **kwargs):
    return True

transformers.modeling_utils.check_torch_load_is_safe = clean_check
transformers.utils.import_utils.check_torch_load_is_safe = clean_check


import math
import gc
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm

# Diffusers & Transformers
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers.optimization import get_scheduler

import bitsandbytes as bnb
import swanlab
from swanlab.integration.accelerate import SwanLabTracker


from dataset import OSDataset
from utils import calculate_metrics, calculate_fid_with_pytorch_fid, get_depth_map
from models import SARPhysicsLatentLoss, SemanticEnhancedTextEncoder, SemanticGuidedControlNetModel, setup_semantic_lora_for_unet



def log_validation(vae, text_encoder, tokenizer, unet, controlnet_wrapper, val_dataset, args, accelerator, epoch,
                   depth_processor, depth_estimator):
    print(f"Running validation for epoch {epoch}...")
    base_controlnet = accelerator.unwrap_model(controlnet_wrapper).controlnet

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        controlnet=base_controlnet,
        safety_checker=None,
        torch_dtype=torch.float32
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        pipeline.enable_xformers_memory_efficient_attention()

    indices = torch.randperm(len(val_dataset))[:min(args.num_validation_samples, len(val_dataset))]
    real_images, generated_images, comparison_images = [], [], []
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    for idx in indices:
        sample = val_dataset[idx]

        source_tensor = sample["source"]
        source_pil = transforms.ToPILImage()(source_tensor * 0.5 + 0.5)
        prompt = sample["prompt"]

        depth_map_tensor = get_depth_map([source_pil], depth_processor, depth_estimator, accelerator.device,
                                         args.resolution)
        depth_map_pil = transforms.ToPILImage()(depth_map_tensor.squeeze(0))

        with torch.autocast(accelerator.device.type):
            generated_pil = pipeline(
                prompt,
                image=depth_map_pil,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator
            ).images[0]

        generated_pil_gray = generated_pil.convert("L").convert("RGB")
        generated_tensor = transforms.ToTensor()(generated_pil_gray).unsqueeze(0)

        target_tensor_3ch = sample["target"].repeat(1, 1, 1) if sample["target"].shape[0] == 1 else sample["target"]
        target_pil = transforms.ToPILImage()(target_tensor_3ch * 0.5 + 0.5)

        real_images.append(target_tensor_3ch.unsqueeze(0) * 0.5 + 0.5)
        generated_images.append(generated_tensor)

        size = (256, 256)
        combined = Image.new('RGB', (size[0] * 4, size[1]))
        combined.paste(source_pil.resize(size), (0, 0))
        combined.paste(depth_map_pil.resize(size), (size[0], 0))
        combined.paste(generated_pil_gray.resize(size), (size[0] * 2, 0))
        combined.paste(target_pil.resize(size), (size[0] * 3, 0))

        comparison_images.append(swanlab.Image(combined, caption=f"Val: {sample['base_name']}"))

    if len(real_images) > 0:
        real_tensor = torch.cat(real_images).to(accelerator.device)
        gen_tensor = torch.cat(generated_images).to(accelerator.device)
        metrics = calculate_metrics(real_tensor, gen_tensor, accelerator.device)
        metrics['fid'] = calculate_fid_with_pytorch_fid(real_tensor, gen_tensor, accelerator.device)
        accelerator.log({
            "val_fid": metrics['fid'], "val_ssim": metrics['ssim'],
            "val_psnr": metrics['psnr'], "val_mse": metrics['mse'],
            "validation_comparison": comparison_images, "epoch": epoch
        })

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    swanlab_tracker = SwanLabTracker("Your project name.", experiment_name="Your experiment name.")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=swanlab_tracker,
        project_dir=args.logging_dir
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    controlnet = ControlNetModel.from_unet(unet)

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    semantic_controlnet = SemanticGuidedControlNetModel(controlnet)
    semantic_text_encoder = SemanticEnhancedTextEncoder(device=accelerator.device)

    sar_physics_loss = SARPhysicsLatentLoss(device=accelerator.device)

    if args.use_semantic_lora:
        unet = setup_semantic_lora_for_unet(unet)

    depth_image_processor = DPTImageProcessor.from_pretrained(args.depth_model_path)
    depth_estimator = DPTForDepthEstimation.from_pretrained(args.depth_model_path)
    depth_estimator = depth_estimator.to(accelerator.device)
    depth_estimator.eval()
    depth_estimator.requires_grad_(False)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    params_to_optimize = list(semantic_controlnet.parameters())
    if args.use_semantic_lora:
        params_to_optimize += list(unet.parameters())
    else:
        unet.requires_grad_(False)

    optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=args.learning_rate)

    train_dataset = OSDataset(args.train_json_path, size=args.resolution)

    val_dataset = OSDataset(args.val_json_path, size=args.resolution)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.max_train_steps
    )

    (semantic_controlnet, unet, optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
        semantic_controlnet, unet, optimizer, train_dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers("Your project name.", config=dict(vars(args)))

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        semantic_controlnet.train()
        if args.use_semantic_lora:
            unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(semantic_controlnet):
                target_img = batch["target"].to(dtype=torch.float32)
                if target_img.shape[1] == 1: target_img = target_img.repeat(1, 3, 1, 1)

                latents = vae.encode(target_img).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompts = batch["prompt"]
                text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt").to(accelerator.device)
                encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

                with torch.no_grad():
                    optical_images_tensor = batch["source"]
                    optical_pil_list = []
                    for t in optical_images_tensor:
                        t_norm = (t + 1) / 2.0
                        t_np = t_norm.permute(1, 2, 0).cpu().numpy()
                        t_pil = Image.fromarray((t_np * 255).astype(np.uint8))
                        optical_pil_list.append(t_pil)

                    control_depth = get_depth_map(optical_pil_list, depth_image_processor, depth_estimator,
                                                  accelerator.device, args.resolution)

                    semantic_embeddings_list = []
                    for p in prompts:
                        emb = semantic_text_encoder.encode_text_with_semantic_hierarchy(p)
                        semantic_embeddings_list.append(emb)
                    semantic_embeddings = torch.cat(semantic_embeddings_list, dim=0)

                control_depth = control_depth.detach()
                semantic_embeddings = semantic_embeddings.detach()
                torch.cuda.empty_cache()

                down_samples, mid_sample = semantic_controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    control_depth,
                    semantic_embeddings
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample
                )[0]

                loss_mse = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(bsz, 1, 1, 1).to(latents.device)
                beta_prod_t = 1 - alpha_prod_t
                pred_original_latents = (noisy_latents - beta_prod_t ** 0.5 * model_pred) / alpha_prod_t ** 0.5

                physics_losses = sar_physics_loss(pred_original_latents, latents)

                w_geo = 0.05
                w_scat = 0.1
                w_stat = 0.05
                w_sparse = 0.01
                w_topo = 0.01

                loss_physics = (
                        w_geo * physics_losses['loss_geo'] +
                        w_scat * physics_losses['loss_scat'] +
                        w_stat * physics_losses['loss_stat']
                )

                loss_reg = (
                        w_sparse * physics_losses['loss_sparse'] +
                        w_topo * physics_losses['loss_topo']
                )

                loss = loss_mse + loss_physics + loss_reg

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {
                        "train_loss": loss.item(),
                        "mse_loss": loss_mse.item(),
                        "phy_loss": loss_physics.item(),
                        "reg_loss": loss_reg.item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch + 1) % args.validation_epochs == 0:
                log_validation(vae, text_encoder, tokenizer, unet, semantic_controlnet, val_dataset, args, accelerator,
                               epoch + 1, depth_image_processor, depth_estimator)

            if (epoch + 1) % args.checkpointing_epochs == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped_net = accelerator.unwrap_model(semantic_controlnet)
                torch.save(unwrapped_net.state_dict(), os.path.join(save_path, "semantic_controlnet.pth"))
                unwrapped_net.controlnet.save_pretrained(os.path.join(save_path, "controlnet_base"))
                if args.use_semantic_lora:
                    accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet_lora"))
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--depth_model_path", type=str, help="Path to DPT model")

    parser.add_argument("--train_json_path", type=str,  help="Path to train_json")
    parser.add_argument("--val_json_path", type=str,  default="Path to val_json")

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_dir", type=str)

    parser.add_argument("--resolution", type=int)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--no_semantic_lora", action="store_false", dest="use_semantic_lora")
    parser.set_defaults(use_semantic_lora=True)

    parser.add_argument("--num_validation_samples", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=50)
    parser.add_argument("--checkpointing_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    swanlab.login(api_key="You api_key of swanlab.")

    main(args)