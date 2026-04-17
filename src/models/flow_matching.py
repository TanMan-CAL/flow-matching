# src/models/flow_matching.py
import torch
from torch import nn


def time_fm_forward(unet, real_images, num_timesteps):
    # num_timesteps not actually used during training, just keeping it for consistency
    unet.train()
    batch_size = real_images.shape[0]
    device = real_images.device

    random_times = torch.rand(batch_size, device=device)
    noise = torch.randn_like(real_images)

    # linear interp between noise and data
    times_expanded = random_times.view(batch_size, 1, 1, 1)
    noisy_images = times_expanded * real_images + (1 - times_expanded) * noise

    target_velocity = real_images - noise
    predicted_velocity = unet(noisy_images, random_times)
    
    return nn.functional.mse_loss(predicted_velocity, target_velocity)


@torch.inference_mode()
def time_fm_sample(unet, image_size, num_timesteps, seed=0):
    unet.eval()
    torch.manual_seed(seed)
    device = next(unet.parameters()).device

    generated = torch.randn(1, 3, *image_size, device=device)
    step_size = 1.0 / num_timesteps
    
    for step in range(num_timesteps):
        current_time = torch.full((1,), step / num_timesteps, device=device)
        generated = generated + step_size * unet(generated, current_time)  # euler step
    
    return generated


def class_fm_forward(unet, real_images, class_labels, uncond_dropout_prob, num_timesteps):
    unet.train()
    batch_size = real_images.shape[0]
    device = real_images.device

    random_times = torch.rand(batch_size, device=device)
    noise = torch.randn_like(real_images)
    times_expanded = random_times.view(batch_size, 1, 1, 1)
    noisy_images = times_expanded * real_images + (1 - times_expanded) * noise
    target_velocity = real_images - noise

    # drop conditioning randomly so model learns unconditional path too
    conditioning_mask = (torch.rand(batch_size, device=device) >= uncond_dropout_prob).long()
    predicted_velocity = unet(noisy_images, class_labels, random_times, mask=conditioning_mask)
    
    return nn.functional.mse_loss(predicted_velocity, target_velocity)


@torch.inference_mode()
def class_fm_sample(unet, class_labels, image_size, num_timesteps, guidance_scale=5.0, seed=0):
    unet.eval()
    torch.manual_seed(seed)
    device = next(unet.parameters()).device
    batch_size = class_labels.shape[0]

    generated = torch.randn(batch_size, 3, *image_size, device=device)
    step_size = 1.0 / num_timesteps

    use_conditioning = torch.ones(batch_size, device=device).long()
    skip_conditioning = torch.zeros(batch_size, device=device).long()

    intermediate_frames = []
    for step in range(num_timesteps):
        current_time = torch.full((batch_size,), step / num_timesteps, device=device)

        cond_velocity = unet(generated, class_labels, current_time, mask=use_conditioning)
        uncond_velocity = unet(generated, class_labels, current_time, mask=skip_conditioning)

        # cfg: push harder in the conditional direction
        guided_velocity = uncond_velocity + guidance_scale * (cond_velocity - uncond_velocity)
        generated = generated + step_size * guided_velocity

        if step % max(1, num_timesteps // 10) == 0:
            intermediate_frames.append(generated.clone())

    return generated, torch.stack(intermediate_frames, dim=1)