# src/models/wrapper.py
import torch
from flow_matching import *

class TimeConditionalFM(nn.Module):
    def __init__(self, unet, num_timesteps=50, image_size=(32, 32)):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.image_size = image_size

    def forward(self, real_images):
        return time_fm_forward(self.unet, real_images, self.num_timesteps)

    @torch.inference_mode()
    def sample(self, image_size, seed=0):
        return time_fm_sample(self.unet, image_size, self.num_timesteps, seed)


class ClassConditionalFM(nn.Module):
    def __init__(self, unet, num_timesteps=300, uncond_dropout_prob=0.1):
        super().__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.uncond_dropout_prob = uncond_dropout_prob

    def forward(self, real_images, class_labels):
        return class_fm_forward(self.unet, real_images, class_labels, self.uncond_dropout_prob, self.num_timesteps)

    @torch.inference_mode()
    def sample(self, class_labels, image_size, guidance_scale=5.0, seed=0):
        return class_fm_sample(self.unet, class_labels, image_size, self.num_timesteps, guidance_scale, seed)