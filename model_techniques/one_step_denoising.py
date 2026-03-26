# get noisy image through UNet to get noise estimate, not good for greater timesteps
noise_est = stage_1.unet(
    im_noisy, t,
    encoder_hidden_states=prompt_embeds,
    return_dict=False
)[0]

# first 3 channels is noise estimate, last 3 is variance
noise_est = noise_est[:, :3]

# rearrange forward equation to solve for x_0
x0_est = (im_noisy - torch.sqrt(1 - alpha_cumprod) * noise_est) / torch.sqrt(alpha_cumprod)
