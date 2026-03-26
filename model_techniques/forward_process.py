def forward(image, t):
    alpha_bar = alphas_cumprod[t]
    noise = torch.randn_like(im) # sample random Gaussian noise
    im_noisy = torch.sqrt(alpha_bar) * image + torch.sqrt(1 - alpha_bar) * noise
    return image_noisy
