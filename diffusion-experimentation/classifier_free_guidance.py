def iterative_denoise_cfg(im_noisy, i_start, prompt_embeds, 
                          uncond_prompt_embeds, timesteps, scale=7):
    for i in range(i_start, len(timesteps) - 1):
        # Run UNet TWICE: once with text prompt, once with epty prompt
        
        # Conditional: "a high quality photo of a dog"
        model_output = stage_1.unet(image, t,
                                    encoder_hidden_states=prompt_embeds,
                                    return_dict=False)[0]
        
        # Unconditional: "" (empty string)
        uncond_model_output = stage_1.unet(image, t,
                                           encoder_hidden_states=uncond_prompt_embeds,
                                           return_dict=False)[0]
        
        noise_est, predicted_variance = torch.split(model_output, ...)
        uncond_noise_est, _           = torch.split(uncond_model_output, ...)
        
        # CFG formula: push noise estimate toward conditional direction
        cfg_noise = uncond_noise_est + scale * (noise_est - uncond_noise_est)
        
        # Rest of denoising uses cfg_noise instead of noise_est
        x0_est = (image - torch.sqrt(1-alpha_cumprod) * cfg_noise) / torch.sqrt(alpha_cumprod)
        ...
