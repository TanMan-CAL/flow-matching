# guassian blur: averages neighboring pixels, which does remove some noise but also destroys fine details and edges
blurred = TF.gaussian_blur(noisy, kernel_size=5, sigma=3)



