dataset = MNIST('./data', train=False, download=True, transform=ToTensor())
clean, lbl = dataset[0]
clean = clean.unsqueeze(0) # (1,1,28,28)
sigmas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

fig, axes = plt.subplots(1, len(sigmas), figsize=(14, 2.5))
fig.suptitle(f'Noising process — digit {lbl}', fontsize=11)
for ax, s in zip(axes, sigmas):
    noisy = (clean + s * torch.randn_like(clean)).squeeze().clamp(0, 1)
    ax.imshow(noisy, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'σ={s}', fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
