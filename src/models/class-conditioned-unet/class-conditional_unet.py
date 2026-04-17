import torch
from torch import nn
from .unet_blocks import ConvBlock, DownBlock, UpBlock, Flatten, Unflatten, FCBlock


class ClassConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, num_hiddens=128):
        super().__init__()
        hidden_dim = num_hiddens

        # encoder
        self.conv1 = ConvBlock(in_channels, hidden_dim)
        self.down1 = DownBlock(hidden_dim, hidden_dim*2)
        self.down2 = DownBlock(hidden_dim*2, hidden_dim*4)

        # bottleneck
        self.flatten = Flatten(kernel_size=8)
        self.unflatten = Unflatten(hidden_dim*4, kernel_size=8)

        # decoder
        self.up1 = UpBlock(hidden_dim*8, hidden_dim*2)
        self.up2 = UpBlock(hidden_dim*4, hidden_dim)
        self.up3 = ConvBlock(hidden_dim*2, hidden_dim)

        self.out_conv = nn.Conv2d(hidden_dim, in_channels, 3, 1, 1)

        # time conditioning projections
        self.time_proj_deep = FCBlock(1, hidden_dim*4)
        self.time_proj_mid = FCBlock(1, hidden_dim*2)

        # class conditioning projections
        self.class_proj_deep = FCBlock(num_classes, hidden_dim*4)
        self.class_proj_mid = FCBlock(num_classes, hidden_dim*2)

        self.num_classes = num_classes

    def forward(self, noisy_images, class_labels, timesteps, mask=None):
        assert noisy_images.shape[-2:] == (32, 32), f"expected 32x32 input, got {noisy_images.shape[-2:]}"
        batch_size = noisy_images.shape[0]
        device = noisy_images.device

        # one-hot encode class labels, zero out if mask says to drop conditioning
        class_onehot = torch.zeros(batch_size, self.num_classes, device=device)
        class_onehot.scatter_(1, class_labels.view(batch_size, 1), 1.0)
        if mask is not None:
            class_onehot = class_onehot * mask.view(batch_size, 1).float()

        timesteps_in = timesteps.view(batch_size, 1).float()

        # project time and class into spatial conditioning signals
        time_cond_deep = self.time_proj_deep(timesteps_in).view(batch_size, -1, 1, 1)
        time_cond_mid = self.time_proj_mid(timesteps_in).view(batch_size, -1, 1, 1)
        class_cond_deep = self.class_proj_deep(class_onehot).view(batch_size, -1, 1, 1)
        class_cond_mid = self.class_proj_mid(class_onehot).view(batch_size, -1, 1, 1)

        # encoder
        enc1 = self.conv1(noisy_images)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)

        # bottleneck + deep conditioning
        bottleneck = self.unflatten(self.flatten(enc3))
        bottleneck = class_cond_deep * bottleneck + time_cond_deep

        # decoder with skip connections + mid conditioning
        decoded = self.up1(torch.cat([bottleneck, enc3], dim=1))
        decoded = class_cond_mid * decoded + time_cond_mid
        decoded = self.up2(torch.cat([decoded, enc2], dim=1))
        decoded = self.up3(torch.cat([decoded, enc1], dim=1))

        return self.out_conv(decoded)