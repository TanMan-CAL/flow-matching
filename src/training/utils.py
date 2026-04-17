# src/training/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(7)

    def forward(self, x):
        return F.gelu(self.pool(x))

class Unflatten(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 7, stride=7)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return F.gelu(self.bn(self.up(x)))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = Conv(in_channels, out_channels)
        self.c2 = Conv(out_channels, out_channels)

    def forward(self, x):
        x = self.c1(x)
        return self.c2(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = DownConv(in_channels, out_channels)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = UpConv(in_channels, out_channels)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.block(x)