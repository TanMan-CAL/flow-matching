class UnconditionalUNet(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int):
        super().__init__()
        D = num_hiddens

        # encoder
        self.conv1 = ConvBlock(in_channels, D) # spatial: 28->28
        self.down1 = DownBlock(D, D) # spatial: 28->14
        self.down2 = DownBlock(D, 2*D) #spatial: 14->7

        # bottleneck
        self.flatten = Flatten() # 2D x7x7 -> 2D x1x1
        self.unflatten = Unflatten(2*D) # 2D x1x1 -> 2D x7x7

        # decoder
        self.up1 = UpBlock(4*D, D) # 4D x7x7  -> D  x14x14
        self.up2 = UpBlock(2*D, D) # 2D x14x14 -> D  x28x28
        self.up3 = ConvBlock(2*D, D) # 2D x 28x28 -> D  x 28x28, was wrong to upblock here

        # conv to map back to image channels
        self.out_conv = nn.Conv2d(D, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."

        # encoder
        e1 = self.conv1(x) # D x 28x28
        e2 = self.down1(e1) # D x 14x14
        e3 = self.down2(e2) # 2D x 7x7

        # bottleneck
        b = self.flatten(e3) # 2D x 1x1
        b = self.unflatten(b) # 2D x 7x7

        # decoder
        # each part is skip -> UpBlock -> next level
        d = self.up1(torch.cat([b,  e3], dim=1)) # cat=4D x7x7  -> D x14x14
        d = self.up2(torch.cat([d,  e2], dim=1)) # cat=2D x14x14 -> D x28x28
        d = self.up3(torch.cat([d,  e1], dim=1)) # cat=2D x28x28 -> D x28x28

        # output projection
        return self.out_conv(d) # D x28x28 -> in_channels x28x28
