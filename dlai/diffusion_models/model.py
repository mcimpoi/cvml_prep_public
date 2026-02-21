import torch.nn as nn


class ContextUnet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_features: int = 256,
        n_context_features: int = 16,
        img_dim: int = 28,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_context_features = n_context_features
        self.img_dim = img_dim  # h == w; must be divisible by 4 for the downsampling/upsampling to work

        self.init_conv = ResidualConvBlock(in_channels, n_features, is_res=True)

        # TODO: unetDown; unetUp; bottleneck; skip connections; time embedding; context embedding


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False,
        scale_factor: float = 0.5**0.5,
    ):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.shortcut = nn.Identity()
        if is_res and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = self.shortcut(x) + x2
            return out * self.scale_factor
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
