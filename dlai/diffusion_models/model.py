import torch
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

        self.down1 = UnetDown(n_features, n_features)  # down1: [B, n_features, H/2, W/2]
        self.down2 = UnetDown(n_features, n_features * 2)  # down2: [B, 2*n_features, H/4, W/4]

        self.to_vec = nn.Sequential(nn.AvgPool2d(self.img_dim // 4), nn.GELU())  # to_vec: [B, 2*n_features, 1, 1]

        self.pos_emb1 = SinusoidalPositionEmbeddings(2 * n_features)
        self.pos_emb2 = SinusoidalPositionEmbeddings(n_features)

        self.time_emb1 = EmbedFC(2 * n_features, 2 * n_features)
        self.time_emb2 = EmbedFC(n_features, n_features)
        self.context_emb1 = EmbedFC(n_context_features, 2 * n_features)
        self.context_emb2 = EmbedFC(n_context_features, n_features)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * n_features, 2 * n_features, self.img_dim // 4, self.img_dim // 4
            ),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_features, n_features)  # up1: [B, 2*n_features, H/2, W/2]
        self.up2 = UnetUp(2 * n_features, n_features)  # up2: [B, n_features, H, W]

        self.out = nn.Sequential(
            nn.Conv2d(
                2 * n_features,
                n_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t, context=None):
        """
        x: (B, C, H, W)
        t: (B, n_context_features?)
        context: (B, n_context_features)
        """
        x = self.init_conv(x)  # [B, n_features, H, W]
        down1 = self.down1(x)  # [B, n_features, H/2, W/2] ## [10, 256, 8, 8]
        down2 = self.down2(down1)  # [B, 2*n_features, H/4, W/4] ## [10, 256, 4, 4]

        # convert feature maps to vector and apply activation:
        hidden_vec = self.to_vec(down2)  # [B, 2*n_features] ## [10, 512]

        if context is None:
            context = torch.zeros(
                (x.shape[0], self.n_context_features), device=x.device
            )  # [B, n_context_features]

        context_emb1 = self.context_emb1(context).view(
            -1, 2 * self.n_features, 1, 1
        )  # [B, 2*n_features, 1, 1]
        context_emb2 = self.context_emb2(context).view(
            -1, self.n_features, 1, 1
        )  # [B, n_features, 1, 1]

        time_emb1 = self.time_emb1(self.pos_emb1(t)).view(
            -1, 2 * self.n_features, 1, 1
        )  # [B, 2*n_features, 1, 1]
        time_emb2 = self.time_emb2(self.pos_emb2(t)).view(
            -1, self.n_features, 1, 1
        )  # [B, n_features, 1, 1]

        up1 = self.up0(hidden_vec)
        up2 = self.up1(context_emb1 * up1 + time_emb1, down2)
        up3 = self.up2(context_emb2 * up2 + time_emb2, down1)
        out = self.out(torch.cat((up3, x), dim=1))
        return out


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False,
        scale_factor: float = 0.5**0.5,
    ):
        super().__init__()
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
            if self.in_channels == self.out_channels:
                out = x + x2
            else:
                out = self.shortcut(x) + x2
            return out * self.scale_factor
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)  # concatenate along channel dimension
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.input_dim = in_dim
        layers = [
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time.float()
        device = time.device
        half_dim = self.dim // 2

        # Frequencies
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # multiply time by frequencies
        embeddings = time[:, None] * embeddings[None, :]
        
        # sine and cosine pairs
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings