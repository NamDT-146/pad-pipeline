import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSplitConv(nn.Module):
    """
    Dynamic Multi-scale Context (DMC) style block.

    Approximates the Dite-HRNet DMC idea by splitting into multiple
    depthwise convolution branches with different receptive fields and
    using a lightweight attention to dynamically fuse them.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 8):
        super().__init__()
        mid_channels = max(in_channels // 2, 16)

        # Pointwise expansion to retain capacity
        self.expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(mid_channels)

        # Depthwise conv branches with different kernels
        self.dw3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.dw5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels, bias=False)
        self.dw7 = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, groups=mid_channels, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        self.bn7 = nn.BatchNorm2d(mid_channels)

        # Attention to fuse branches dynamically (gating per branch)
        squeeze_channels = max(mid_channels // reduction_ratio, 8)
        self.attn_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_fc = nn.Sequential(
            nn.Conv2d(mid_channels, squeeze_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, 3, kernel_size=1)  # 3 gates for 3 branches
        )

        # Projection to desired out channels
        self.project = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn_project = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # Shortcut if needed
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn_expand(self.expand(x)))

        b3 = self.act(self.bn3(self.dw3(x)))
        b5 = self.act(self.bn5(self.dw5(x)))
        b7 = self.act(self.bn7(self.dw7(x)))

        # Compute gates and fuse
        gates = self.attn_fc(self.attn_pool(x))  # [B, 3, 1, 1]
        gates = F.softmax(gates, dim=1)
        fused = b3 * gates[:, 0:1] + b5 * gates[:, 1:2] + b7 * gates[:, 2:3]

        out = self.act(self.bn_project(self.project(fused)))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)


class DynamicGlobalContext(nn.Module):
    """
    Adaptive/Dynamic Global Context (DGC) style block.
    Channel attention that models long-range dependency.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(channels // reduction_ratio, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class FuseUnit(nn.Module):
    """
    Two-branch HR-style fuse: maintains high-res and low-res branches and fuses them.
    """

    def __init__(self, ch_high: int, ch_low: int):
        super().__init__()
        # High-resolution branch processing
        self.h_proc = nn.Sequential(
            DynamicSplitConv(ch_high, ch_high),
            DynamicGlobalContext(ch_high),
        )
        # Low-resolution branch processing
        self.l_proc = nn.Sequential(
            DynamicSplitConv(ch_low, ch_low),
            DynamicGlobalContext(ch_low),
        )

        # For fusion: downsample high->low and upsample low->high
        self.down_h2l = nn.Sequential(
            nn.Conv2d(ch_high, ch_low, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_low),
        )
        self.up_l2h = nn.Sequential(
            nn.Conv2d(ch_low, ch_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_high),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, h: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.h_proc(h)
        l1 = self.l_proc(l)

        # Fuse
        l_from_h = self.down_h2l(h1)
        h_from_l = self.up_l2h(F.interpolate(l1, size=h1.shape[-2:], mode='bilinear', align_corners=False))

        h_out = self.act(h1 + h_from_l)
        l_out = self.act(l1 + l_from_h)
        return h_out, l_out


class TinyDiteHRBackbone(nn.Module):
    """
    A compact, Dite-HRNet-inspired backbone suitable for fingerprint feature extraction.

    Structure:
    - Stem conv -> Stage with two parallel resolutions and several FuseUnits
    - Final aggregation by concatenating multi-resolution features
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_fuse_units: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        ch_high = base_channels
        ch_low = base_channels * 2

        # Transition to create low-res branch
        self.to_low = nn.Sequential(
            nn.Conv2d(ch_high, ch_low, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_low),
            nn.SiLU(inplace=True),
        )

        self.fuse_units = nn.ModuleList([FuseUnit(ch_high, ch_low) for _ in range(num_fuse_units)])

        # Head to mix multi-scale features back to a single stream
        self.head_high = nn.Conv2d(ch_high, ch_high, kernel_size=1, bias=False)
        self.head_low = nn.Conv2d(ch_low, ch_high, kernel_size=1, bias=False)
        self.bn_head_h = nn.BatchNorm2d(ch_high)
        self.bn_head_l = nn.BatchNorm2d(ch_high)
        self.act = nn.SiLU(inplace=True)

        self.out_channels = ch_high * 2  # concatenate high and upsampled low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        h = x
        l = self.to_low(x)

        for unit in self.fuse_units:
            h, l = unit(h, l)

        h2 = self.act(self.bn_head_h(self.head_high(h)))
        l2 = self.act(self.bn_head_l(self.head_low(F.interpolate(l, size=h.shape[-2:], mode='bilinear', align_corners=False))))

        # Concatenate multi-scale features
        return torch.cat([h2, l2], dim=1)


class DiteHRNetNetwork(nn.Module):
    """
    Siamese-style network using a compact Dite-HRNet-inspired backbone for feature extraction.

    This mirrors the API of the other models in this project.
    """

    def __init__(self, input_channels: int = 1, embedding_size: int = 512, base_channels: int = 32, num_fuse_units: int = 3):
        super().__init__()
        self.backbone = TinyDiteHRBackbone(in_channels=input_channels, base_channels=base_channels, num_fuse_units=num_fuse_units)

        # Projection head
        self.proj = nn.Conv2d(self.backbone.out_channels, embedding_size, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(embedding_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Similarity network (same shape as other models)
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_size, 16),
            nn.LeakyReLU(0.15, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def unit_normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,H,W) or (B,H,W,1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)

        feats = self.backbone(x)
        feats = self.bn_proj(self.proj(feats))
        feats = self.pool(feats)
        feats = torch.flatten(feats, 1)
        feats = self.unit_normalize(feats)
        return feats

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_one(x1)
        f2 = self.forward_one(x2)
        diff = (f1 - f2) * (f1 - f2)
        return self.similarity_net(diff)

    def get_feature_extractor(self) -> nn.Sequential:
        # Expose a sequential module for feature extraction pipeline
        return nn.Sequential(self.backbone, self.proj, self.bn_proj, nn.SiLU(inplace=True), self.pool)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_one(x)

    def compute_similarity(self, ft_vec1: torch.Tensor, ft_vec2: torch.Tensor) -> torch.Tensor:
        diff = (ft_vec1 - ft_vec2) * (ft_vec1 - ft_vec2)
        return self.similarity_net(diff)


def create_ditehrnet_model(device, input_channels: int = 1, embedding_size: int = 512, base_channels: int = 32, num_fuse_units: int = 3, **kwargs):
    model = DiteHRNetNetwork(
        input_channels=input_channels,
        embedding_size=embedding_size,
        base_channels=base_channels,
        num_fuse_units=num_fuse_units,
    ).to(device)
    return model

