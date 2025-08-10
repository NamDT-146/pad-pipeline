import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Lightweight residual conv block used in the CoarseNet-style backbone.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down = None
        if in_channels != out_channels or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out


class MultiBranchConv(nn.Module):
    """
    A small multi-branch conv module to mimic CoarseNet's multi-cue fusion idea.

    Branches: 1x1, 3x3, 5x5 depthwise followed by pointwise to fuse.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.branch3_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.branch5_dw = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn5 = nn.BatchNorm2d(channels)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.act(self.bn1(self.branch1(x)))
        b3 = self.act(self.bn3(self.branch3_dw(x)))
        b5 = self.act(self.bn5(self.branch5_dw(x)))
        out = torch.cat([b1, b3, b5], dim=1)
        out = self.act(self.bn_fuse(self.fuse(out)))
        return out


class InceptionResBlock(nn.Module):
    """
    Inception-ResNet style refinement block used to approximate FineNet behavior.
    """

    def __init__(self, channels: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        reduced = max(channels // reduction, 16)
        self.reduce = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn_r = nn.BatchNorm2d(reduced)

        self.b1 = nn.Conv2d(reduced, reduced, kernel_size=1, bias=False)
        self.b3 = nn.Conv2d(reduced, reduced, kernel_size=3, padding=1, bias=False)
        self.b5 = nn.Conv2d(reduced, reduced, kernel_size=5, padding=2, bias=False)

        self.bn1 = nn.BatchNorm2d(reduced)
        self.bn3 = nn.BatchNorm2d(reduced)
        self.bn5 = nn.BatchNorm2d(reduced)

        self.expand = nn.Conv2d(reduced * 3, channels, kernel_size=1, bias=False)
        self.bn_e = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.dp = nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn_r(self.reduce(x)))
        b1 = self.act(self.bn1(self.b1(x)))
        b3 = self.act(self.bn3(self.b3(x)))
        b5 = self.act(self.bn5(self.b5(x)))
        out = torch.cat([b1, b3, b5], dim=1)
        out = self.bn_e(self.expand(out))
        out = self.dp(out)
        out = self.act(out + identity)
        return out


class CoarseNetBackbone(nn.Module):
    """
    A compact CoarseNet-inspired backbone: residual blocks + multi-branch convs.

    It produces a spatial feature map to be refined by FineNet-style blocks.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(c, c),
            MultiBranchConv(c),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(c, c * 2, stride=2),
            MultiBranchConv(c * 2),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(c * 2, c * 4, stride=2),
            MultiBranchConv(c * 4),
        )
        self.out_channels = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class FineNetRefiner(nn.Module):
    """
    A small stack of Inception-ResNet blocks to mimic FineNet's patch refinement,
    applied in a dense fashion to the feature map.
    """

    def __init__(self, channels: int, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(InceptionResBlock(channels, reduction=4, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MinutiaeNetBackbone(nn.Module):
    """
    MinutiaeNet-inspired backbone combining CoarseNet-like and FineNet-like modules.

    Reference: MinutiaeNet (ICB 2018) repository and paper
    - GitHub: https://github.com/luannd/MinutiaeNet
    - Paper: arxiv.org/pdf/1712.09401.pdf
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_refine_blocks: int = 3):
        super().__init__()
        self.coarse = CoarseNetBackbone(in_channels=in_channels, base_channels=base_channels)
        self.refine = FineNetRefiner(self.coarse.out_channels, num_blocks=num_refine_blocks, dropout=0.1)
        self.out_channels = self.coarse.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coarse(x)
        x = self.refine(x)
        return x


class MinutiaeNetSiamese(nn.Module):
    """
    Siamese-style network using a MinutiaeNet-inspired backbone as a feature extractor.

    API mirrors other models in this project.
    """

    def __init__(self, input_channels: int = 1, embedding_size: int = 512, base_channels: int = 32, num_refine_blocks: int = 3):
        super().__init__()
        self.backbone = MinutiaeNetBackbone(
            in_channels=input_channels,
            base_channels=base_channels,
            num_refine_blocks=num_refine_blocks,
        )

        # Projection head to embedding
        self.proj = nn.Conv2d(self.backbone.out_channels, embedding_size, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(embedding_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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
        # Expose sequential feature extractor similar to other models
        return nn.Sequential(self.backbone, self.proj, self.bn_proj, nn.SiLU(inplace=True), self.pool)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_one(x)

    def compute_similarity(self, ft_vec1: torch.Tensor, ft_vec2: torch.Tensor) -> torch.Tensor:
        diff = (ft_vec1 - ft_vec2) * (ft_vec1 - ft_vec2)
        return self.similarity_net(diff)


def create_minutiaenet_model(device, input_channels: int = 1, embedding_size: int = 512, base_channels: int = 32, num_refine_blocks: int = 3, **kwargs):
    model = MinutiaeNetSiamese(
        input_channels=input_channels,
        embedding_size=embedding_size,
        base_channels=base_channels,
        num_refine_blocks=num_refine_blocks,
    ).to(device)
    return model

