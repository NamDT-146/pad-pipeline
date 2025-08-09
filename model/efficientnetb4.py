import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetB4Network(nn.Module):
    """
    Siamese-style network using EfficientNet-B4 as the feature extractor for fingerprint verification.
    Mirrors the API and behavior of MobileNetV2Network.
    """

    def __init__(self, input_channels: int = 1, embedding_size: int = 512, pretrained: bool = True):
        super(EfficientNetB4Network, self).__init__()

        # Load EfficientNet-B4 backbone with compatibility for different torchvision versions
        efficientnet = None
        try:
            # Newer torchvision uses the Weights enum
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            efficientnet = models.efficientnet_b4(weights=weights)
        except Exception:
            # Older torchvision used the 'pretrained' flag
            efficientnet = models.efficientnet_b4(pretrained=pretrained)

        # Adjust first conv layer for grayscale if needed
        if input_channels == 1:
            # In torchvision EfficientNet, the stem is features[0], a Conv2dNormActivation
            stem_conv: nn.Conv2d = efficientnet.features[0][0]
            with torch.no_grad():
                weight = stem_conv.weight.data.sum(dim=1, keepdim=True)
            new_stem_conv = nn.Conv2d(1, stem_conv.out_channels, kernel_size=stem_conv.kernel_size,
                                      stride=stem_conv.stride, padding=stem_conv.padding, bias=False)
            with torch.no_grad():
                new_stem_conv.weight.data = weight
            efficientnet.features[0][0] = new_stem_conv

        self.feature_net = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine the input dimension to the embedding layer from the classifier
        # Torchvision EfficientNet classifier is Dropout -> Linear
        try:
            in_features = efficientnet.classifier[1].in_features
        except Exception:
            # Fallback if structure differs
            in_features = 1792  # Known for EfficientNet-B4

        self.embedding = nn.Linear(in_features, embedding_size)
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_size, 16),
            nn.LeakyReLU(0.15),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def unit_normalize(x: torch.Tensor) -> torch.Tensor:
        """Normalize feature vectors to unit length."""
        return F.normalize(x, p=2, dim=1)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        # Support inputs of shape (B, H, W) or (B, H, W, 1)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)

        x = self.feature_net(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = self.unit_normalize(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = out1 - out2
        diff = diff * diff
        score = self.similarity_net(diff)
        return score

    def get_feature_extractor(self) -> nn.Sequential:
        return nn.Sequential(self.feature_net, self.pool, self.embedding)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_one(x)

    def compute_similarity(self, ft_vec1: torch.Tensor, ft_vec2: torch.Tensor) -> torch.Tensor:
        diff = ft_vec1 - ft_vec2
        diff = diff * diff
        score = self.similarity_net(diff)
        return score


def create_efficientnetb4_model(device, pretrained: bool = True):
    model = EfficientNetB4Network(pretrained=pretrained).to(device)
    return model

