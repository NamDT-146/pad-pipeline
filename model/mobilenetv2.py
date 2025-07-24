import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobileNetV2Network(nn.Module):
    """
    Siamese-style network using MobileNetV2 as the feature extractor for fingerprint verification.
    """
    def __init__(self, input_channels=1, embedding_size=512, pretrained=False):
        super(MobileNetV2Network, self).__init__()
        # Load MobileNetV2 backbone
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        # Adjust first conv layer for grayscale if needed
        if input_channels == 1:
            weight = mobilenet.features[0][0].weight.data.sum(dim=1, keepdim=True)
            mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            mobilenet.features[0][0].weight.data = weight
        self.feature_net = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(1280, embedding_size)
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_size, 16),
            nn.LeakyReLU(0.15),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def unit_normalize(self, x):
        """Normalize feature vectors to unit length"""
        return F.normalize(x, p=2, dim=1)

    def forward_one(self, x):
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

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = out1 - out2
        diff = diff * diff
        score = self.similarity_net(diff)
        return score

    def get_feature_extractor(self):
        return nn.Sequential(self.feature_net, self.pool, self.embedding)

    def extract_features(self, x):
        return self.forward_one(x)

def create_mobilenetv2_model(device, pretrained=False):
    model = MobileNetV2Network(pretrained=pretrained).to(device)
    return model
