import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese network for fingerprint verification.
    Uses a shared CNN for feature extraction and a similarity network for matching.
    """
    def __init__(self, input_channels=1, base_filters=32, embedding_size=512):
        super(SiameseNetwork, self).__init__()
        
        # activation = nn.LeakyReLU(0.15)
        self.activation = nn.SiLU()

        self.feature_net = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            self.activation,
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            self.activation,
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            self.activation,
            nn.Dropout(0.4),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            self.activation,
            nn.Dropout(0.4),
            nn.MaxPool2d(2),

            # Block 5
            nn.Conv2d(base_filters*8, embedding_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_size),
            self.activation,
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((1, 1)),  # Add this instead
        )

        self.fc_feature = nn.Linear(embedding_size, 16)        
        self.fc_final = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def unit_normalize(self, x):
        return F.normalize(x, p=2, dim=1)
    
    def forward_one(self, x):
        # Reshape to [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[3] == 1:
            x = x.permute(0, 3, 1, 2)
            
        # Extract features
        x = self.feature_net(x)
        x = torch.flatten(x, 1)
        x = self.unit_normalize(x)
        return x
        
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = out1 - out2
        x = self.fc_feature(diff)
        x = x * x  # Element-wise square
        x = self.fc_final(x)
        score = self.sigmoid(x)
        return score
    
    def get_feature_extractor(self):
        return self.feature_net
    
    def extract_features(self, x):
        return self.forward_one(x)

    def compute_similarity(self, ft_vec1, ft_vec2):
        diff = ft_vec1 - ft_vec2
        x = self.fc_feature(diff)
        x = x * x
        x = self.fc_final(x)
        score = self.sigmoid(x)
        return score



# Add a file with metrics utility functions
def create_siamese_model(device):
    model = SiameseNetwork().to(device)
    return model