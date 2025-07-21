from torch import nn
from torchvision import models

class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerCNN, self).__init__()

        # Feature Extractor: Pretrained ResNet backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Attention Module for Explainability
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        # Global Pooling + Fully Connected Layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Apply attention mask
        attn_map = self.attention(x)
        x = x * attn_map
        x = self.pool(x)
        x = self.classifier(x)
        x = nn.Softmax(dim=1)(x)
        return x, attn_map