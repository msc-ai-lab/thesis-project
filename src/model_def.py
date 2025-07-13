import torch.nn as nn
from transformers import AutoModelForImageClassification

class SkinCancerViT(nn.Module):
    """
    Definitive Vision Transformer model for the project.
    Loads a pre-trained model from Hugging Face and adapts its classifier head.
    """
    def __init__(self, num_classes=2):
        super(SkinCancerViT, self).__init__()
        model_name = "Anwarkh1/Skin_Cancer-Image_Classification"
        # The core model is loaded from Hugging Face
        self.vit_model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True # Adapts the final layer for our 2-class problem
        )

    def forward(self, x):
        # The loaded model handles the full forward pass, including the classifier
        return self.vit_model(x)