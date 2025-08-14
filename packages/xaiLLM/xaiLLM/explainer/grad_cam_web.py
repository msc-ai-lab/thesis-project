import matplotlib
matplotlib.use('Agg')

import torch
from PIL import Image
import io
from captum.attr import LayerGradCam
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .XaiModel import XaiModel
from .wrapper import WrapperModel

class GradCAMWeb(XaiModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def generate(self, input_tensor: torch.Tensor, input_image: Image, predicted_class_index: int) -> Image:
        """
        Generates a Grad-CAM visualisation and returns it as a PIL Image
        without displaying it on screen.
        """
        print('Generating Grad-CAM visualisation (Web Version)...')
        model_wrapper = WrapperModel(self.model)

        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        grad_cam_layer = self.model.features[-1]
        layer_gc = LayerGradCam(model_wrapper.get_logits, grad_cam_layer)
        attribution_gc = layer_gc.attribute(input_tensor, target=predicted_class_index)

        heatmap = F.interpolate(attribution_gc, size=input_image.size, mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().detach().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(input_image)
        heatmap_img = ax.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04, label='Attribution Intensity')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        composite_image = Image.open(buf).convert('RGB')
        
        plt.close(fig)
        buf.close()

        return composite_image