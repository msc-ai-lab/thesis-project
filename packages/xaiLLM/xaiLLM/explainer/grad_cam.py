"""
Grad-CAM visual explanation utilities.

Provides the GradCAM class to compute and render class-discriminative
localisation maps for CNNs using Captum's LayerGradCam.

Key class:
- GradCAM(model): generate(input_tensor, input_image, predicted_class_index, show_image=True)
"""

import torch
from PIL import Image
import io
from captum.attr import LayerGradCam
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from xaiLLM.explainer.XaiModel import XaiModel
from xaiLLM.explainer.wrapper import WrapperModel


class GradCAM(XaiModel):
    def __init__(self, model: nn.Module):
        """
        Initialize the GradCAM explainer with a PyTorch model.
        
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to be explained.
        """
        super().__init__(model)

    def generate(self, input_tensor: torch.Tensor, input_image: Image, predicted_class_index: int, show_image: bool = True) -> Image:
        """
        Generate Grad-CAM visualisation for the input tensor using the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.
        input_image : Image
            The original image for visualisation.
        predicted_class_index : int
            The index of the predicted class.
        show_image : bool, default=True
            Whether to display the image during processing.
        
        Returns
        -------
        Image
            A PIL Image with Grad-CAM overlay.
        """
        print('Generating Grad-CAM visualisation...')
        model_wrapper = WrapperModel(self.model)

        # Ensure input tensor requires gradients
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        # Apply Grad-CAM
        grad_cam_layer = self.model.features[-1]
        layer_gc = LayerGradCam(model_wrapper.get_logits, grad_cam_layer)
        attribution_gc = layer_gc.attribute(input_tensor, target=predicted_class_index)

        # Visualise GRAD-CAM
        heatmap = F.interpolate(attribution_gc, size=input_image.size, mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().detach().numpy()

        # Create a composite image for return
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Grad-CAM Visualisation')
        ax.imshow(input_image)
        heatmap_img = ax.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04, label='Attribution Intensity')
        ax.axis('off')
        if show_image:
            plt.show()

        # Save to buffer and convert to image for return
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        composite_image = Image.open(buf).convert('RGB')
        plt.close()
        buf.close()

        return composite_image
