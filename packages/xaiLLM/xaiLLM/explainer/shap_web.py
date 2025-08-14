import matplotlib
matplotlib.use('Agg')

import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import Occlusion
import torch
import torch.nn as nn

from .XaiModel import XaiModel
from .wrapper import WrapperModel

class SHAPExplainerWeb(XaiModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def _visualise_shap_blended_heatmap(self, attribution: np.ndarray, original_image: np.ndarray) -> Image.Image:
        plt.ioff()
        max_abs = np.max(np.abs(attribution)) + 1e-8
        norm_attr = np.clip(attribution / max_abs, -1, 1)
        heatmap = norm_attr.mean(axis=-1) if norm_attr.ndim == 3 else norm_attr
        cmap = LinearSegmentedColormap.from_list("red_green", ["red", "white", "green"])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.axis('off')

        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image.astype(np.uint8)

        ax.imshow(original_image)
        heatmap_img = ax.imshow(heatmap, cmap=cmap, alpha=0.5, vmin=-1, vmax=1)
        plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04, label='Attribution Intensity')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        
        plt.close(fig)
        buf.close()
        return pil_img

    def generate(self, input_tensor: torch.Tensor, input_image: Image.Image, predicted_class_index: int) -> Image.Image:
        """
        Generates a SHAP visualisation and returns it as a PIL Image
        without displaying it on screen.
        """
        print('Generating SHAP visualisation (Web Version)...')
        
        model_wrapper = WrapperModel(self.model)
        occlusion = Occlusion(model_wrapper.get_logits)

        attribution_shap = occlusion.attribute(
            input_tensor,
            strides=(3, 32, 32),
            target=predicted_class_index,
            sliding_window_shapes=(3, 48, 48),
            baselines=0
        )
        
        attr_np = np.transpose(attribution_shap.squeeze().cpu().detach().numpy(), (1, 2, 0))
        input_np = np.array(input_image)
        composite_image = self._visualise_shap_blended_heatmap(attr_np, input_np)

        return composite_image