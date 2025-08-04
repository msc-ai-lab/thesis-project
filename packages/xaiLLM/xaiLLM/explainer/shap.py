import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import Occlusion
from captum.attr import visualization as viz

import torch
import torch.nn as nn

from xaiLLM.explainer.XaiModel import XaiModel
from xaiLLM.explainer.wrapper import WrapperModel


class SHAPExplainer(XaiModel):
    def __init__(self, model: nn.Module):
        """
        Initialize the SHAP explainer with a PyTorch model.
        
        Args:
            model (nn.Module): The PyTorch model to be explained.
        """
        super().__init__(model)


    def _visualise_shap_blended_heatmap(self, attribution: np.ndarray, original_image: np.ndarray) -> Image.Image:
        """
        Visualise SHAP attributions as a red-green blended heatmap over the original image.
        
        Parameters
        ----------
        attribution : numpy.ndarray
            Attribution map of shape (H, W, C), values in [-1, 1].
        original_image : numpy.ndarray
            Original image as a NumPy array of shape (H, W, 3).
            
        Returns
        -------
        PIL.Image.Image
            Composite image with blended heatmap.
        """
        # Set matplotlib to non-interactive mode to prevent display
        plt.ioff()
        
        # Normalize attribution to [-1, 1]
        max_abs = np.max(np.abs(attribution)) + 1e-8
        norm_attr = np.clip(attribution / max_abs, -1, 1)

        # Convert to grayscale if needed
        heatmap = norm_attr.mean(axis=-1) if norm_attr.ndim == 3 else norm_attr

        # Create custom red-green colormap
        cmap = LinearSegmentedColormap.from_list("red_green", ["red", "white", "green"])

        # Create figure and axis with no display
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_title('SHAP Visualisation')
        ax.axis('off')

        # Ensure original_image is in correct format [0, 255] uint8
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)

        # Show original image
        ax.imshow(original_image)

        # Overlay heatmap
        heatmap_img = ax.imshow(heatmap, cmap=cmap, alpha=0.5, vmin=-1, vmax=1)

        # Add colorbar
        plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04, label='Attribution Intensity')
        fig.tight_layout()

        # Convert figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB')
        
        # Clean up immediately
        plt.close(fig)
        buf.close()

        return pil_img


    def generate(self, input_tensor: torch.Tensor, input_image: Image.Image, predicted_class_index: int, show_image: bool = False) -> Image.Image:
        """
        Generate SHAP visualisation for the input tensor using the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.
        input_image : Image.Image
            The original image for visualisation.
        predicted_class_index : int
            The index of the predicted class.
        show_image : bool, default=False
            Whether to show image visualisation

        Returns
        -------
        Image.Image
            A PIL Image with SHAP overlay.
        """
        print('Generating SHAP visualisation...')
        
        model_wrapper = WrapperModel(self.model)
        occlusion = Occlusion(model_wrapper.get_logits)

        # Compute SHAP attributions
        attribution_shap = occlusion.attribute(
            input_tensor,
            strides=(3, 32, 32),
            target=predicted_class_index,
            sliding_window_shapes=(3, 48, 48),
            baselines=0
        )

        if show_image:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis('off')
            
            viz.visualize_image_attr(
                np.transpose(attribution_shap.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                np.array(input_image),
                method="blended_heat_map",
                sign="all",
                show_colorbar=True,
                plt_fig_axis=(fig, ax),
                title="SHAP Explanation"
            )

            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # Convert buffer to PIL Image
            composite_image = Image.open(buf).convert('RGB')

            # Clean up
            plt.close(fig)
            buf.close()
        else:            
            # Set matplotlib to non-interactive mode to prevent any display
            plt.ioff()

            # Use custom method that doesn't display anything
            attr_np = np.transpose(attribution_shap.squeeze().cpu().detach().numpy(), (1, 2, 0))
            input_np = np.array(input_image)

            composite_image = self._visualise_shap_blended_heatmap(attr_np, input_np)

        return composite_image
