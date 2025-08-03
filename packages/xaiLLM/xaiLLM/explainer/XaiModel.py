import io
import torch
from torch import nn
from abc import ABC, abstractmethod
import base64
from PIL import Image

class XaiModel(ABC):
  """
  Abstract base class for model explainers.
  """
  
  def __init__(self, model: nn.Module):
      """
      Initialize the base model with a PyTorch model.
      
      Parameters
      ----------
      model : nn.Module
          The PyTorch model to be explained.
      """
      self.model = model
      self.model.eval()  # Set model to evaluation mode by default
  
  @abstractmethod
  def generate(self, input_tensor: torch.Tensor, predicted_class_index: int, **kwargs):
      """
      Abstract method to generate visualisation that should be implemented by all subclasses.
      
      Parameters
      ----------
      input_tensor : torch.Tensor
          The input tensor to be explained.
      predicted_class_index : int
          The predicted class index from the model.
      **kwargs : dict
          Additional arguments for explanation.
      """
      pass


  @staticmethod
  def pil_image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
      """
      Convert a PIL Image to a Base64-encoded string.

      Parameters
      ----------
      image : Image.Image
          The PIL image to convert.
      format : str, default='PNG'
          The format to encode the image in (e.g., 'PNG', 'JPEG').

      Returns
      -------
      str
          Base64-encoded string of the image.
      """
      buffered = io.BytesIO()
      image.save(buffered, format=format)
      img_bytes = buffered.getvalue()
      base64_str = base64.b64encode(img_bytes).decode('utf-8')
      buffered.close()
      return base64_str
