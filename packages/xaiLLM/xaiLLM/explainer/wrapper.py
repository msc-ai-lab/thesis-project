import torch
from torch import nn

class WrapperModel(nn.Module):
    """Wrapper for model to handle output format"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def get_logits(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """        
        Forward pass through the model to get logits.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor for which to get the logits.
        
        Returns
        -------
        torch.Tensor
            The logits output from the model.
        """
        logits = self.model(input_tensor)
        # Handle tuple output (logits, attention_map) from SkinCancerCNN
        return logits[0] if isinstance(logits, tuple) else logits







