import torch


def extract_size(input_tensor: torch.Tensor) -> tuple:
    """
    Extract height and width from input tensor for resizing.
    This function determines the size based on the shape of the input tensor,
    which can either include a batch dimension or not.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor from which to extract height and width.
    
    Returns
    -------
    tuple
        A tuple containing the height and width of the input tensor.
    """
    if len(input_tensor.shape) == 4:  # Batch dimension included
        _, _, height, width = input_tensor.shape
        size = (height, width)
    else:  # No batch dimension
        _, height, width = input_tensor.shape
        size = (height, width)
    return size