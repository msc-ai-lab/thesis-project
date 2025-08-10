"""
Utility helpers for xaiLLM.

Includes tensor/image utilities, dataset loading, and simple display helpers
used by the main pipeline.
"""

import torch
from torch.utils.data import TensorDataset
from PIL import Image
from matplotlib import pyplot as plt

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


def show_title(text: str) -> None:
    """
    Prints a title with a specific format.
    
    Parameters
    ----------
    text : str
        The text to print as a title.
    """
    print(f"\n{'=' * 22}\n{text}\n{'=' * 22}\n")


def load_datasets(dataset_path: str) -> tuple:
    """
    Load the train dataset with filenames from the specified path.
    Parameters
    ----------
    path : str
        The path where the datasets are saved.

    Returns
    -------
    tuple
        A  tuple of training dataset and its filenames
    """
    
    try:
        # Load the saved datasets
        train_data = torch.load(dataset_path, weights_only=False)

        # Check if the required keys exist in the loaded data
        required_keys = ['images', 'labels', 'filenames']
        for key in required_keys:
            if key not in train_data:
                raise KeyError(f"Required key '{key}' not found in dataset.")

        return TensorDataset(train_data['images'], train_data['labels']), train_data['filenames']
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset_path}. Please ensure the path is correct.")
        return None, None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading the dataset: {e}")
        return None, None


def show_image(image: Image, title: str) -> None:
    """
    Show an image with a title.

    Parameters
    ----------
    image : Image
        The image to display.
    title : str
        The title to display above the image.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()
