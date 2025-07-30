import os
import torch
from torch.utils.data import TensorDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from scd.model import SkinCancerCNN

def load_datasets(path: str, only_train_dataset_with_filenames: bool = False) -> tuple:
    """
    Load the train, validation, and test datasets from the specified path.
    Parameters
    ----------
        path : str
            The path where the datasets are saved.
        only_train_dataset_with_filenames : bool
            If True, only the training dataset with filenames will be returned. Default is False.
    Returns
    -------
        tuple
            A tuple containing three TensorDataset objects: train, validation, and test datasets.
            If `only_train_dataset_with_filenames` is True, it returns only the tuple of
            training dataset and its filenames
    """
    
    try:
        # Load the saved datasets
        train_data = torch.load(os.path.join(path, 'train_dataset.pt'), weights_only=False)

        if only_train_dataset_with_filenames:
            # If only training data with filenames is needed, return only the train dataset
            return TensorDataset(train_data['images'], train_data['labels']), train_data['filenames']

        # Load validation and test datasets
        val_data = torch.load(os.path.join(path, 'val_dataset.pt'), weights_only=False)
        test_data = torch.load(os.path.join(path, 'test_dataset.pt'), weights_only=False)

        # Extract images and labels
        train_images, train_labels = train_data['images'], train_data['labels']
        val_images, val_labels = val_data['images'], val_data['labels']
        test_images, test_labels = test_data['images'], test_data['labels']

        # Create TensorDataset objects
        train_tensor_dataset = TensorDataset(train_images, train_labels)
        val_tensor_dataset = TensorDataset(val_images, val_labels)
        test_tensor_dataset = TensorDataset(test_images, test_labels)

        return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset files not found in the specified path: {path}. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading datasets from {path}: {e}")


def get_test_transforms(resize: tuple = (384, 384)) -> A.Compose:
    """
    Get the testing transformations for the dataset.
    
    Parameters
    ----------
        resize : tuple
            The size to which the images will be resized. Default is (384, 384).

    Returns
    -------
        A.Compose
            A composition of transformations to be applied to the images.
    """
    width, height = resize
    test_transforms = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return test_transforms


def load_model(model_path: str) -> SkinCancerCNN:
    """
    Load a pre-trained model from the specified path.
    
    Parameters
    ----------
        model_path : str
            The path to the pre-trained model file.
    
    Returns
    -------
        SkinCancerCNN
            An instance of the SkinCancerCNN model loaded with the pre-trained weights.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = SkinCancerCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model