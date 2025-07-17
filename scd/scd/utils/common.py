import os
import timm
import torch
from torch import nn
from torch.utils.data import TensorDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import ViTForImageClassification

def load_datasets(path: str) -> tuple:
    """
    Load the train, validation, and test datasets from the specified path.
    Parameters
    ----------
        path : str
            The path where the datasets are saved.
    Returns
    -------
        tuple
            A tuple containing three TensorDataset objects: train, validation, and test datasets.
    """
    # Load the saved datasets
    train_data = torch.load(os.path.join(path, 'train_dataset.pt'), weights_only=False)
    val_data = torch.load(os.path.join(path, 'val_dataset.pt'), weights_only=False)
    test_data = torch.load(os.path.join(path, 'test_dataset.pt'), weights_only=False)

    # Extract images and labels
    train_images, train_labels, train_filenames = train_data['images'], train_data['labels'], train_data['filenames']
    val_images, val_labels = val_data['images'], val_data['labels']
    test_images, test_labels = test_data['images'], test_data['labels']

    # Create TensorDataset objects
    train_tensor_dataset = TensorDataset(train_images, train_labels, train_filenames)
    val_tensor_dataset = TensorDataset(val_images, val_labels)
    test_tensor_dataset = TensorDataset(test_images, test_labels)

    return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset

def get_test_transforms(resize: tuple = (224, 224)) -> A.Compose:
    """
    Get the testing transformations for the dataset.
    
    Parameters
    ----------
        resize : tuple
            The size to which the images will be resized. Default is (224, 224).

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

def get_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """
    Get the model based on the provided model name.
    
    Parameters
    ----------
        model_name : str
            The name of the model to retrieve. Supported models are 'Xception' and 'ViT'.
        num_classes : int
            The number of output classes for the model. Default is 2.
    Returns
    -------
        torch.nn.Module
            The model instance corresponding to the provided model name.
    """
    if model_name == 'Xception':
        model = timm.create_model('xception', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ViT':
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            attn_implementation="sdpa"
        )
        # model = ViTForImageClassification.from_pretrained(
        #     'google/vit-base-patch32-384',
        #     num_labels=num_classes,
        #     ignore_mismatched_sizes=True,
        #     attn_implementation="sdpa"
        # )
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model