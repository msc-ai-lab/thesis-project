import pytest
from pathlib import Path
from scd.model import SkinCancerCNN
from torch import nn

from scd.utils.common import load_datasets, get_test_transforms, load_model
from scd.utils.constants import INPUT_SIZE_FOR_MODELS

ROOT_DIR = Path(__file__).parent.parent.parent.parent

def test_load_datasets_with_wrong_path():
    path = 'wrong/path/to/datasets'
    with pytest.raises(FileNotFoundError):
      _, _, _ = load_datasets(path)


def test_load_datasets_with_correct_path():
    # Set up a valid path for testing
    resize = INPUT_SIZE_FOR_MODELS['ResNet34']
    data_dir = ROOT_DIR / 'data' / 'processed'

    # Test loading datasets from the correct path
    train_dataset, val_dataset, test_dataset = load_datasets(data_dir)
    assert train_dataset is not None and val_dataset is not None and test_dataset is not None
    assert len(train_dataset) > 0 and len(val_dataset) > 0 and len(test_dataset) > 0
    assert len(train_dataset.tensors) == 2 and len(val_dataset.tensors) == 2 and len(test_dataset.tensors) == 2
    assert train_dataset.tensors[0].shape[1:] == (3, resize[0], resize[1])

    # Test with only training dataset and filenames
    train_dataset, filenames = load_datasets(data_dir, only_train_dataset_with_filenames=True)
    assert train_dataset is not None and len(train_dataset.tensors) == 2
    assert filenames is not None and len(filenames) > 0


def test_get_test_transforms():
    from albumentations import Compose

    # Test with default resize
    transforms = get_test_transforms()
    assert isinstance(transforms, Compose)

    # Test with custom resize
    custom_resize = (128, 128)
    transforms = get_test_transforms(resize=custom_resize)
    assert isinstance(transforms, Compose)
    assert transforms.transforms[0].height == custom_resize[0]
    assert transforms.transforms[0].width == custom_resize[1]


def test_load_model_with_invalid_path():
    invalid_path = Path('invalid/path/to/model.pth')
    with pytest.raises(FileNotFoundError):
        load_model(invalid_path)


def test_load_model_with_valid_path():
    model_path = ROOT_DIR / 'models' / 'ResNet34-class_weights.pth'
    model = load_model(model_path)
    assert isinstance(model, SkinCancerCNN)