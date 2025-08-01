"""
Explainable AI Module
====================

This module provides explainable AI capabilities for skin lesion classification models
using Grad-CAM (Gradient-weighted Class Activation Mapping) visualisation techniques.
It helps interpret model decisions by highlighting important regions in input images.

The explainability pipeline includes:
- Grad-CAM attribution computation using Captum library
- Layer-specific gradient analysis (targeting model.act4 layer)
- Heatmap generation and interpolation to match input image size
- Visualisation overlay combining original image with activation heatmap
- Transparent overlay rendering for clear interpretation

Functions:
---------
- grad_cam(model, input_tensor, input_image, predicted_class_index): Generate Grad-CAM visualisation

Dependencies:
------------
- torch: For PyTorch model operations and tensor handling
- torch.nn: For neural network modules
- torch.nn.functional: For interpolation and tensor operations
- captum.attr: For LayerGradCam attribution computation
- matplotlib.pyplot: For visualisation and plotting

Usage:
-----
    from src.explainer import grad_cam
    
    # Generate Grad-CAM visualisation
    grad_cam(model, input_tensor, original_image, predicted_class_idx)
"""


import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from captum.attr import LayerGradCam, Occlusion
from captum.attr import visualization as viz
from xaiLLM.utils.helpers import extract_size

class WrapperModel(nn.Module):
    """Wrapper for model to handle output format"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def get_logits(self, input_tensor):
        logits = self.model(input_tensor)
        # Handle tuple output (logits, attention_map) from SkinCancerCNN
        return logits[0] if isinstance(logits, tuple) else logits

def grad_cam(model: nn.Module, input_tensor: torch.Tensor, input_image: torch.Tensor, predicted_class_index: int):
  model.eval()  # Set model to evaluation mode
  model_wrapper = WrapperModel(model)
  size = extract_size(input_tensor)

  # Open and convert the image to RGB
  image = Image.open(input_image).convert('RGB')
  image = image.resize(size, resample=Image.BILINEAR)

  # Ensure input tensor requires gradients
  if not input_tensor.requires_grad:
    input_tensor.requires_grad_(True)

  # Apply Grad-CAM
  grad_cam_layer = model.features[-1]
  layer_gc = LayerGradCam(model_wrapper.get_logits, grad_cam_layer)
  attribution_gc = layer_gc.attribute(input_tensor, target=predicted_class_index)

  # Visualise GRAD-CAM
  heatmap = F.interpolate(attribution_gc, size=image.size, mode='bilinear', align_corners=False)
  heatmap = heatmap.squeeze().cpu().detach().numpy()
  
  # Create a composite image for return
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.imshow(image)
  heatmap_img = ax.imshow(heatmap, cmap='jet', alpha=0.4)
  plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04, label='Attribution Intensity')
  ax.axis('off')
  plt.show()
  
  # Save to buffer and convert to image for return
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
  buf.seek(0)
  composite_image = Image.open(buf)
  plt.close()
  
  return composite_image


def shap(model: nn.Module, input_tensor: torch.Tensor, input_image: torch.Tensor, predicted_class_index: int):
  model.eval()  # Set model to evaluation mode
  model_wrapper = WrapperModel(model)

  size = extract_size(input_tensor)

  # Open and convert the image to RGB
  image = Image.open(input_image).convert('RGB')
  image = image.resize(size, resample=Image.BILINEAR)

  occlusion = Occlusion(model_wrapper.get_logits)

  # Adjust sliding window shapes for the new 384x384 size
  attribution_shap = occlusion.attribute(input_tensor, 
                                        strides=(3, 32, 32), 
                                        target=predicted_class_index, 
                                        sliding_window_shapes=(3, 48, 48),
                                        baselines=0)

  fig_shap, ax_shap = plt.subplots(figsize=(8, 8))
  ax_shap.axis('off')
  viz.visualize_image_attr(np.transpose(attribution_shap.squeeze().cpu().detach().numpy(), (1,2,0)), 
                          np.array(image), 
                          method="blended_heat_map", 
                          sign="all", 
                          show_colorbar=True, 
                          plt_fig_axis=(fig_shap, ax_shap))
  plt.show()

  # Save to buffer and convert to image for return
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
  buf.seek(0)
  composite_image = Image.open(buf)
  plt.close()
  
  return composite_image


def get_gradient_and_prediction(model, data_tensor, target, loss_fn):
    model.eval()
    model_wrapper = WrapperModel(model)

    logits = model_wrapper.get_logits(data_tensor)
    _, pred_idx = torch.max(logits.data, 1)
    loss = loss_fn(logits, target)
    loss.backward()
    # Get gradient from the final classifier layer of the ResNet model
    grad = model.classifier[-1].weight.grad.detach().clone()
    return grad.flatten(), pred_idx.item()


def calculate_real_influence(model, train_loader, filenames, test_tensor, test_target, device):
    model.to(device)
    model.eval()
    test_tensor = test_tensor.to(device)
    test_target = torch.tensor([test_target]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Calculating gradient for the test image...")
    test_grad, _ = get_gradient_and_prediction(model, test_tensor, test_target, loss_fn)
    global_idx = 0
    results = []
    print("Iterating through the training dataset to calculate influence scores...")
    for (train_imgs, train_labels) in tqdm(train_loader):
        train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
        for i in range(len(train_imgs)):
            train_grad, train_pred = get_gradient_and_prediction(model, train_imgs[i].unsqueeze(0), train_labels[i].unsqueeze(0), loss_fn)
            influence_score = torch.dot(test_grad, train_grad).item()
            results.append({'score': influence_score, 'prediction': train_pred, 'filename': filenames[global_idx]})
            global_idx += 1
    return results


def calculate_influence(model, input_tensor, predicted_class_index, training_dataset, filenames):
    try:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=False)

      influence_results = calculate_real_influence(model, training_dataloader, filenames, input_tensor, predicted_class_index, device)
      # The ground truth is now part of the dataset, not a separate CSV
      gt_map = {fname: label.item() for (_, label), fname in zip(training_dataset, filenames)}

      label_map = {0: 'Benign', 1: 'Malignant'}
      report_data = [{'case_id': r['filename'].split('.')[0], 
                      'influence_score': r['score'], 
                      'ground_truth': label_map.get(gt_map.get(r['filename'], -1), 'Unknown'), # Look up ground truth
                      'prediction': label_map.get(r['prediction'], 'Unknown')} for r in influence_results]

      report_df = pd.DataFrame(report_data)
      report_df['abs_influence'] = report_df['influence_score'].abs()
      report_df = report_df.sort_values(by='abs_influence', ascending=False).drop(columns='abs_influence')
      return report_df.head(100) # Return top 100 most influential cases

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
