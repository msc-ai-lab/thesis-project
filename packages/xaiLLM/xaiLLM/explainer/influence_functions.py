import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight

from xaiLLM.explainer.XaiModel import XaiModel
from xaiLLM.explainer.wrapper import WrapperModel


class InfluenceFunctions(XaiModel):
    def __init__(self, model: nn.Module, training_dataset: Dataset, filenames: list):
        """
        Initialize the Influence Functions explainer with a PyTorch model.
        
        Args:
            model (nn.Module): The PyTorch model to be explained.
        """
        super().__init__(model)
        self.training_dataset = training_dataset
        self.filenames = filenames
        self.training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=False)

    def _get_gradient_and_prediction(self, data_tensor, target, loss_fn) -> tuple:
        """
        Calculate the gradient and prediction for a given input tensor.

        Parameters
        ----------
        data_tensor : torch.Tensor
            The input tensor for which to compute the gradient and prediction.
        target : torch.Tensor
            The ground truth labels.
        loss_fn : nn.Module
            The loss function to use for backpropagation.

        Returns
        -------
        tuple
            A tuple containing the gradient and the predicted class index.
        """
        model_wrapper = WrapperModel(self.model)
        self.model.zero_grad()

        logits = model_wrapper.get_logits(data_tensor)
        _, pred_idx = torch.max(logits.data, 1)
        loss = loss_fn(logits, target)
        loss.backward()
        # Get gradient from the final classifier layer of the ResNet model
        grad = self.model.classifier[-1].weight.grad.detach().clone()
        return grad.flatten(), pred_idx.item()


    def _calculate_real_influence(self, test_tensor, test_target, device) -> list:
        """
        Calculate the influence of training examples on an image classification using gradient similarity.

        Parameters
        ----------
        test_tensor : torch.Tensor
            The input tensor for the test example.
        test_target : torch.Tensor
            The ground truth label for the test example.
        device : torch.device
            The device to perform calculations on (CPU or GPU).

        Returns
        -------
        list
            A list of influence scores for each training example.
        """
        self.model.to(device)
        test_tensor = test_tensor.to(device)
        test_target = torch.tensor([test_target]).to(device)

        train_labels = [label.item() for _, label in self.training_dataset]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        print("\tCalculating gradient for the test image...")
        test_grad, _ = self._get_gradient_and_prediction(test_tensor, test_target, loss_fn)
        global_idx = 0
        results = []
        print("\tIterating through the training dataset to calculate influence scores...\n")
        for (train_imgs, train_labels) in tqdm(self.training_dataloader):
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            for i in range(len(train_imgs)):
                train_grad, train_pred = self._get_gradient_and_prediction(train_imgs[i].unsqueeze(0), train_labels[i].unsqueeze(0), loss_fn)
                influence_score = torch.dot(test_grad, train_grad).item()
                results.append({'score': influence_score, 'prediction': train_pred, 'filename': self.filenames[global_idx]})
                global_idx += 1
        return results


    def _influence_functions_stats(self, predicted_class: str, influencers: pd.DataFrame) -> tuple:
        """
        Calculate statistics for influence functions.

        Parameters
        ----------
        predicted_class : str
            The predicted class label for the input image.
        influencers : pd.DataFrame
            A DataFrame containing influential training cases.

        Returns
        -------
        tuple
            A tuple containing:
            - groundtruth_alignment_percentage: Percentage of influential training cases that share ground truth with predicted class.
            - groundtruth_misalignment_percentage: Percentage of influential training cases whose ground truth does NOT match the predicted class.
            - misclassified_percentage: Percentage of aligned cases that were misclassified during training.
        """
        total_influencers = len(influencers)

        # Filter for influential training cases that share ground truth with predicted class
        aligned_groundtruth = influencers[influencers['ground_truth'] == predicted_class]
        print(len(aligned_groundtruth))

        # Set default values
        groundtruth_alignment_percentage, groundtruth_misalignment_percentage, misclassified_percentage = 0, 100, 0

        # Check for the count of aligned cases
        if len(aligned_groundtruth) > 0:
            # Calculate the percentage of influential training cases that share ground truth with predicted class
            groundtruth_alignment_percentage = round((len(aligned_groundtruth) / total_influencers) * 100, 2)

            # Calculate the percentage of the aligned cases that were misclassified during training
            misclassified_percentage = round((len(aligned_groundtruth[aligned_groundtruth["ground_truth"] != aligned_groundtruth["prediction"]]) / len(aligned_groundtruth)) * 100, 2)

        # Calculate the percentage of influential training cases whose ground truth does NOT match the predicted class
        groundtruth_misalignment_percentage = total_influencers - groundtruth_alignment_percentage

        return groundtruth_alignment_percentage, groundtruth_misalignment_percentage, misclassified_percentage


    def generate(self, input_tensor: torch.Tensor, predicted_class_index: int) -> tuple:
        """
        Calculate the influence of training examples on a given input tensor using gradient similarity.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor for which to calculate influence.
        predicted_class_index : int
            The index of the predicted class.

        Returns
        -------
        tuple
            A tuple containing:
            - influencers: A DataFrame containing the top 100 influential training cases.
            - influence_stats: A tuple containing statistics about the influence functions.
        """
        try:
            print("Calculating influence...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            

            influence_results = self._calculate_real_influence(input_tensor, predicted_class_index, device)
            # The ground truth is now part of the dataset, not a separate CSV
            gt_map = {fname: label.item() for (_, label), fname in zip(self.training_dataset, self.filenames)}

            label_map = {0: 'Benign', 1: 'Malignant'}
            report_data = [{'case_id': r['filename'].split('.')[0], 
                            'influence_score': r['score'], 
                            'ground_truth': label_map.get(gt_map.get(r['filename'], -1), 'Unknown'), # Look up ground truth
                            'prediction': label_map.get(r['prediction'], 'Unknown')} for r in influence_results]

            report_df = pd.DataFrame(report_data)
            report_df['abs_influence'] = report_df['influence_score'].abs()
            report_df = report_df.sort_values(by='abs_influence', ascending=False).drop(columns='abs_influence')
            influencers = report_df.head(100)

            groundtruth_alignment_percentage, groundtruth_misalignment_percentage, misclassified_percentage = self._influence_functions_stats(
                predicted_class=label_map.get(predicted_class_index, 'Unknown'),
                influencers=influencers
            )
            return influencers, (groundtruth_alignment_percentage, groundtruth_misalignment_percentage, misclassified_percentage)

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
