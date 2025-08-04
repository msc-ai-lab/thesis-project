import torch
from torch import nn
from PIL import Image
from xaiLLM.utils.helpers import show_title, load_datasets, extract_size
from xaiLLM.explainer.grad_cam import GradCAM
from xaiLLM.explainer.shap import SHAPExplainer
from xaiLLM.explainer.influence_functions import InfluenceFunctions
from xaiLLM.interpreter.llm_interpreter import LLMInterpreter

def run_xaiLLM(model: nn.Module, image_tensor: torch.Tensor, input_path: str, pred_idx: int, dataset_path: str, probabilities: dict, show_images: bool = True) -> tuple:
    """
    Run the XAI pipeline to generate explanations and interpretations for a given input image.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be explained.
    image_tensor : torch.Tensor
        The preprocessed input image tensor.
    input_path : str
        The path to the input image file.
    pred_idx : int
        The predicted class index from the model.
    dataset_path : str
        The path to the dataset used for influence function calculations.
    probabilities : dict
        The prediction probabilities for the input image.
    show_images : bool, default=True
        Whether to display the generated visualisations.

    Returns
    -------
    tuple
        A tuple containing:
        - gradcam_viz: Grad-CAM visualisation as a PIL Image.
        - shap_viz: SHAP visualisation as a PIL Image.
        - influencers: Influence function output.
        - llm_output: LLM interpretation output as a string.
    """
    try:
        show_title('Running Explainer...')
        size = extract_size(image_tensor)
        image = Image.open(input_path).convert('RGB')
        image = image.resize(size, resample=Image.BILINEAR)

        # Generate Grad-CAM visualisation
        grad_cam = GradCAM(model)
        gradcam_viz = grad_cam.generate(image_tensor, image, predicted_class_index=pred_idx, show_image=show_images)
        gradcam_enc = grad_cam.pil_image_to_base64(gradcam_viz)
        input_image_enc = grad_cam.pil_image_to_base64(image)

        # Generate SHAP visualisation
        shap = SHAPExplainer(model)
        shap_viz = shap.generate(image_tensor, image, predicted_class_index=pred_idx, show_image=show_images)
        shap_enc = shap.pil_image_to_base64(shap_viz)

        # Influence Function
        dataset, filenames = load_datasets(dataset_path)
        influence_functions = InfluenceFunctions(model, dataset, filenames)
        influencers = influence_functions.generate(image_tensor, pred_idx)

        interpreter = LLMInterpreter()
        llm_output = interpreter.inference(
            probs=probabilities,
            influencers=influencers.to_json(),
            xai_gradcam_enc=gradcam_enc,
            xai_shap_enc=shap_enc,
            input_image_enc=input_image_enc
        )

        return gradcam_viz, shap_viz, influencers, llm_output
    except Exception as e:
        print(f"An error occurred while running xaiLLM: {e}")
        return
