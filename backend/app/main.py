import shutil
from pathlib import Path
import torch
import base64
from PIL import Image
from io import BytesIO
import os
import sys
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

IS_DOCKER = os.environ.get('IS_DOCKER', False)
if IS_DOCKER:
    ROOT_DIR = Path("/app")
    sys.path.append(str(ROOT_DIR / "packages"))
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from scd.preprocess import preprocess_input
from scd.inference import predict
from scd.utils.common import load_model
from xaiLLM.explainer.grad_cam_web import GradCAMWeb
from xaiLLM.explainer.shap_web import SHAPExplainerWeb
from xaiLLM.explainer.influence_functions import InfluenceFunctions
from xaiLLM.interpreter.llm_interpreter import LLMInterpreter
from xaiLLM.utils.helpers import load_datasets as xai_load_datasets

app = FastAPI(title="Skin Cancer XAI Thesis Project")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = ROOT_DIR / "user_uploads"
MODEL_PATH = ROOT_DIR / "models" / "ResNet_skin_cancer_classification.pth"
DATASET_PATH = ROOT_DIR / "data" / "processed" / "train_dataset.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_RESIZE = (384, 384)
model = None
try:
    model = load_model(str(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def pil_to_bytes(image: Image.Image):
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def run_full_analysis(image_path: Path):
    if not model:
        raise RuntimeError("Model is not loaded.")
    image_tensor = preprocess_input(str(image_path), resize=IMAGE_RESIZE).to(DEVICE)
    original_image = Image.open(image_path).convert('RGB').resize(IMAGE_RESIZE, Image.Resampling.BILINEAR)
    (pred_idx, pred_class), probs = predict(model, image_tensor)
    grad_cam_explainer = GradCAMWeb(model)
    gradcam_viz = grad_cam_explainer.generate(image_tensor, original_image, pred_idx)
    shap_explainer = SHAPExplainerWeb(model)
    shap_viz = shap_explainer.generate(image_tensor, original_image, pred_idx)
    dataset, filenames = xai_load_datasets(str(DATASET_PATH))
    influence_explainer = InfluenceFunctions(model, dataset, filenames)
    influencers, influence_stats = influence_explainer.generate(image_tensor, pred_idx)
    interpreter = LLMInterpreter()
    llm_output = interpreter.inference(
        probabilities=probs,
        influence_stats=influence_stats,
        xai_gradcam_enc=base64.b64encode(pil_to_bytes(gradcam_viz)).decode('utf-8'),
        xai_shap_enc=base64.b64encode(pil_to_bytes(shap_viz)).decode('utf-8'),
        input_image_enc=base64.b64encode(pil_to_bytes(original_image)).decode('utf-8')
    )
    return {
        "prediction": pred_class, "probabilities": probs, "llm_report": llm_output,
        "influencers_top5": influencers.head(5).to_dict(orient='records'),
        "images": {
            "original": pil_image_to_base64(original_image),
            "grad_cam": pil_image_to_base64(gradcam_viz),
            "shap": pil_image_to_base64(shap_viz)
        }
    }

@app.on_event("startup")
def on_startup():
    UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results = run_full_analysis(file_path)
        
        file_path.unlink()
        return JSONResponse(content=results)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
