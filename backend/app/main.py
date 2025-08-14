import shutil
from pathlib import Path
import torch
import base64
from PIL import Image
from io import BytesIO
import os
import sys
import traceback
import matplotlib
import json
import asyncio
from asyncio import Queue

matplotlib.use('Agg')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

IS_DOCKER = os.environ.get('IS_DOCKER', True)
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = ROOT_DIR / "user_uploads"
MODEL_PATH = ROOT_DIR / "models" / "ResNet_skin_cancer_classification.pth"
DATASET_PATH = ROOT_DIR / "data" / "processed" / "train_dataset.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_RESIZE = (384, 384)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ANALYSIS_TIMEOUT = 600  # 10 minutes

model = None
try:
    model = load_model(str(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def pil_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 encoded string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def pil_to_bytes(image: Image.Image):
    """Convert PIL Image to bytes"""
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

async def run_analysis_in_background(image_path: Path, queue: Queue):
    """
    Run the complete analysis pipeline in the background.
    Sends progress updates through the queue.
    """
    if not model:
        await queue.put(json.dumps({"error": "Model is not loaded"}))
        return
    
    try:
        total_steps = 7
        current_step = 0
        
        async def update_progress(status: str, detail: str = None):
            nonlocal current_step
            current_step += 1
            progress = (current_step / total_steps) * 100
            message = {
                "status": status,
                "progress": progress,
                "step": f"{current_step}/{total_steps}"
            }
            if detail:
                message["detail"] = detail
            await queue.put(json.dumps(message))
        
        # Step 1: Preprocessing
        await update_progress("Preprocessing input...", "Resizing and normalizing image")
        image_tensor = await asyncio.to_thread(preprocess_input, str(image_path), resize=IMAGE_RESIZE)
        image_tensor = image_tensor.to(DEVICE)
        original_image = await asyncio.to_thread(Image.open, image_path)
        original_image = original_image.convert('RGB').resize(IMAGE_RESIZE, Image.Resampling.BILINEAR)

        # Step 2: Prediction
        await update_progress("Making prediction...", "Running CNN inference")
        (pred_idx, pred_class), probs = await asyncio.to_thread(predict, model, image_tensor)

        # Step 3: Grad-CAM
        await update_progress("Generating Grad-CAM explanation...", "Computing gradient-based heatmap")
        grad_cam_explainer = GradCAMWeb(model)
        gradcam_viz = await asyncio.to_thread(grad_cam_explainer.generate, image_tensor, original_image, pred_idx)

        # Step 4: SHAP
        await update_progress("Generating SHAP explanation...", "This may take a few minutes")
        shap_explainer = SHAPExplainerWeb(model)
        shap_viz = await asyncio.to_thread(shap_explainer.generate, image_tensor, original_image, pred_idx)

        # Step 5: Load dataset
        await update_progress("Loading dataset for influence functions...", "Preparing training data")
        dataset, filenames = await asyncio.to_thread(xai_load_datasets, str(DATASET_PATH))
        
        # Step 6: Influence Functions
        await update_progress("Calculating influence functions...", "This is the longest step")
        influence_explainer = InfluenceFunctions(model, dataset, filenames)
        influencers, influence_stats = await asyncio.to_thread(influence_explainer.generate, image_tensor, pred_idx)

        # Step 7: LLM Report
        await update_progress("Generating LLM report...", "Creating comprehensive analysis")
        interpreter = LLMInterpreter()
        llm_output = await asyncio.to_thread(
            interpreter.inference,
            probabilities=probs,
            influence_stats=influence_stats,
            xai_gradcam_enc=base64.b64encode(pil_to_bytes(gradcam_viz)).decode('utf-8'),
            xai_shap_enc=base64.b64encode(pil_to_bytes(shap_viz)).decode('utf-8'),
            input_image_enc=base64.b64encode(pil_to_bytes(original_image)).decode('utf-8')
        )

        # Send final result
        final_result = {
            "prediction": pred_class,
            "probabilities": probs,
            "llm_report": llm_output,
            "influencers_top5": influencers.head(5).to_dict(orient='records'),
            "images": {
                "original": pil_image_to_base64(original_image),
                "grad_cam": pil_image_to_base64(gradcam_viz),
                "shap": pil_image_to_base64(shap_viz)
            }
        }
        await queue.put(json.dumps({"data": final_result, "complete": True}))

    except asyncio.CancelledError:
        await queue.put(json.dumps({"error": "Analysis was cancelled"}))
        raise
    except Exception as e:
        traceback.print_exc()
        error_message = f"An error occurred during analysis: {str(e)}"
        await queue.put(json.dumps({"error": error_message}))
    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception as e:
            print(f"Error cleaning up file: {e}")
        await queue.put(None)

@app.on_event("startup")
def on_startup():
    """Create upload directory on startup"""
    UPLOAD_DIR.mkdir(exist_ok=True)
    print(f"✅ Upload directory ready: {UPLOAD_DIR}")

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "API is running",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "upload_dir": str(UPLOAD_DIR),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }

@app.get("/health")
def health_check():
    """Detailed health check for monitoring"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded", "device": str(DEVICE)}

@app.post("/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Analyze an uploaded image for skin cancer detection.
    Returns a stream of Server-Sent Events with progress updates.
    """
    file_path = None
    
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG)")
        
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f}MB"
            )
        
        file_path = UPLOAD_DIR / f"{asyncio.get_event_loop().time()}_{file.filename}"
        with file_path.open("wb") as buffer:
            buffer.write(contents)
        
        async def event_stream():
            """Generate SSE stream with progress updates"""
            queue = Queue()
            analysis_task = asyncio.create_task(run_analysis_in_background(file_path, queue))
            heartbeat_count = 0
            
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=10.0)
                        
                        if message is None:
                            break
                        
                        yield f"data: {message}\n\n"
                        
                    except asyncio.TimeoutError:
                        heartbeat_count += 1
                        heartbeat_msg = json.dumps({
                            "heartbeat": True,
                            "status": "Processing...",
                            "heartbeat_count": heartbeat_count
                        })
                        yield f"data: {heartbeat_msg}\n\n"
                        
                        if analysis_task.done():
                            break
                            
            except asyncio.CancelledError:
                print("Client disconnected, cancelling analysis task")
                analysis_task.cancel()
                try:
                    await analysis_task
                except asyncio.CancelledError:
                    pass
                raise
            except Exception as e:
                error_msg = json.dumps({"error": f"Stream error: {str(e)}"})
                yield f"data: {error_msg}\n\n"
                analysis_task.cancel()
                raise
            finally:
                if not analysis_task.done():
                    try:
                        await analysis_task
                    except asyncio.CancelledError:
                        pass
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except HTTPException:
        if file_path and file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        if file_path and file_path.exists():
            file_path.unlink()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.middleware("http")
async def add_timeout_middleware(request, call_next):
    """Add timeout to all requests"""
    try:
        return await asyncio.wait_for(call_next(request), timeout=ANALYSIS_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timeout - analysis took too long"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)