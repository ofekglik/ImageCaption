import os
os.environ['PYTHONPATH'] = os.getcwd()
os.environ['MPS_ENABLE_DEVICE'] = '1'

import numpy as np

import torch
torch.manual_seed(42)  # For reproducibility
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import shutil
import os
from pathlib import Path
import uvicorn
from PIL import Image
from torchvision import transforms
from typing import Tuple

from src.modeling import load_model_state
from src.utils import generate_caption_for_image

# Update device setup for Mac MPS
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)

# Load the model
try:
    print(f"Loading model from: {os.path.abspath('my_model.pt')}")
    model, state = load_model_state('my_model.pt')
    print("Model loaded successfully")
    print(f"Vocab size: {len(model.vocab)}")
    model = model.to(device)
    model.eval()
    print(f"Model moved to device: {device}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

app = FastAPI()

# Create directories if they don't exist
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return templates.TemplateResponse(
                "upload.html",
                {"request": request, "error": "Please upload an image file."}
            )
        
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate caption for the image
        caption = generate_caption_for_image(file_path, model, device='mps')
        
        # Generate the image URL
        image_url = f"/static/uploads/{file.filename}"
        
        return templates.TemplateResponse(
            "display.html",
            {
                "request": request,
                "image": image_url,
                "text": text,
                "caption": caption
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "error": f"Error processing upload: {str(e)}"}
        )

@app.get('/favicon.ico')
async def favicon():
    return FileResponse('../Flick_30k/static/images/favicon.ico')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
