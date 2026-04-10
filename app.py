# app.py - UPDATED with better XAI error handling and debug endpoints

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import base64
from typing import Dict
import time
import traceback
import torch
import numpy as np

from model_loader import model_loader
from predict import predictor
from gradcam import get_gradcam_for_model
from captum_explain import get_captum_explanation
from occlusion import get_occlusion_explanation

app = FastAPI(title="TB Detection Web App", description="Clinical decision support for TB detection")

# Mount frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Store current image tensor and original image for XAI
current_input_tensor = None
current_original_image = None

@app.get("/")
async def serve_index():
    """Serve the main upload page"""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/results")
async def serve_results():
    """Serve the results page"""
    with open("frontend/results.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make predictions on uploaded chest X-ray"""
    global current_input_tensor, current_original_image
    
    # Read and validate image
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    
    try:
        # Process image
        from PIL import Image
        image_bytes = io.BytesIO(contents)
        input_tensor, original_image = model_loader.preprocess_image(image_bytes)
        
        # Store for XAI endpoints
        current_input_tensor = input_tensor
        current_original_image = original_image
        
        # Get predictions
        results = predictor.predict_all(input_tensor)
        
        # Convert image to base64 for frontend display
        buffer = io.BytesIO()
        original_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "predictions": results,
            "image": f"data:image/png;base64,{img_base64}"
        })
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/xai/gradcam/{model_name}")
async def gradcam_explanation(model_name: str):
    """Generate Grad-CAM++ explanation"""
    global current_input_tensor, current_original_image
    
    if current_input_tensor is None or current_original_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded yet")
    
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not loaded")
    
    try:
        print(f"Generating Grad-CAM for {model_name}...")
        gradcam_img = get_gradcam_for_model(
            model_name, 
            current_input_tensor, 
            current_original_image
        )
        print(f"Grad-CAM generation successful for {model_name}")
        return JSONResponse({"explanation": gradcam_img})
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating Grad-CAM: {str(e)}")

@app.post("/xai/captum/{model_name}")
async def captum_explanation(model_name: str):
    """Generate Captum Integrated Gradients explanation"""
    global current_input_tensor, current_original_image
    
    if current_input_tensor is None or current_original_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded yet")
    
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not loaded")
    
    try:
        print(f"Generating Captum explanation for {model_name}...")
        captum_img = get_captum_explanation(
            model_name,
            current_input_tensor,
            current_original_image
        )
        print(f"Captum explanation successful for {model_name}")
        return JSONResponse({"explanation": captum_img})
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating Captum explanation: {str(e)}")

@app.post("/xai/occlusion/{model_name}")
async def occlusion_explanation(model_name: str):
    """Generate Occlusion sensitivity map (slow operation)"""
    global current_input_tensor, current_original_image
    
    if current_input_tensor is None or current_original_image is None:
        raise HTTPException(status_code=400, detail="No image uploaded yet")
    
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not loaded")
    
    try:
        print(f"Generating Occlusion map for {model_name} (this may take 30-60 seconds)...")
        occlusion_img = get_occlusion_explanation(
            model_name,
            current_input_tensor,
            current_original_image
        )
        print(f"Occlusion map generation successful for {model_name}")
        return JSONResponse({"explanation": occlusion_img})
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating occlusion map: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = model_loader.get_model_names()
    return {
        "status": "healthy",
        "loaded_models": loaded_models,
        "device": str(model_loader.device)
    }

# =====================================================
# DEBUG ENDPOINTS - Add these after health check
# =====================================================

@app.get("/debug/layers/{model_name}")
async def debug_model_layers(model_name: str):
    """Debug endpoint to see available layers"""
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        return JSONResponse({"error": f"Model {model_name} not loaded"})
    
    model = model_loader.models[model_name]
    
    # Get all layers with their shapes
    layers_info = []
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['norm', 'block', 'layer', 'feature', 'head', 'classifier']):
            layers_info.append({
                "name": name,
                "type": module.__class__.__name__
            })
    
    return JSONResponse({
        "model": model_name,
        "total_relevant_layers": len(layers_info),
        "relevant_layers": layers_info[-30:]  # Last 30 relevant layers
    })

@app.get("/debug/all_layers/{model_name}")
async def debug_all_layers(model_name: str):
    """Debug endpoint to see ALL layers"""
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        return JSONResponse({"error": f"Model {model_name} not loaded"})
    
    model = model_loader.models[model_name]
    
    all_layers = []
    for name, module in model.named_modules():
        all_layers.append({
            "name": name,
            "type": module.__class__.__name__
        })
    
    return JSONResponse({
        "model": model_name,
        "total_layers": len(all_layers),
        "all_layers": all_layers
    })

@app.get("/debug/shapes/{model_name}")
async def debug_model_shapes(model_name: str):
    """Debug endpoint to see model input/output shapes"""
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        return JSONResponse({"error": f"Model {model_name} not loaded"})
    
    model = model_loader.models[model_name]
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(model_loader.device)
    
    # Store intermediate shapes
    shapes = {}
    
    def hook_fn(module, input, output):
        module_name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            shapes[module_name] = output.shape
        else:
            shapes[module_name] = "Not a tensor"
    
    # Register hooks to all modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
            final_output = output.shape if isinstance(output, torch.Tensor) else "Not a tensor"
    except Exception as e:
        final_output = f"Error: {str(e)}"
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get last 20 shapes for readability
    shape_list = list(shapes.items())
    last_shapes = shape_list[-20:] if len(shape_list) > 20 else shape_list
    
    return JSONResponse({
        "model": model_name,
        "input_shape": dummy_input.shape,
        "output_shape": final_output,
        "intermediate_shapes": dict(last_shapes),
        "total_layers_traced": len(shapes)
    })

@app.get("/debug/prediction/{model_name}")
async def debug_prediction(model_name: str):
    """Debug endpoint to test model prediction"""
    global current_input_tensor
    
    if current_input_tensor is None:
        return JSONResponse({"error": "No image uploaded yet. Please upload an image first."})
    
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        return JSONResponse({"error": f"Model {model_name} not loaded"})
    
    model = model_loader.models[model_name]
    model.eval()
    
    with torch.no_grad():
        output = model(current_input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        pred_class = torch.argmax(output, dim=1).item()
        
    return JSONResponse({
        "model": model_name,
        "prediction": "Tuberculosis" if pred_class == 1 else "Normal",
        "confidence": float(probabilities[pred_class] * 100),
        "class_0_prob": float(probabilities[0] * 100),
        "class_1_prob": float(probabilities[1] * 100)
    })

@app.get("/debug/target_layers/{model_name}")
async def debug_recommended_layers(model_name: str):
    """Get recommended target layers for Grad-CAM"""
    if model_name not in model_loader.models or model_loader.models[model_name] is None:
        return JSONResponse({"error": f"Model {model_name} not loaded"})
    
    model = model_loader.models[model_name]
    
    recommended_layers = []
    
    if model_name == 'convnext':
        recommended_layers = [
            {"name": "features[-1]", "type": "Last convolutional layer", "priority": "high"},
            {"name": "features[-2]", "type": "Second last convolutional layer", "priority": "medium"}
        ]
    
    elif model_name == 'vit':
        for name, module in model.named_modules():
            if 'encoder.layers' in name and 'ln_1' in name:
                recommended_layers.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "priority": "high"
                })
            elif 'encoder.layers' in name and 'mlp' in name:
                recommended_layers.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "priority": "medium"
                })
    
    elif model_name == 'swin':
        for name, module in model.named_modules():
            if name == 'norm':
                recommended_layers.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "priority": "high"
                })
            elif 'layers' in name and 'norm' in name and 'blocks' in name:
                recommended_layers.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "priority": "medium"
                })
    
    # If no specific recommendations found, add some general ones
    if not recommended_layers:
        for name, module in list(model.named_modules())[-10:]:
            if hasattr(module, 'weight') or hasattr(module, 'out_features'):
                recommended_layers.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "priority": "low"
                })
    
    return JSONResponse({
        "model": model_name,
        "recommended_layers": recommended_layers,
        "note": "Use 'norm' layer for Swin and 'encoder.layers[-1].ln_1' for ViT for best results"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)