# occlusion.py - CORRECTED VERSION with optimizations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64
from tqdm import tqdm


def generate_occlusion_map(model, input_tensor, original_image,
                           patch_size=32, stride=16):
    """Generate occlusion sensitivity map (OPTIMIZED CORRECTED VERSION)"""
    
    device = input_tensor.device
    h, w = 224, 224
    
    # =====================================================
    # STEP 1: Get baseline prediction
    # =====================================================
    with torch.no_grad():
        baseline_output = model(input_tensor)
        baseline_probs = F.softmax(baseline_output, dim=1)
    
    pred_class = torch.argmax(baseline_probs, dim=1).item()
    baseline_prob = baseline_probs[0, pred_class].item()
    
    # =====================================================
    # STEP 2: Create importance map
    # =====================================================
    importance_map = np.zeros((h, w))
    
    # Better occlusion value: ImageNet mean (normalized)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    
    # =====================================================
    # STEP 3: Calculate number of patches for progress tracking
    # =====================================================
    num_patches_y = (h - patch_size) // stride + 1
    num_patches_x = (w - patch_size) // stride + 1
    total_patches = num_patches_y * num_patches_x
    
    # =====================================================
    # STEP 4: Sliding window occlusion (with progress tracking)
    # =====================================================
    print(f"Generating occlusion map for {model.__class__.__name__}...")
    print(f"Processing {total_patches} patches...")
    
    patch_count = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_count += 1
            
            # Occlude patch
            occluded = input_tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = mean
            
            with torch.no_grad():
                occluded_output = model(occluded)
                occluded_probs = F.softmax(occluded_output, dim=1)
            
            occluded_prob = occluded_probs[0, pred_class].item()
            
            # Importance = drop in confidence
            importance = baseline_prob - occluded_prob
            
            # Add to importance map (weighted by patch size)
            importance_map[y:y+patch_size, x:x+patch_size] += importance
            
            # Print progress every 10%
            if patch_count % max(1, total_patches // 10) == 0:
                progress = (patch_count / total_patches) * 100
                print(f"Progress: {progress:.1f}%")
    
    print("Occlusion map generation complete!")
    
    # =====================================================
    # STEP 5: Normalize properly (removed duplicate blur)
    # =====================================================
    # Apply Gaussian blur once for smoothing
    importance_map = cv2.GaussianBlur(importance_map, (9, 9), 0)
    
    # Normalize to [0, 1]
    if importance_map.max() > importance_map.min():
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-7)
    
    # =====================================================
    # STEP 6: Apply colormap
    # =====================================================
    importance_map_uint8 = np.uint8(255 * importance_map)
    importance_map_resized = cv2.resize(importance_map_uint8, (224, 224))
    
    heatmap = cv2.applyColorMap(importance_map_resized, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # =====================================================
    # STEP 7: Overlay on original image
    # =====================================================
    if isinstance(original_image, Image.Image):
        original_image_np = np.array(original_image.resize((224, 224)))
    else:
        original_image_np = original_image
    
    # Convert to RGB if grayscale
    if len(original_image_np.shape) == 2:
        original_image_np = np.stack([original_image_np] * 3, axis=2)
    
    # Blend original and heatmap
    blended = cv2.addWeighted(original_image_np.astype(np.uint8), 0.5, heatmap, 0.5, 0)
    
    # =====================================================
    # STEP 8: Convert to base64
    # =====================================================
    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"


def get_occlusion_explanation(model_name, input_tensor, original_image):
    """Wrapper function for occlusion explanation"""
    from model_loader import model_loader
    
    model = model_loader.models[model_name]
    model.eval()
    
    # Move to device
    input_tensor = input_tensor.to(model_loader.device)
    
    # Generate occlusion map
    return generate_occlusion_map(model, input_tensor, original_image)