# gradcam.py - COMPLETE WORKING VERSION FOR CONVNEXT, VIT, AND SWIN

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64


class GradCAMPlusPlus:
    """Complete Grad-CAM++ implementation for CNN and Transformer models"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cnn_cam(self, activations, gradients):
        """Generate Grad-CAM++ for CNN models (4D tensors)"""
        # Grad-CAM++ weights calculation
        alpha_num = gradients ** 2
        alpha_denom = 2 * gradients ** 2 + (activations * gradients ** 3).sum(dim=(2, 3), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        # Weighted gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)
        return cam
    
    def generate_vit_cam(self, activations, gradients):
        """Generate Grad-CAM++ specifically for ViT"""
        # Remove CLS token (first token)
        if activations.shape[1] == 197:  # ViT has CLS token
            activations = activations[:, 1:, :]
            gradients = gradients[:, 1:, :]
        
        batch_size, num_tokens, feat_dim = activations.shape
        
        # ViT uses 14x14 grid (196 tokens)
        grid_size = 14
        
        # Reshape to spatial format
        activations_spatial = activations.transpose(1, 2).reshape(
            batch_size, feat_dim, grid_size, grid_size
        )
        gradients_spatial = gradients.transpose(1, 2).reshape(
            batch_size, feat_dim, grid_size, grid_size
        )
        
        # Grad-CAM++ weights calculation
        alpha_num = gradients_spatial ** 2
        alpha_denom = 2 * gradients_spatial ** 2 + (
            activations_spatial * gradients_spatial ** 3
        ).sum(dim=(2, 3), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        # Weighted gradients
        weights = (alpha * F.relu(gradients_spatial)).sum(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * activations_spatial).sum(dim=1, keepdim=True)
        
        return cam
    
    def generate_swin_cam(self, activations, gradients):
        """Generate Grad-CAM++ specifically for Swin Transformer"""
        batch_size, num_tokens, feat_dim = activations.shape
        
        # Determine grid size for Swin
        if num_tokens == 49:
            grid_size = 7
        elif num_tokens == 196:
            grid_size = 14
        else:
            grid_size = int(np.sqrt(num_tokens))
        
        # Reshape to spatial format
        activations_spatial = activations.transpose(1, 2).reshape(
            batch_size, feat_dim, grid_size, grid_size
        )
        gradients_spatial = gradients.transpose(1, 2).reshape(
            batch_size, feat_dim, grid_size, grid_size
        )
        
        # Grad-CAM++ weights calculation
        alpha_num = gradients_spatial ** 2
        alpha_denom = 2 * gradients_spatial ** 2 + (
            activations_spatial * gradients_spatial ** 3
        ).sum(dim=(2, 3), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        # Weighted gradients
        weights = (alpha * F.relu(gradients_spatial)).sum(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * activations_spatial).sum(dim=1, keepdim=True)
        
        return cam
    
    def generate(self, input_tensor, target_class, model_name):
        """Generate Grad-CAM++ heatmap with model-specific handling"""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for target class
        output[0, target_class].backward(retain_graph=True)
        
        activations = self.activations
        gradients = self.gradients
        
        if activations is None or gradients is None:
            raise ValueError("Failed to capture activations or gradients")
        
        print(f"  Activation shape: {activations.shape}")
        
        # Model-specific CAM generation
        if model_name == 'convnext':
            cam = self.generate_cnn_cam(activations, gradients)
        elif model_name == 'vit':
            cam = self.generate_vit_cam(activations, gradients)
        elif model_name == 'swin':
            cam = self.generate_swin_cam(activations, gradients)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Convert to numpy and normalize
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Handle different shapes
        if len(cam.shape) == 3:
            cam = cam[0]
        
        # Normalize to [0, 1]
        if cam.max() - cam.min() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        # Resize to 224x224
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        return cam
    
    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        """Overlay heatmap on original image"""
        if isinstance(original_image, Image.Image):
            original = np.array(original_image.resize((224, 224)))
        else:
            original = original_image
        
        if len(original.shape) == 2:
            original = np.stack([original] * 3, axis=2)
        
        # Normalize original if needed
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)


def get_gradcam_for_model(model_name, input_tensor, original_image):
    """Generate Grad-CAM++ explanation for any model"""
    from model_loader import model_loader
    
    # Get model
    model = model_loader.models[model_name]
    model.eval()
    
    # Move input to device
    device = model_loader.device
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        print(f"\n{model_name.upper()} Grad-CAM++")
        print(f"Prediction: {'TB' if pred_class == 1 else 'Normal'}")
        print(f"Confidence: {probabilities[pred_class].item() * 100:.2f}%")
    
    # Select appropriate target layer based on model type
    target_layer = None
    
    if model_name == 'convnext':
        # ConvNeXt: use the last convolutional layer
        target_layer = model.features[-1]
        print(f"Target layer: features[-1] ({target_layer.__class__.__name__})")
        
    elif model_name == 'vit':
        # ViT: use the layer norm after the last attention block
        # This gives the best spatial features
        target_layer = model.encoder.layers[-1].ln_1
        print(f"Target layer: encoder.layers[-1].ln_1 ({target_layer.__class__.__name__})")
        
    elif model_name == 'swin':
        # Swin: use the final norm layer for best results
        # This prevents the "strips" issue
        if hasattr(model, 'norm'):
            target_layer = model.norm
            print(f"Target layer: norm ({target_layer.__class__.__name__})")
        else:
            # Fallback to last block's norm2
            target_layer = model.layers[-1].blocks[-1].norm2
            print(f"Target layer: layers[-1].blocks[-1].norm2 ({target_layer.__class__.__name__})")
    
    if target_layer is None:
        raise ValueError(f"Could not find suitable target layer for {model_name}")
    
    # Create Grad-CAM++ object
    gradcam = GradCAMPlusPlus(model, target_layer)
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, pred_class, model_name)
    
    # Overlay on original image
    blended = gradcam.overlay_heatmap(original_image, heatmap)
    
    # Convert to base64
    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    print(f"✓ Grad-CAM++ generated successfully\n")
    
    return f"data:image/png;base64,{img_base64}"