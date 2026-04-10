import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import torch.nn as nn

class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.models = {}
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models from .pth files"""
        
        # Load ConvNeXt-Tiny
        try:
            print("\nLoading ConvNeXt-Tiny...")
            # Create model with 2 classes
            convnext = models.convnext_tiny(weights=None)
            # Replace the classifier head for 2 classes
            convnext.classifier[2] = nn.Linear(768, 2)
            
            # Load the state dict
            state_dict = torch.load('models/convnext_tb.pth', map_location=self.device)
            
            # Load with strict=False to allow flexibility
            convnext.load_state_dict(state_dict, strict=False)
            convnext = convnext.to(self.device)
            convnext.eval()
            self.models['convnext'] = convnext
            print("✓ ConvNeXt-Tiny loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load ConvNeXt: {e}")
            import traceback
            traceback.print_exc()
            self.models['convnext'] = None
        
        # Load Vision Transformer ViT-B/16
        try:
            print("\nLoading Vision Transformer ViT-B/16...")
            # Create model with 2 classes
            vit = models.vit_b_16(weights=None)
            # Replace the head for 2 classes
            vit.heads.head = nn.Linear(768, 2)
            
            # Load the state dict
            vit_state = torch.load('models/vit_tb.pth', map_location=self.device)
            
            # Load with strict=False
            vit.load_state_dict(vit_state, strict=False)
            vit = vit.to(self.device)
            vit.eval()
            self.models['vit'] = vit
            print("✓ ViT-B/16 loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load ViT: {e}")
            import traceback
            traceback.print_exc()
            self.models['vit'] = None
        
        # Load Swin Transformer
        try:
            print("\nLoading Swin Transformer...")
            import timm
            swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
            swin_state = torch.load('models/swin_tb.pth', map_location=self.device)
            swin.load_state_dict(swin_state)
            swin = swin.to(self.device)
            swin.eval()
            self.models['swin'] = swin
            print("✓ Swin Transformer loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load Swin: {e}")
            self.models['swin'] = None
    
    def preprocess_image(self, image_bytes):
        """Preprocess uploaded image for model input"""
        image = Image.open(image_bytes).convert('RGB')
        input_tensor = self.transforms(image).unsqueeze(0)
        return input_tensor.to(self.device), image
    
    def get_model_names(self):
        """Return list of successfully loaded model names"""
        return [name for name, model in self.models.items() if model is not None]

# Global instance
model_loader = ModelLoader()