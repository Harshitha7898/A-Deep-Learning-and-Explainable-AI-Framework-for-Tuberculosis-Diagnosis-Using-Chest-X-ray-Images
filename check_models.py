import torch
import os

print("=" * 60)
print("Checking Model Files")
print("=" * 60)

# Check model files exist and their sizes
model_files = {
    'convnext_tb.pth': 'models/convnext_tb.pth',
    'vit_tb.pth': 'models/vit_tb.pth',
    'swin_tb.pth': 'models/swin_tb.pth'
}

print("\n1. Checking file sizes:")
print("-" * 40)
for name, path in model_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
        print(f"✓ {name}: {size:.2f} MB")
    else:
        print(f"✗ {name}: NOT FOUND")

# Check ConvNeXt
print("\n2. Analyzing ConvNeXt model file:")
print("-" * 40)
try:
    convnext_state = torch.load('models/convnext_tb.pth', map_location='cpu')
    print(f"Type: {type(convnext_state)}")
    
    if isinstance(convnext_state, dict):
        print(f"Top-level keys: {list(convnext_state.keys())}")
        
        # Check for common patterns
        if 'model' in convnext_state:
            print("✓ Found 'model' key - likely contains the actual model weights")
            model_dict = convnext_state['model']
            if isinstance(model_dict, dict):
                sample_keys = list(model_dict.keys())[:3]
                print(f"  Sample keys from model: {sample_keys}")
        elif 'state_dict' in convnext_state:
            print("✓ Found 'state_dict' key - likely contains the actual model weights")
            model_dict = convnext_state['state_dict']
            if isinstance(model_dict, dict):
                sample_keys = list(model_dict.keys())[:3]
                print(f"  Sample keys from state_dict: {sample_keys}")
        else:
            # Direct state dict
            sample_keys = list(convnext_state.keys())[:5]
            print(f"Direct state dict - sample keys: {sample_keys}")
    else:
        print(f"Unexpected type: {type(convnext_state)}")
        
except Exception as e:
    print(f"✗ Error loading ConvNeXt: {e}")

# Check ViT
print("\n3. Analyzing ViT model file:")
print("-" * 40)
try:
    vit_state = torch.load('models/vit_tb.pth', map_location='cpu')
    print(f"Type: {type(vit_state)}")
    
    if isinstance(vit_state, dict):
        print(f"Top-level keys: {list(vit_state.keys())}")
        
        if 'model' in vit_state:
            print("✓ Found 'model' key")
            model_dict = vit_state['model']
            if isinstance(model_dict, dict):
                sample_keys = list(model_dict.keys())[:3]
                print(f"  Sample keys from model: {sample_keys}")
        elif 'state_dict' in vit_state:
            print("✓ Found 'state_dict' key")
            model_dict = vit_state['state_dict']
            if isinstance(model_dict, dict):
                sample_keys = list(model_dict.keys())[:3]
                print(f"  Sample keys from state_dict: {sample_keys}")
        else:
            sample_keys = list(vit_state.keys())[:5]
            print(f"Direct state dict - sample keys: {sample_keys}")
    else:
        print(f"Unexpected type: {type(vit_state)}")
        
except Exception as e:
    print(f"✗ Error loading ViT: {e}")

# Check Swin (this one works)
print("\n4. Analyzing Swin model file (should work):")
print("-" * 40)
try:
    swin_state = torch.load('models/swin_tb.pth', map_location='cpu')
    print(f"Type: {type(swin_state)}")
    
    if isinstance(swin_state, dict):
        print(f"Top-level keys: {list(swin_state.keys())}")
        
        if 'model' in swin_state:
            model_dict = swin_state['model']
            if isinstance(model_dict, dict):
                sample_keys = list(model_dict.keys())[:3]
                print(f"Sample keys from model: {sample_keys}")
        elif 'state_dict' in swin_state:
            model_dict = swin_state['state_dict']
            if isinstance(model_dict, dict):
                sample_keys = list(model_dict.keys())[:3]
                print(f"Sample keys from state_dict: {sample_keys}")
        else:
            sample_keys = list(swin_state.keys())[:5]
            print(f"Sample keys: {sample_keys}")
    print("✓ Swin file loaded successfully")
    
except Exception as e:
    print(f"✗ Error loading Swin: {e}")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)