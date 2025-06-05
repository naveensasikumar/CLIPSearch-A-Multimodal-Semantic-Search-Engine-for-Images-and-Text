# Replace your entire backend/model.py with this:

from transformers import CLIPProcessor, CLIPModel
import torch
import warnings

warnings.filterwarnings("ignore")

def load_clip():
    """Load CLIP model avoiding meta device issues"""
    print("Loading CLIP model...")
    
    try:
        # Method 1: Load with init_empty_weights=False
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=False,  # This prevents meta device
            torch_dtype=torch.float32
        )
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Force CPU without using .to() method that causes meta device issues
        model.cpu()
        model.eval()
        
        print("✅ CLIP model loaded successfully")
        return model, processor
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        # Method 2: Use to_empty() for meta device tensors
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Handle meta device tensors properly
            device = torch.device("cpu")
            model = model.to_empty(device=device)
            
            # Reload the state dict to populate the empty tensors
            state_dict = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").state_dict()
            model.load_state_dict(state_dict)
            model.eval()
            
            print("✅ CLIP model loaded with to_empty method")
            return model, processor
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            # Method 3: Force local loading
            try:
                import os
                
                # Disable meta device usage entirely
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    force_download=False,
                    local_files_only=False,
                    device_map=None
                )
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                # Just use cpu() instead of to()
                model.cpu()
                model.eval()
                
                print("✅ CLIP model loaded with method 3")
                return model, processor
                
            except Exception as e3:
                print(f"All methods failed. Last error: {e3}")
                raise Exception("Could not load CLIP model. Try: pip install --upgrade transformers==4.35.0")

def get_image_embedding(image, model, processor):
    """Get image embedding"""
    try:
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        
        # Ensure inputs are on CPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        
        return features[0].detach().cpu().numpy()
        
    except Exception as e:
        print(f"Image embedding error: {e}")
        raise e

def get_text_embedding(text, model, processor):
    """Get text embedding"""
    try:
        # Process text
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        
        # Ensure inputs are on CPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        
        return features[0].detach().cpu().numpy()
        
    except Exception as e:
        print(f"Text embedding error: {e}")
        raise e

# Alternative: If above still fails, use this simpler version:

def load_clip_simple():
    """Ultra-simple CLIP loading"""
    try:
        # Try with older transformers behavior
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Don't use .to() at all, just ensure it's on CPU by default
        model.eval()
        
        return model, processor
        
    except Exception as e:
        print(f"Simple loading failed: {e}")
        raise e

# If the main functions don't work, uncomment these lines in your app.py:
# try:
#     clip_model, clip_processor = load_clip()
# except:
#     print("Trying simple loading...")
#     clip_model, clip_processor = load_clip_simple()