from transformers import CLIPProcessor, CLIPModel
import torch
import warnings

warnings.filterwarnings("ignore")

def load_clip():
    print("Loading CLIP model...")
    
    try:
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32
        )
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        model.cpu()
        model.eval()
        
        print(" CLIP model loaded successfully")
        return model, processor
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            device = torch.device("cpu")
            model = model.to_empty(device=device)
            
            state_dict = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").state_dict()
            model.load_state_dict(state_dict)
            model.eval()
            
            print(" CLIP model loaded with to_empty method")
            return model, processor
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            try:
                import os
                
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
                model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    force_download=False,
                    local_files_only=False,
                    device_map=None
                )
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                model.cpu()
                model.eval()
                
                print("✅ CLIP model loaded with method 3")
                return model, processor
                
            except Exception as e3:
                print(f"All methods failed. Last error: {e3}")
                raise Exception("Could not load CLIP model. Try: pip install --upgrade transformers==4.35.0")

def get_image_embedding(image, model, processor):
    try:
        inputs = processor(images=image, return_tensors="pt")
        
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        
        return features[0].detach().cpu().numpy()
        
    except Exception as e:
        print(f"Image embedding error: {e}")
        raise e

def get_text_embedding(text, model, processor):
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        
        return features[0].detach().cpu().numpy()
        
    except Exception as e:
        print(f"Text embedding error: {e}")
        raise e

def load_clip_simple():
    try:
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        model.eval()
        
        return model, processor
        
    except Exception as e:
        print(f"Simple loading failed: {e}")
        raise e