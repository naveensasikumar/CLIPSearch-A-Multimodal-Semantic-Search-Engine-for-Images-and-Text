from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip():
    try:
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to("cpu")
        model.eval()
        print("CLIP model loaded successfully")
        return model, processor
    except Exception as e:
        print(f"Failed to load CLIP: {e}")
        raise e

def get_image_embedding(image, model, processor):
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features[0].cpu().numpy()
    except Exception as e:
        print(f"Image embedding error: {e}")
        raise e

def get_text_embedding(text, model, processor):
    try:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        return features[0].cpu().numpy()
    except Exception as e:
        print(f"Text embedding error: {e}")
        raise e
