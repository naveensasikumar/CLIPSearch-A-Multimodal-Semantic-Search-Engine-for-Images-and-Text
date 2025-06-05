import numpy as np
import faiss
import json
import os
from datetime import datetime
from pathlib import Path
import pickle
import streamlit as st

class EnhancedSearchEngine:
    def __init__(self, 
                 image_paths_file="../image_paths.npy",
                 image_embeddings_file="../image_embeddings.npy",
                 collections_file="../data/collections.json",
                 favorites_file="../data/favorites.json"):
        
        self.image_paths_file = image_paths_file
        self.image_embeddings_file = image_embeddings_file
        self.collections_file = collections_file
        self.favorites_file = favorites_file
        
        self.load_search_data()
        self.ensure_data_files()
    
    def load_search_data(self):
        try:
            self.image_paths = np.load(self.image_paths_file, allow_pickle=True)
            self.image_embeddings = np.load(self.image_embeddings_file)
            
            self.image_embeddings = self.image_embeddings / np.linalg.norm(
                self.image_embeddings, axis=1, keepdims=True
            )
            
            d = self.image_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.image_embeddings)
            
            print(f"Loaded {len(self.image_paths)} images for search")
            
        except Exception as e:
            print(f"Error loading search data: {e}")
            self.image_paths = np.array([])
            self.image_embeddings = np.array([])
            self.index = None
    
    def ensure_data_files(self):
        data_dir = Path("../data")
        data_dir.mkdir(exist_ok=True)
        
        for file_path in [self.collections_file, self.favorites_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump({}, f)
    
    def search_similar(self, query_vector, top_k=9, exclude_indices=None):
        if self.index is None:
            return []
        
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = np.array([query_vector], dtype="float32")
        
        search_k = min(top_k * 3, len(self.image_paths))
        D, I = self.index.search(query_vector, search_k)
        
        results = []
        for j, i in enumerate(I[0]):
            if exclude_indices and i in exclude_indices:
                continue
            
            if len(results) >= top_k:
                break
                
            results.append({
                "path": str(self.image_paths[i]),
                "score": float(D[0][j]),
                "index": int(i)
            })
        
        return results
    
    def save_to_favorites(self, image_path, query="", user_id="default"):
        try:
            with open(self.favorites_file, 'r') as f:
                favorites = json.load(f)
        except:
            favorites = {}
        
        if user_id not in favorites:
            favorites[user_id] = []
        
        if not any(fav['path'] == image_path for fav in favorites[user_id]):
            favorites[user_id].append({
                "path": image_path,
                "added_date": datetime.now().isoformat(),
                "query": query
            })
            
            with open(self.favorites_file, 'w') as f:
                json.dump(favorites, f, indent=2)
            return True
        return False
    
    def get_favorites(self, user_id="default"):
        try:
            with open(self.favorites_file, 'r') as f:
                favorites = json.load(f)
            return favorites.get(user_id, [])
        except:
            return []
    
    def remove_from_favorites(self, image_path, user_id="default"):
        try:
            with open(self.favorites_file, 'r') as f:
                favorites = json.load(f)
            
            if user_id in favorites:
                favorites[user_id] = [
                    fav for fav in favorites[user_id] 
                    if fav['path'] != image_path
                ]
                
                with open(self.favorites_file, 'w') as f:
                    json.dump(favorites, f, indent=2)
                return True
        except:
            pass
        return False
    
    def create_collection(self, name, description="", user_id="default"):
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
        except:
            collections = {}
        
        if user_id not in collections:
            collections[user_id] = {}
        
        collection_id = f"{name.lower().replace(' ', '_')}_{len(collections[user_id])}"
        collections[user_id][collection_id] = {
            "name": name,
            "description": description,
            "created_date": datetime.now().isoformat(),
            "images": []
        }
        
        with open(self.collections_file, 'w') as f:
            json.dump(collections, f, indent=2)
        
        return collection_id
    
    def add_to_collection(self, collection_id, image_path, user_id="default"):
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            
            if (user_id in collections and 
                collection_id in collections[user_id] and
                image_path not in collections[user_id][collection_id]["images"]):
                
                collections[user_id][collection_id]["images"].append(image_path)
                
                with open(self.collections_file, 'w') as f:
                    json.dump(collections, f, indent=2)
                return True
        except:
            pass
        return False
    
    def get_collections(self, user_id="default"):
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            return collections.get(user_id, {})
        except:
            return {}
    
    def get_collection_images(self, collection_id, user_id="default"):
        collections = self.get_collections(user_id)
        if collection_id in collections:
            return collections[collection_id]["images"]
        return []
    
    def search_in_collection(self, query_vector, collection_id, user_id="default", top_k=9):
        collection_images = self.get_collection_images(collection_id, user_id)
        if not collection_images:
            return []
        
        collection_indices = []
        for img_path in collection_images:
            for i, path in enumerate(self.image_paths):
                if str(path) == img_path:
                    collection_indices.append(i)
                    break
        
        if not collection_indices:
            return []
        
        collection_embeddings = self.image_embeddings[collection_indices]
        
        d = collection_embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(d)
        temp_index.add(collection_embeddings)
        
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = np.array([query_vector], dtype="float32")
        
        D, I = temp_index.search(query_vector, min(top_k, len(collection_indices)))
        
        results = []
        for j, i in enumerate(I[0]):
            original_index = collection_indices[i]
            results.append({
                "path": str(self.image_paths[original_index]),
                "score": float(D[0][j]),
                "index": int(original_index)
            })
        
        return results
    
    def find_similar_to_image(self, target_image_path, top_k=9, exclude_self=True):
        target_index = None
        for i, path in enumerate(self.image_paths):
            if str(path) == target_image_path:
                target_index = i
                break
        
        if target_index is None:
            return []
        
        query_vector = self.image_embeddings[target_index]
        
        exclude_indices = [target_index] if exclude_self else None
        return self.search_similar(query_vector, top_k, exclude_indices)
    
    def get_image_metadata(self, image_path):
        try:
            path_obj = Path(image_path)
            if path_obj.exists():
                stat = path_obj.stat()
                return {
                    "filename": path_obj.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "exists": True
                }
        except:
            pass
        
        return {
            "filename": Path(image_path).name,
            "size_mb": 0,
            "modified_date": None,
            "exists": False
        }

enhanced_search_engine = EnhancedSearchEngine()