import sys
import os
import time
import uuid
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import torch

from backend.preprocessing import process_query
from backend.model import get_text_embedding, get_image_embedding, load_clip
from backend.search import search_clip
from backend.analytics import analytics
from backend.enhanced_search import enhanced_search_engine
from PIL import Image
import numpy as np
import base64
import json
        
def get_absolute_image_path(relative_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    possible_bases = [
        project_root,
        os.path.join(project_root, "downloaded_images"),
        os.path.join(project_root, "data"),
        os.path.join(project_root, "data", "downloaded_images"),
        os.path.join(project_root, "images"),
    ]
    
    path_variations = [
        str(relative_path),
        os.path.basename(str(relative_path)),
    ]
    
    for base in possible_bases:
        for path_var in path_variations:
            try:
                full_path = os.path.abspath(os.path.join(base, path_var))
                if os.path.exists(full_path):
                    return full_path
            except:
                continue
    
    return str(relative_path)

st.set_page_config(
    layout="wide", 
    page_title="CLIPSearch AI", 
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

try:
    clip_model, clip_processor = load_clip()
except:
    print("🔄 Trying alternative loading method...")
    clip_model, clip_processor = load_clip_simple()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 20px;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="white" opacity="0.1"/><circle cx="80" cy="30" r="1.5" fill="white" opacity="0.1"/><circle cx="40" cy="70" r="1" fill="white" opacity="0.1"/><circle cx="90" cy="80" r="2.5" fill="white" opacity="0.1"/></svg>');
        animation: float 20s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translateX(-100px); }
        100% { transform: translateX(100px); }
    }
    
    .hero-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 15px 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 25px;
        position: relative;
        z-index: 1;
    }
    
    .stat-item {
        text-align: center;
        color: white;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .search-container {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 15px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .stFileUploader {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 2px dashed #c7d2fe;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1v3fvcr {
        color: white;
    }
    
    .results-header {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3);
    }
    
    .query-display {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin: 15px 0;
        font-weight: 500;
        box-shadow: 0 6px 20px rgba(237, 137, 54, 0.3);
    }
    
    .stImage > div {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
        border: 3px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #667eea, #764ba2) border-box;
    }
    
    .stImage > div:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-stats {
            flex-direction: column;
            gap: 20px;
        }
        .search-container {
            padding: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

if "recent_input_type" not in st.session_state:
    st.session_state.recent_input_type = None
if "query_vector" not in st.session_state:
    st.session_state.query_vector = None
if "cleaned_query" not in st.session_state:
    st.session_state.cleaned_query = ""
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "current_collection" not in st.session_state:
    st.session_state.current_collection = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "query_info" not in st.session_state:
    st.session_state.query_info = ""

st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🔍 CLIPSearch AI</h1>
    <p class="hero-subtitle">Advanced Multimodal Semantic Search Engine</p>
    <div class="hero-stats">
        <div class="stat-item">
            <span class="stat-number">1M+</span>
            <span class="stat-label">Images Indexed</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">99.2%</span>
            <span class="stat-label">Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">0.1s</span>
            <span class="stat-label">Search Time</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: white;'>
        <h2>⚙️ Search Control</h2>
        <p style='opacity: 0.8;'>Fine-tune your search experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_selected = st.radio(
        "Navigation",
        ["🔍 Search", "📊 Analytics", "⭐ Favorites", "📁 Collections"],
        horizontal=True
    )
    
    if tab_selected == "🔍 Search":
        st.markdown("### 🎯 Search Parameters")
        search_mode = st.selectbox(
            "Search Mode",
            ["Semantic", "Visual", "Hybrid"],
            help="Choose how to interpret your query"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum confidence for results"
        )
        
        max_results = st.slider(
            "Max Results",
            min_value=3,
            max_value=20,
            value=9,
            help="Maximum number of results to display"
        )
        
        collections = enhanced_search_engine.get_collections()
        if collections:
            st.markdown("### 📁 Search in Collection")
            collection_options = ["All Images"] + [collections[coll_id]["name"] for coll_id in collections.keys()]
            collection_ids = ["all"] + list(collections.keys())
            
            selected_idx = st.selectbox(
                "Collection",
                range(len(collection_options)),
                format_func=lambda x: collection_options[x],
                help="Search within a specific collection"
            )
            
            st.session_state.current_collection = None if collection_ids[selected_idx] == "all" else collection_ids[selected_idx]
    
    elif tab_selected == "⭐ Favorites":
        st.markdown("### ⭐ Your Favorites")
        favorites = enhanced_search_engine.get_favorites()
        
        if favorites:
            st.write(f"You have {len(favorites)} favorite images")
            
            for fav in favorites[-5:]:
                if st.button(f"🖼️ {Path(fav['path']).name}", key=f"fav_{fav['path']}"):
                    similar_results = enhanced_search_engine.find_similar_to_image(fav['path'])
                    st.session_state.search_results = similar_results
                    st.session_state.query_info = f"Similar to {Path(fav['path']).name}"
                    st.session_state.recent_input_type = "favorite"
                    st.rerun()
        else:
            st.info("No favorites yet. Add some by clicking the ⭐ button on search results!")
    
    elif tab_selected == "📁 Collections":
        st.markdown("### 📁 Manage Collections")
        
        with st.expander("➕ Create New Collection"):
            new_collection_name = st.text_input("Collection Name")
            new_collection_desc = st.text_area("Description (optional)")
            
            if st.button("Create Collection") and new_collection_name:
                collection_id = enhanced_search_engine.create_collection(
                    new_collection_name, new_collection_desc
                )
                st.success(f"Created collection: {new_collection_name}")
                st.rerun()
        
        collections = enhanced_search_engine.get_collections()
        if collections:
            for coll_id, coll_data in collections.items():
                with st.expander(f"📁 {coll_data['name']} ({len(coll_data['images'])} images)"):
                    st.write(f"**Description:** {coll_data.get('description', 'No description')}")
                    st.write(f"**Created:** {coll_data['created_date'][:10]}")
                    
                    if st.button(f"Browse {coll_data['name']}", key=f"browse_{coll_id}"):
                        st.session_state.view_collection = coll_id
                        st.rerun()
    
    st.markdown("---")
    if st.session_state.search_history:
        st.markdown("### 📜 Recent Searches")
        for i, search in enumerate(st.session_state.search_history[-5:]):
            if st.button(f"🔄 {search[:30]}{'...' if len(search) > 30 else ''}", key=f"history_{i}"):
                st.session_state.text_input = search
                st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
        st.session_state.query_vector = None
        st.session_state.cleaned_query = ""
        st.session_state.recent_input_type = None
        st.session_state.text_input = ""
        st.session_state.search_history = []
        st.session_state.search_results = []
        st.session_state.upload_counter += 1
        st.rerun()

def display_enhanced_gallery(results, query_info=None):
    if not results:
        st.markdown("""
        <div style='text-align: center; padding: 60px; background: #f8f9fa; border-radius: 15px; margin: 20px 0;'>
            <h3 style='color: #666; margin-bottom: 20px;'>🔍 No Results Found</h3>
            <p style='color: #888;'>Try adjusting your search terms or upload a different image</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if query_info:
        st.markdown(f"""
        <div class="query-display">
            <strong>🎯 Query:</strong> "{query_info}" | <strong>📊 Found:</strong> {len(results)} results
        </div>
        """, unsafe_allow_html=True)
    
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        row_results = results[i:i + cols_per_row]
        
        for col, result in zip(cols, row_results):
            img_path = result["path"]
            score = result["score"]
            img_index = result.get("index", 0)
            
            match_percentage = max(0, min(100, score * 100))
            
            resolved_path = get_absolute_image_path(img_path)
            
            if os.path.exists(resolved_path):
                with col:
                    st.image(resolved_path, use_container_width=True)
                    
                    metadata = enhanced_search_engine.get_image_metadata(resolved_path)
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%); 
                                border-radius: 12px; margin: 10px 0; border: 1px solid #e2e8f0;'>
                        <div style='font-weight: 600; color: #2d3748; margin-bottom: 8px;'>
                            📷 {metadata['filename'][:20]}{'...' if len(metadata['filename']) > 20 else ''}
                        </div>
                        <div style='display: flex; justify-content: space-between; font-size: 12px; color: #718096; margin-bottom: 8px;'>
                            <span>💾 {metadata['size_mb']} MB</span>
                            <span style='background: linear-gradient(135deg, #48bb78, #38a169); color: white; 
                                        padding: 2px 8px; border-radius: 10px; font-weight: 600;'>
                                ✨ {match_percentage:.1f}%
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    button_cols = st.columns(4)
                    
                    with button_cols[0]:
                        is_favorited = resolved_path in [fav['path'] for fav in enhanced_search_engine.get_favorites()]
                        fav_icon = "💛" if is_favorited else "⭐"
                        if st.button(fav_icon, key=f"fav_{img_index}_{i}", help="Add to favorites"):
                            if is_favorited:
                                enhanced_search_engine.remove_from_favorites(resolved_path)
                                st.success("Removed from favorites!")
                            else:
                                query_text = query_info if query_info else "search"
                                enhanced_search_engine.save_to_favorites(resolved_path, query_text)
                                st.success("Added to favorites!")
                            time.sleep(1)
                            st.rerun()
                    
                    with button_cols[1]:
                        if st.button("🔄", key=f"similar_{img_index}_{i}", help="Find similar"):
                            similar_results = enhanced_search_engine.find_similar_to_image(resolved_path)
                            st.session_state.search_results = similar_results
                            st.session_state.recent_input_type = "similar"
                            st.session_state.query_info = f"Similar to {metadata['filename']}"
                            st.rerun()
                    
                    with button_cols[2]:
                        collections = enhanced_search_engine.get_collections()
                        if collections and st.button("📁", key=f"collect_{img_index}_{i}", help="Add to collection"):
                            st.session_state[f"show_collections_{img_index}_{i}"] = True
                    
                    with button_cols[3]:
                        if st.button("🔍", key=f"expand_{img_index}_{i}", help="View details"):
                            with st.expander("🖼️ Full Size Preview", expanded=True):
                                st.image(resolved_path, caption=f"📁 {metadata['filename']}")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("File Size", f"{metadata['size_mb']} MB")
                                with col_b:
                                    st.metric("Match Score", f"{match_percentage:.1f}%")
                                with col_c:
                                    st.metric("Raw Score", f"{score:.4f}")
                    
                    if st.session_state.get(f"show_collections_{img_index}_{i}", False):
                        collections = enhanced_search_engine.get_collections()
                        if collections:
                            collection_names = [f"{coll_data['name']}" for coll_id, coll_data in collections.items()]
                            collection_ids = list(collections.keys())
                            
                            selected_idx = st.selectbox(
                                "Select Collection",
                                range(len(collection_names)),
                                format_func=lambda x: collection_names[x],
                                key=f"coll_select_{img_index}_{i}"
                            )
                            
                            col_add, col_cancel = st.columns(2)
                            with col_add:
                                if st.button("Add", key=f"add_to_coll_{img_index}_{i}"):
                                    selected_coll_id = collection_ids[selected_idx]
                                    success = enhanced_search_engine.add_to_collection(selected_coll_id, resolved_path)
                                    if success:
                                        st.success(f"Added to {collection_names[selected_idx]}!")
                                    else:
                                        st.warning("Already in collection or error occurred")
                                    st.session_state[f"show_collections_{img_index}_{i}"] = False
                                    time.sleep(1)
                                    st.rerun()
                            
                            with col_cancel:
                                if st.button("Cancel", key=f"cancel_coll_{img_index}_{i}"):
                                    st.session_state[f"show_collections_{img_index}_{i}"] = False
                                    st.rerun()
            else:
                with col:
                    st.error("❌ Image not found")
                    st.write(f"**Original:** `{img_path}`")
                    st.write(f"**Tried:** `{resolved_path}`")
                    st.write(f"**Looking for:** `{os.path.basename(img_path)}`")

def process_search_with_analytics(query_type, query_data, max_results):
    start_time = time.time()
    
    try:
        if query_type == "text":
            text_vector, cleaned_query = process_query(query_data)
            
            if st.session_state.current_collection:
                results = enhanced_search_engine.search_in_collection(
                    text_vector, st.session_state.current_collection, top_k=max_results
                )
            else:
                results = enhanced_search_engine.search_similar(text_vector, top_k=max_results)
            
            fixed_results = []
            for result in results:
                resolved_path = get_absolute_image_path(result["path"])
                fixed_results.append({
                    "path": resolved_path,
                    "score": result["score"],
                    "index": result.get("index", 0),
                    "original_path": result["path"]
                })
            
            response_time = (time.time() - start_time) * 1000
            analytics.log_search("text", query_data, cleaned_query, len(fixed_results), response_time)
            
            return fixed_results, cleaned_query
            
        elif query_type == "image":
            image_vector = get_image_embedding(query_data, clip_model, clip_processor)
            
            if st.session_state.current_collection:
                results = enhanced_search_engine.search_in_collection(
                    image_vector, st.session_state.current_collection, top_k=max_results
                )
            else:
                results = enhanced_search_engine.search_similar(image_vector, top_k=max_results)
            
            fixed_results = []
            for result in results:
                resolved_path = get_absolute_image_path(result["path"])
                fixed_results.append({
                    "path": resolved_path,
                    "score": result["score"],
                    "index": result.get("index", 0),
                    "original_path": result["path"]
                })
            
            response_time = (time.time() - start_time) * 1000
            analytics.log_search("image", "uploaded_image", "uploaded_image", len(fixed_results), response_time)
            
            return fixed_results, "uploaded image"
            
    except Exception as e:
        st.exception(e)
        return [], ""

if tab_selected == "🔍 Search":
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        
        text_input_value = ""
        if st.session_state.get('reset_text', False):
            st.session_state.reset_text = False
            text_input_value = ""
        else:
            text_input_value = st.session_state.text_input
        
        col1, col2 = st.columns([4, 1])
        with col1:
            text_query = st.text_input(
                "🔍 Enter your search query",
                value=text_input_value,
                key="text_input_widget",
                placeholder="e.g., 'a red sports car' or 'sunset over mountains'",
                help="Describe what you're looking for in natural language"
            )
            st.session_state.text_input = text_query
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🚀 Search", type="primary", use_container_width=True)
        
        st.markdown("### 📸 Or Upload an Image")
        
        upload_key = f"image_input_{st.session_state.upload_counter}"
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "webp"],
            key=upload_key,
            help="Upload an image to find similar ones"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if text_query and (st.session_state.recent_input_type != "image" or uploaded_image is None):
        if text_query not in st.session_state.search_history:
            st.session_state.search_history.append(text_query)
        
        with st.spinner("🧠 Processing your query..."):
            results, processed_query = process_search_with_analytics("text", text_query, max_results)
            st.session_state.search_results = results
            st.session_state.query_info = processed_query
            st.session_state.recent_input_type = "text"
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
        with st.spinner("🔍 Analyzing your image..."):
            results, processed_query = process_search_with_analytics("image", image, max_results)
            st.session_state.search_results = results
            st.session_state.query_info = processed_query
            st.session_state.recent_input_type = "image"
    
    if st.session_state.search_results:
        query_info = st.session_state.query_info
        display_enhanced_gallery(st.session_state.search_results, query_info)
        
        st.success(f"✅ Found {len(st.session_state.search_results)} results")
        
        st.markdown("### 📝 Rate this search")
        rating = st.radio(
            "How satisfied are you with these results?",
            ["😞 Poor", "😐 Okay", "😊 Good", "🤩 Excellent"],
            horizontal=True,
            key="search_rating"
        )
        
        if st.button("Submit Rating"):
            rating_value = {"😞 Poor": 1, "😐 Okay": 2, "😊 Good": 3, "🤩 Excellent": 4}[rating]
            st.success("Thank you for your feedback!")
            st.rerun()

elif tab_selected == "📊 Analytics":
    analytics.render_analytics_dashboard()

elif tab_selected == "⭐ Favorites":
    st.markdown("## ⭐ Your Favorite Images")
    favorites = enhanced_search_engine.get_favorites()
    
    if favorites:
        fav_results = [
            {
                "path": fav["path"],
                "score": 1.0,
                "index": i
            }
            for i, fav in enumerate(favorites)
        ]
        
        display_enhanced_gallery(fav_results, f"Your {len(favorites)} favorite images")
        
        st.markdown("### 🛠️ Bulk Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Favorites List"):
                fav_list = [fav["path"] for fav in favorites]
                fav_data = "\n".join(fav_list)
                st.download_button(
                    label="Download as Text File",
                    data=fav_data,
                    file_name="favorites_list.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("🗑️ Clear All Favorites"):
                if st.button("Confirm Clear All", type="secondary"):
                    with open(enhanced_search_engine.favorites_file, 'w') as f:
                        json.dump({}, f)
                    st.success("All favorites cleared!")
                    st.rerun()
    else:
        st.info("No favorites yet. Add some by clicking the ⭐ button on search results!")
        
        st.markdown("### 🚀 Try these sample searches:")
        sample_queries = ["beautiful landscape", "modern architecture", "cute animals", "vintage cars"]
        
        cols = st.columns(len(sample_queries))
        for i, query in enumerate(sample_queries):
            with cols[i]:
                if st.button(f"🔍 {query}", key=f"sample_{i}"):
                    st.session_state.text_input = query
                    st.session_state.tab_selected = "🔍 Search"
                    st.rerun()

elif tab_selected == "📁 Collections":
    st.markdown("## 📁 Your Collections")
    
    if hasattr(st.session_state, 'view_collection'):
        collection_id = st.session_state.view_collection
        collections = enhanced_search_engine.get_collections()
        
        if collection_id in collections:
            collection_data = collections[collection_id]
            st.markdown(f"### 📁 {collection_data['name']}")
            st.write(f"**Description:** {collection_data.get('description', 'No description')}")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("← Back to Collections"):
                    del st.session_state.view_collection
                    st.rerun()
            
            with col2:
                if st.text_input("🔍 Search within this collection", key="collection_search"):
                    collection_query = st.session_state.collection_search
                    if collection_query:
                        with st.spinner("Searching within collection..."):
                            text_vector, _ = process_query(collection_query)
                            coll_results = enhanced_search_engine.search_in_collection(
                                text_vector, collection_id, top_k=9
                            )
                            if coll_results:
                                display_enhanced_gallery(coll_results, f"'{collection_query}' in {collection_data['name']}")
                            else:
                                st.info("No matches found in this collection.")
            
            collection_images = enhanced_search_engine.get_collection_images(collection_id)
            if collection_images:
                coll_results = []
                for i, img_path in enumerate(collection_images):
                    if Path(img_path).exists():
                        coll_results.append({
                            "path": img_path,
                            "score": 1.0,
                            "index": i
                        })
                
                if coll_results:
                    display_enhanced_gallery(coll_results, f"Collection: {collection_data['name']}")
                else:
                    st.warning("Some images in this collection no longer exist.")
            else:
                st.info("This collection is empty. Add images by using the 📁 button on search results.")
    
    else:
        collections = enhanced_search_engine.get_collections()
        
        if collections:
            total_images = sum(len(coll_data['images']) for coll_data in collections.values())
            st.markdown(f"**Total Collections:** {len(collections)} | **Total Images:** {total_images}")
            
            col_per_row = 2
            collection_items = list(collections.items())
            
            for i in range(0, len(collection_items), col_per_row):
                cols = st.columns(col_per_row)
                row_collections = collection_items[i:i + col_per_row]
                
                for col, (coll_id, coll_data) in zip(cols, row_collections):
                    with col:
                        with st.container():
                            st.markdown(f"### 📁 {coll_data['name']}")
                            st.write(f"**Images:** {len(coll_data['images'])}")
                            st.write(f"**Created:** {coll_data['created_date'][:10]}")
                            
                            if coll_data.get('description'):
                                st.write(f"**Description:** {coll_data['description'][:100]}...")
                            
                            if coll_data['images']:
                                preview_images = [img for img in coll_data['images'][:3] if Path(img).exists()]
                                if preview_images:
                                    preview_cols = st.columns(len(preview_images))
                                    for j, img_path in enumerate(preview_images):
                                        with preview_cols[j]:
                                            st.image(img_path, use_container_width=True)
                            
                            button_cols = st.columns(2)
                            with button_cols[0]:
                                if st.button(f"Browse", key=f"browse_{coll_id}"):
                                    st.session_state.view_collection = coll_id
                                    st.rerun()
                            
                            with button_cols[1]:
                                if st.button(f"Delete", key=f"delete_{coll_id}", type="secondary"):
                                    if st.button(f"Confirm Delete {coll_data['name']}", key=f"confirm_delete_{coll_id}"):
                                        try:
                                            with open(enhanced_search_engine.collections_file, 'r') as f:
                                                all_collections = json.load(f)
                                            
                                            if "default" in all_collections and coll_id in all_collections["default"]:
                                                del all_collections["default"][coll_id]
                                                
                                                with open(enhanced_search_engine.collections_file, 'w') as f:
                                                    json.dump(all_collections, f, indent=2)
                                                
                                                st.success(f"Deleted collection: {coll_data['name']}")
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting collection: {e}")
        else:
            st.info("No collections yet. Create your first collection in the sidebar!")
            
            st.markdown("""
            ### 🚀 Getting Started with Collections
            
            1. **Create a Collection** - Use the sidebar to create a new collection
            2. **Search for Images** - Go to the Search tab and find interesting images
            3. **Add to Collection** - Click the 📁 button on any search result
            4. **Organize & Browse** - Come back here to browse and search within collections
            
            Collections help you organize your favorite images by theme, project, or any criteria you choose!
            """)

if tab_selected == "🔍 Search" and not st.session_state.search_results:
    st.markdown("""
    <div style='text-align: center; padding: 60px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                border-radius: 20px; margin: 40px 0; border: 2px solid #e1f5fe;'>
        <h2 style='color: #0277bd; margin-bottom: 20px;'>🎯 Ready to Search</h2>
        <p style='color: #0288d1; font-size: 18px; margin-bottom: 15px;'>
            Enter a text query or upload an image to discover similar content
        </p>
        <p style='color: #0288d1; opacity: 0.8;'>
            Our AI-powered search engine will find the most relevant matches for you
        </p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.search_results and tab_selected == "🔍 Search":
    st.markdown("---")
    st.markdown("### 🛠️ Bulk Actions")
    
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("📊 View Analytics", use_container_width=True):
            st.balloons()
            st.info("📈 Check out the Analytics tab for detailed insights!")
    
    with action_cols[1]:
        if st.button("📥 Export Results", use_container_width=True):
            results_data = []
            for result in st.session_state.search_results:
                results_data.append(f"{result['path']} (Score: {result['score']:.4f})")
            
            results_text = "\n".join(results_data)
            st.download_button(
                label="Download Results as Text",
                data=results_text,
                file_name=f"search_results_{st.session_state.query_info.replace(' ', '_')}.txt",
                mime="text/plain"
            )
    
    with action_cols[2]:
        if st.button("⭐ Add All to Favorites", use_container_width=True):
            added_count = 0
            for result in st.session_state.search_results:
                if enhanced_search_engine.save_to_favorites(result['path'], st.session_state.query_info):
                    added_count += 1
            
            if added_count > 0:
                st.success(f"Added {added_count} images to favorites!")
            else:
                st.info("All images were already in favorites!")
    
    with action_cols[3]:
        if st.button("🔄 Refine Search", use_container_width=True):
            if st.session_state.recent_input_type == "text":
                current_query = st.session_state.get("text_input", "")
                if current_query and current_query not in st.session_state.search_history:
                    st.session_state.search_history.append(current_query)

            st.session_state.search_results = []
            st.session_state.query_info = ""
            st.session_state.recent_input_type = None
            st.session_state.reset_text = True
            st.session_state.upload_counter += 1
            
            st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white; margin-top: 40px;'>
    <h4 style='margin: 0 0 10px 0;'>🚀 Powered by State-of-the-Art AI</h4>
    <p style='margin: 0; opacity: 0.9;'>
        Built with CLIP • Streamlit • Modern Web Technologies
    </p>
    <p style='margin: 10px 0 0 0; font-size: 14px; opacity: 0.8;'>
        © 2025 CLIPSearch AI - Revolutionizing Visual Search
    </p>
</div>
""", unsafe_allow_html=True)