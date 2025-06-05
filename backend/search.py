import numpy as np
import streamlit as st

import numpy as np
import faiss


image_paths = np.load("image_paths.npy", allow_pickle=True)
image_embeddings = np.load("image_embeddings.npy")

image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

d = image_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(image_embeddings)


def search_clip(query_vector, top_k=5):
    query_vector = query_vector / np.linalg.norm(query_vector)
    query_vector = np.array([query_vector], dtype="float32")  # cast to float32
    D, I = index.search(query_vector, top_k)
    results = [{"path": image_paths[i], "score": float(D[0][j])} for j, i in enumerate(I[0])]
    return results

def show_results(results, image_paths):
    st.subheader("Search Results")
    for i in results:
        st.image(image_paths[i], use_column_width=True)
