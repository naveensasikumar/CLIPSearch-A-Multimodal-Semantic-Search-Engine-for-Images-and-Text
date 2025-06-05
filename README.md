# 🎯 CLIPSearch: A Multimodal Semantic Search Engine for Images and Text

This project implements a multimodal AI-powered semantic search system using OpenAI’s CLIP and Sentence-BERT models. It enables users to search for images using natural language queries, image uploads, or a combination of both. The backend is powered by FastAPI, and the frontend offers an elegant Streamlit-based interface.

---

## 📌 Project Overview

### Objective
To build a fast, flexible, and visually appealing semantic search engine that retrieves relevant images from a local dataset based on natural language or multimodal queries.

### Key Features
- **CLIP-Based Text & Image Embeddings** for semantic similarity  
- **Multimodal Search** using both natural language and image inputs  
- **FAISS Vector Indexing** for high-speed nearest neighbor search  
- **Interactive Streamlit Frontend** with gallery view and filtering  
- **Spell Correction Module** (optional T5-based correction)  
- **Tag-Based Semantic Filtering** via sidebar selection  

---

## 🛠️ Technologies Used

| Component         | Tools / Frameworks              |
|-------------------|---------------------------------|
| Programming       | Python                          |
| Embedding Models  | OpenAI CLIP, Sentence-BERT      |
| Semantic Indexing | FAISS                           |
| Backend           | FastAPI                         |
| Frontend          | Streamlit                       |
| Image Handling    | PIL, NumPy                      |
| Spell Correction  | T5ForConditionalGeneration      |
| Deployment Ready  | Local scripts (Docker optional) |

---

## 📊 System Architecture

1. **Data Preparation**  
   - Store images locally and assign associated textual metadata  
   - Encode both image and text embeddings using CLIP/Sentence-BERT  
   - Save vectors using `NumPy` for FAISS-based search  

2. **Indexing & Retrieval**  
   - Build FAISS index on image embeddings  
   - Optionally filter results using semantic tags or categories  
   - Retrieve and rank results using cosine similarity  

3. **Spell Correction (Optional)**  
   - Noisy query correction using a fine-tuned T5 transformer  
   - Useful for improving results in casual or error-prone queries  

4. **FastAPI Backend**  
   - Exposes `/search`, `/upload`, and `/correct-query` endpoints  
   - Handles query preprocessing, retrieval, and result formatting  

5. **Streamlit UI**  
   - Upload image or enter a text/natural query  
   - View ranked results with semantic tags and responsive cards  

---

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CLIPSearch.git
cd CLIPSearch
```

### 2. Frontend Setup (Streamlit UI)

```bash
cd ..
streamlit run app.py
```

---

## 🔍 Example Use Case

- A user enters the query: `peaceful beach in winter`  
- The system encodes the query using Sentence-BERT / CLIP  
- FAISS retrieves the top 10 similar images based on vector similarity  
- Images are displayed in an elegant scrollable gallery  
- Optionally, the user can filter results by tags like `calm`, `nature`, `snow`  

---

## 🧠 Advanced Features

- **Multimodal Input:** Combine text and image for refined results  
- **Semantic Filtering:** Choose tags from a sidebar to narrow down results  
- **Query Correction:** Automatically correct common typos (e.g., “beech” → “beach”)  
- **Vector File Storage:** Supports `.npy` vector loading for scalability  

---

## 📽️ Demo

![CLIPSearch Demo](screenshots/demo1.png)  
_More demo images available in the `/screenshots` folder._

---

## 🚀 Future Improvements

- Support for multilingual queries  
- Integration with cloud storage (e.g., S3 or GDrive)  
- Re-ranking using cross-modal transformers  
- Add user history and bookmarking features  
- Deploy as a HuggingFace Space or Streamlit Sharing app  

---

## 📄 License

This project is licensed under the MIT License.  
© 2025 Naveen Sasikumar
