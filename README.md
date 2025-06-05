# 🎯 CLIPSearch: A Multimodal Semantic Search Engine for Images and Text

CLIPSearch AI is a Streamlit-based multimodal retrieval system that encodes images and text into 512-dim CLIP embeddings, indexes them with FAISS, and delivers sub-second text-to-image or image-to-image search. Queries are normalized by a fine-tuned T5-small spell corrector; initial results are fetched via cosine similarity and reranked using GPT-2 logits for enhanced semantic relevance. Includes a Plotly analytics dashboard, offline operation once embeddings are built, and support for user-defined favorites and collections.

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

## 📁 Dataset & Images

Due to GitHub size constraints, the full `downloaded_images/` folder is **not included** in the repository.

To extract and download images:

- Use the image URLs provided in `data/photos.csv`
- Refer the Jupyter notebook to extract the images
- This will save images into the `downloaded_images/` directory

## 📦 Download Pretrained Spell Correction Model

The `spellfix_t5_small/model.safetensors` file is too large for GitHub and is hosted on Google Drive.

### 🔽 Download Instructions

1. [Click here to download model.safetensors](https://drive.google.com/file/d/1408dJqMTyd2fPiGVB5qBnklFjwwsGzKg/view?usp=sharing)
2. Create the following folder structure in your project:
```
spellfix_t5_small/s
└── model.safetensors
```
3. Place the downloaded `model.safetensors` file inside the `spellfix_t5_small/` directory.

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
[![Watch the demo](screenshots/demo-thumbnail.png)](https://drive.google.com/file/d/12h5_ccwOK68EXA9ppikOgrcon_PPXtFo/view?usp=share_link)

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
