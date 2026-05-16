# 🔍 Multimodal RAG System — Text → Image Semantic Search

![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-blueviolet)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b?logo=streamlit)

---

## 📌 Overview

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system that performs semantic search across images using natural language queries — and generates grounded answers from the retrieved visual context.

Built as a **production-style prototype** to demonstrate modern AI system design including multimodal embeddings, vector similarity search, and LLM-based answer synthesis with graceful fallback.

> 🚀 **Live Demo →** [multimodal-rag-systemgit-grizzly.streamlit.app](https://multimodal-rag-systemgit-grizzly.streamlit.app)

---

## 🎯 What It Does

Type any natural language query → the system retrieves the most semantically relevant images from the indexed dataset → generates a contextual answer using GPT-3.5 (or a structured fallback with no API key needed).

```
"show me an invoice"     →  retrieves bill.jpg, invoice.png   →  "Found financial documents..."
"a cat resting"          →  retrieves cat.jpg, sofa.jpg        →  "The images show a cat..."
"system architecture"    →  retrieves flowchart.png            →  "Retrieved a diagram showing..."
```

---

## 🧠 Key Concepts Demonstrated

- **Multimodal embeddings** — CLIP (ViT-B/32) aligns text and images into a shared vector space
- **Zero-shot retrieval** — no manual labels or fine-tuning required
- **Vector similarity search** — FAISS for fast approximate nearest-neighbour lookup
- **RAG pipeline** — retrieval context fed into LLM for grounded answer generation
- **Graceful degradation** — structured fallback when OpenAI API is unavailable
- **Clean modular Python** — each concern separated into its own module

---

## 🏗️ Architecture

```
User Query (Natural Language)
        ↓
CLIP Text Encoder (ViT-B/32)
        ↓
L2-Normalised Query Vector
        ↓
FAISS Index Search  ←── Pre-indexed Image Embeddings (CLIP)
        ↓
Top-K Relevant Images + Similarity Scores
        ↓
Context Augmentation
        ↓
GPT-3.5 Generation  /  Local Fallback
        ↓
Final Answer + Retrieved Images
```

---

## 📂 Project Structure

```
multimodal-rag/
│
├── streamlit_app.py     # Live UI — Streamlit frontend
├── app.py               # CLI entry point
├── embed.py             # CLIP image embedding + FAISS index creation
├── search.py            # Multimodal similarity search
├── rag.py               # RAG generation logic (LLM + fallback)
│
├── image_index.faiss    # Pre-built FAISS vector index
├── image_paths.npy      # Image path mapping for the index
│
├── data/
│   └── images/          # Sample images (animals, objects, documents, charts)
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/shivam-eng/multimodal-rag-system.git
cd multimodal-rag-system
```

**2. Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

**3. Install dependencies**
```bash
pip install torch torchvision faiss-cpu pillow numpy python-dotenv openai streamlit
pip install git+https://github.com/openai/CLIP.git
```

**4. Add your OpenAI API key (optional)**
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```
> ⚡ The system works without an API key — it uses the local fallback mode.

**5. Generate the FAISS index**
```bash
python embed.py
```

**6. Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

Or run the CLI version:
```bash
python app.py
```

---

## 🖼️ Dataset

| Category | Example Files |
|---|---|
| Animals | `cat.jpg` |
| Vehicles | `car.jpg` |
| Documents | `bill.jpg`, `invoice.png` |
| Diagrams | `flowchart.png` |
| Electronics | `phone.jpg` |

10–20 heterogeneous images across multiple categories. No manual labelling required — CLIP handles zero-shot alignment.

---

## 🔄 Fallback Mode

If no OpenAI API key is configured, the system:
- Completes the full CLIP + FAISS retrieval pipeline
- Returns a structured explanation based on retrieved image names and similarity scores
- Labels the response clearly as fallback mode

This ensures the **retrieval pipeline is always fully demonstrable** regardless of API availability.

---

## 🧪 Sample Output

**Query:** `order invoice`

```
Results found: 3  |  Search time: 126 ms  |  Top score: 82.821

Retrieved Images:
  #1  data/images/bill.jpg      — similarity: 82.821
  #2  data/images/car.jpg       — similarity: 92.870
  #3  data/images/phone.jpg     — similarity: 91.145

Generated Answer (fallback):
  The semantic search retrieved 3 relevant image(s) for 'order invoice'.
  The closest match is bill (similarity score: 82.821).
  This result was generated using CLIP-based vector similarity.
```

---

## 🔍 Limitations

- Small dataset (10–20 images) may cause partial semantic mismatches
- CLIP embeddings are general-purpose; domain-specific fine-tuning would improve accuracy
- Image paths must be relative for cross-platform compatibility

---

## 🚀 Future Enhancements

- [ ] Caption-based hybrid search (CLIP + text captions)
- [ ] OCR integration for invoice/document retrieval
- [ ] Local LLM support via Ollama
- [ ] Audio and video modality support
- [ ] Larger image dataset with diverse categories

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Multimodal Embeddings | OpenAI CLIP (ViT-B/32) |
| Vector Search | FAISS (faiss-cpu) |
| LLM Generation | OpenAI GPT-3.5-turbo |
| Frontend / UI | Streamlit |
| Image Processing | Pillow |
| ML Framework | PyTorch |

---

## 👤 Author

**Shivam** — Built during learning phase to explore multimodal AI system design.

[![GitHub](https://img.shields.io/badge/GitHub-shivam--eng-181717?logo=github)](https://github.com/shivam-eng)
