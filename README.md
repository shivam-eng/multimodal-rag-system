Multimodal Search & RAG System (Text ↔ Image)
📌 Overview

This project demonstrates an end-to-end Multimodal Retrieval-Augmented Generation (RAG) system that enables semantic search across text and images and generates grounded answers using retrieved visual context.

The system uses contrastive multimodal embeddings (CLIP) to align text and images into a shared vector space, FAISS for fast similarity search, and an LLM-based (or local fallback) generation layer for response synthesis.

This project is intentionally scoped as a production-style prototype suitable for interviews, demos, and learning modern AI system design.

🧠 Key Concepts Demonstrated

Multimodal embeddings using CLIP (text ↔ image alignment)

Any-to-any semantic search (Text → Image)

Vector similarity search with FAISS

Retrieval-Augmented Generation (RAG)

Graceful fallback when LLM APIs are unavailable

Clean, modular Python project structure

🏗️ Architecture
User Query
   ↓
Text Embedding (CLIP)
   ↓
FAISS Vector Search
   ↓
Top-K Relevant Images
   ↓
Context Augmentation
   ↓
LLM / Local Fallback
   ↓
Final Answer
📂 Project Structure
multimodal-rag/
│
├── app.py              # Application entry point
├── embed.py            # Image embedding & FAISS index creation
├── search.py           # Multimodal similarity search
├── rag.py              # RAG generation logic (LLM + fallback)
├── image_index.faiss   # FAISS vector index (generated)
├── image_paths.npy     # Image path mapping (generated)
│
├── data/
│   └── images/         # Sample images (multiple categories)
│
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
🖼️ Dataset

10–20 heterogeneous images

Multiple categories (animals, objects, documents, charts)

No manual labels required (CLIP is zero-shot)

Example images:

cat.jpg

car.jpg

invoice.png

flowchart.png

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
2️⃣ Install Dependencies
pip install torch torchvision faiss-cpu pillow numpy python-dotenv
pip install git+https://github.com/openai/CLIP.git
🔐 Environment Variables

Create a .env file (optional):

OPENAI_API_KEY=your_api_key_here

⚠️ If no API key or quota is available, the system automatically switches to fallback mode.

🚀 How to Run
Step 1: Generate Image Embeddings
python embed.py

This creates:

image_index.faiss

image_paths.npy

Step 2: Run the Application
python app.py

Example query:

cat sleeping on sofa
🔄 Fallback Mode (No API / No Quota)

If the LLM API is unavailable, the system:

Skips external generation

Returns a structured explanation based on retrieved context

This ensures the retrieval pipeline remains fully demonstrable.

🧪 Sample Output
Retrieved Images:
- data/images/cat.jpg
- data/images/sofa.jpg


Generated Answer:
The retrieved images show a cat resting comfortably on a sofa...
🔍 Limitations & Improvements

Small dataset may cause partial mismatches

Can be improved using caption-based hybrid search

Can be extended to audio/video modalities

🚀 Future Enhancements

Image captioning + hybrid RAG

OCR for invoices and documents

Local LLM integration (Ollama)

Streamlit-based UI

Cloud deployment (GCP / AWS)
