"""
Multimodal RAG System — Streamlit UI
Drop this file into your repo root. Zero changes to embed.py / search.py / rag.py.
Run: streamlit run streamlit_app.py
"""

import os
import sys
import time
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG | Text → Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Dark hero banner */
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #e2e8f0;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p { color: #94a3b8; font-size: 0.95rem; margin: 0; line-height: 1.6; }
.hero .badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 0.5rem 0.25rem 0 0;
    border: 1px solid #2563eb44;
}

/* Result cards */
.result-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.2s;
}
.result-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.result-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6366f1;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid #0ea5e9;
    border-radius: 0 10px 10px 0;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}
.answer-box h4 {
    font-family: 'Space Mono', monospace;
    color: #0369a1;
    font-size: 0.8rem;
    letter-spacing: 1px;
    margin: 0 0 0.5rem 0;
}
.answer-box p { color: #1e293b; line-height: 1.7; margin: 0; }

/* Fallback tag */
.fallback-tag {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    border: 1px solid #fcd34d;
}

/* Metric pills */
.metric-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #475569;
    margin: 0.2rem;
    font-family: 'Space Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #0f0f1a; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* Search button */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    width: 100%;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Lazy imports with error handling ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    """Load CLIP model + FAISS index once, cache across reruns."""
    try:
        import clip, torch, faiss, numpy as np

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        index = faiss.read_index("image_index.faiss")
        image_paths = np.load("image_paths.npy", allow_pickle=True).tolist()

        return {
            "model": model,
            "preprocess": preprocess,
            "device": device,
            "index": index,
            "image_paths": image_paths,
            "error": None,
        }
    except FileNotFoundError:
        return {"error": "index_missing"}
    except ImportError as e:
        return {"error": f"import_error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def embed_query(query: str, system: dict):
    import clip, torch
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(system["device"])
        vec = system["model"].encode_text(tokens).cpu().numpy().astype("float32")
    import faiss
    faiss.normalize_L2(vec)
    return vec


def search_images(query_vec, system: dict, top_k: int):
    distances, indices = system["index"].search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append({"path": system["image_paths"][idx], "score": float(dist)})
    return results


def generate_answer(query: str, results: list) -> tuple[str, bool]:
    """Try OpenAI, fall back gracefully."""
    context = "\n".join(
        [f"- {os.path.basename(r['path'])} (score: {r['score']:.3f})" for r in results]
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved images:\n{context}\n\n"
        "Based on these retrieved images, provide a concise, helpful answer "
        "explaining what was found and its relevance to the query. "
        "Be informative and specific."
    )

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful multimodal search assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip(), False  # (answer, is_fallback)
        except Exception:
            pass  # fall through to fallback

    # ── Local fallback ────────────────────────────────────────────────────────
    top = results[0] if results else None
    if not top:
        return "No relevant images were found for this query.", True

    names = [os.path.splitext(os.path.basename(r["path"]))[0] for r in results]
    answer = (
        f"The semantic search retrieved {len(results)} relevant image(s) for '{query}'. "
        f"The closest match is **{names[0]}** (similarity score: {top['score']:.3f}). "
        f"Other related results include: {', '.join(names[1:])}. "
        "This result was generated using CLIP-based vector similarity — no LLM API was used."
    )
    return answer, True


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 Multimodal RAG System</h1>
  <p>Semantic search across images using natural language — powered by CLIP embeddings &amp; FAISS vector search.</p>
  <span class="badge">CLIP ViT-B/32</span>
  <span class="badge">FAISS</span>
  <span class="badge">OpenAI GPT-3.5</span>
  <span class="badge">RAG Pipeline</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=3,
                      help="Number of images to retrieve")
    show_scores = st.toggle("Show similarity scores", value=True)
    st.markdown("---")
    st.markdown("## 🏗 Architecture")
    st.markdown("""
```
User Query
    ↓
CLIP Text Encoder
    ↓
FAISS Vector Search
    ↓
Top-K Images
    ↓
GPT-3.5 / Fallback
    ↓
Final Answer
```
    """)
    st.markdown("---")
    st.markdown("## 📌 Try these queries")
    example_queries = [
        "a cat resting",
        "financial document",
        "a vehicle on road",
        "system architecture diagram",
    ]
    for q in example_queries:
        if st.button(f"→ {q}", key=q):
            st.session_state["query_input"] = q

# ── Load system ───────────────────────────────────────────────────────────────
with st.spinner("Loading CLIP model & FAISS index…"):
    system = load_system()

if system.get("error") == "index_missing":
    st.error(
        "**FAISS index not found.** Run `python embed.py` first to generate "
        "`image_index.faiss` and `image_paths.npy`, then restart the app.",
        icon="⚠️",
    )
    st.stop()
elif system.get("error"):
    st.error(f"System failed to load: `{system['error']}`", icon="❌")
    st.stop()
else:
    st.success(
        f"✅ System ready — {len(system['image_paths'])} images indexed · "
        f"Device: `{system['device'].upper()}`",
    )

# ── Query input ───────────────────────────────────────────────────────────────
st.markdown("### 🔎 Enter your search query")
default_q = st.session_state.get("query_input", "")
query = st.text_input(
    label="query",
    value=default_q,
    placeholder="e.g.  cat sleeping on sofa  |  invoice document  |  architecture diagram",
    label_visibility="collapsed",
)
search_clicked = st.button("Search →")

# ── Search & display ──────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("Encoding query & searching…"):
        t0 = time.time()
        query_vec = embed_query(query.strip(), system)
        results = search_images(query_vec, system, top_k)
        elapsed = time.time() - t0

    if not results:
        st.warning("No results found. Try a different query.")
        st.stop()

    # Metrics row
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Results found", len(results))
    col_b.metric("Search time", f"{elapsed*1000:.0f} ms")
    col_c.metric("Top score", f"{results[0]['score']:.3f}")

    st.markdown("---")

    # ── Retrieved images ──────────────────────────────────────────────────────
    st.markdown("### 🖼 Retrieved Images")
    cols = st.columns(min(len(results), 4))
    for i, (col, res) in enumerate(zip(cols, results)):
        with col:
            try:
                img = Image.open(res["path"])
                col.image(img, use_container_width=True)
                label = f"#{i+1} · {os.path.basename(res['path'])}"
                if show_scores:
                    label += f"\nScore: {res['score']:.4f}"
                col.caption(label)
            except Exception:
                col.warning(f"Could not load image: {res['path']}")

    # ── Generated answer ──────────────────────────────────────────────────────
    st.markdown("### 💬 Generated Answer")
    with st.spinner("Generating answer…"):
        answer, is_fallback = generate_answer(query.strip(), results)

    if is_fallback:
        st.markdown('<span class="fallback-tag">⚡ Fallback mode — no LLM API key set</span>',
                    unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-box">
      <h4>RAG RESPONSE</h4>
      <p>{answer}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Raw results expander ──────────────────────────────────────────────────
    with st.expander("📊 Raw retrieval details"):
        for i, r in enumerate(results):
            st.markdown(
                f"**#{i+1}** `{r['path']}` — similarity: `{r['score']:.6f}`"
            )

elif search_clicked and not query.strip():
    st.warning("Please enter a query first.")
