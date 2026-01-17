import clip
import torch
import faiss
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

index = faiss.read_index("image_index.faiss")
image_paths = np.load("image_paths.npy", allow_pickle=True)

def search_images(text_query, top_k=3):
    text = clip.tokenize([text_query]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text)

    text_embedding = text_embedding.cpu().numpy()
    distances, indices = index.search(text_embedding, top_k)

    results = [image_paths[i] for i in indices[0]]
    return results
