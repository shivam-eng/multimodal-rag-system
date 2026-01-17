import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "data/images"
image_embeddings = []
image_paths = []

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding.cpu().numpy()
    image_embeddings.append(embedding)
    image_paths.append(img_path)

image_embeddings = np.vstack(image_embeddings)

# Create FAISS index
dimension = image_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(image_embeddings)

faiss.write_index(index, "image_index.faiss")
np.save("image_paths.npy", image_paths)

print("✅ Image embeddings indexed successfully")
