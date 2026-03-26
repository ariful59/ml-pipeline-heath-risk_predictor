from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

sentences = ["This is a test sentence", "Another one"]
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")
print("Test successful.")
