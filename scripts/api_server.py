from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Allow CORS for your frontend origin
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Load model, index, and metadata once on startup
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index('../vector_store/faiss_index.idx')
with open('../vector_store/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

class SearchQuery(BaseModel):
    query: str
    k: int = 5  # Optional number of results

def search_products(query, k=5):
    query_vector = model.encode([query])[0].astype('float32')
    distances, indices = index.search(np.array([query_vector]), k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

@app.post("/search")
def search_endpoint(search: SearchQuery):
    results = search_products(search.query, search.k)
    return {"results": results}
