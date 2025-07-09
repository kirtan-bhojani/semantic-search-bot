import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the same model used for creating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index('../vector_store/faiss_index.idx')

with open('../vector_store/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

def search_products(query, k=5):
    # Convert query to embedding
    query_vector = model.encode([query])[0].astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(np.array([query_vector]), k)

    # Retrieve matching product metadata
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    
    return results

if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your search query: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        results = search_products(user_query)
        print("\nTop Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['title']}\n{result['description']}\n")
