import faiss
import json
import numpy as np
from tqdm import tqdm

embedding_file = '../products_embeddings_local.jsonl'
index_file = '../vector_store/faiss_index.idx'

def load_embeddings(file_path):
    embeddings = []
    metadata = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            record = json.loads(line)
            embeddings.append(record['embedding'])
            metadata.append({
                'asin': record['asin'],
                'title': record['title'],
                'description': record['description'],
                'category': record['category']
            })
    return np.array(embeddings).astype('float32'), metadata

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) distance index
    index.add(embeddings)
    return index

def main():
    embeddings, metadata = load_embeddings(embedding_file)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    index = build_faiss_index(embeddings)
    print("FAISS index built.")

    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")

    # Save metadata for later retrieval
    with open('../vector_store/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved.")

if __name__ == "__main__":
    main()
