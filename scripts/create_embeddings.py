import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

input_file = '../products_cleaned.json'
output_file = '../products_embeddings_local.jsonl'  # changed filename to avoid confusion

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # load local embedding model once

    with open(input_file, 'r', encoding='utf-8') as f:
        products = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for product in tqdm(products, desc="Creating embeddings"):
            text = f"{product.get('title', '')}. {product.get('description', '')}"
            try:
                embedding = model.encode(text).tolist()  # local embedding as list
                record = {
                    'asin': product.get('asin', ''),
                    'embedding': embedding,
                    'title': product.get('title', ''),
                    'description': product.get('description', ''),
                    'category': product.get('category', '')
                }
                out_f.write(json.dumps(record) + '\n')
            except Exception as e:
                print(f"Error embedding {product.get('asin', 'unknown')}: {e}")

    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()
