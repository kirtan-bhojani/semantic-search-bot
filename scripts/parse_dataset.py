import json
from tqdm import tqdm

input_file = '../data/meta_appliances.json'
output_file = '../products_cleaned.json'

products = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc='Parsing product metadata'):
        item = json.loads(line)
        asin = item.get('asin', '')
        title = item.get('title', '').strip()
        description = item.get('description', '')
        if isinstance(description, list):
            description = ' '.join(description)
        categories = item.get('category', [])

        if asin and title:
            products.append({
                'asin': asin,
                'title': title,
                'description': description,
                'category': categories
            })

print(f"Extracted {len(products)} products")

with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(products, f_out, indent=2)

print(f"Cleaned product metadata saved to {output_file}")
