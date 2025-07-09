import json

input_file = '../data/electronics_5.json'

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 0:
            print(line)
        if i > 5:
            break
