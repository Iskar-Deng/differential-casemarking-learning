import os
import json
import random

input_txt = "/home/dengh/workspace/relational-casemarking-learning/data/perturbed/heuristic/combined.txt"         # 每行一句
output_dir = "/home/dengh/workspace/relational-casemarking-learning/data/perturbed/heuristic/train"
name = "heuristic"
validation_ratio = 0.05

with open(input_txt, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * (1 - validation_ratio))
train_data = lines[:split_idx]
val_data = lines[split_idx:]

train_data = [{"text": line} for line in train_data]
val_data = [{"text": line} for line in val_data]

os.makedirs(output_dir, exist_ok=True)
train_path = os.path.join(output_dir, f"{name}.train.jsonl")
val_path = os.path.join(output_dir, f"{name}.validation.jsonl")

with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
