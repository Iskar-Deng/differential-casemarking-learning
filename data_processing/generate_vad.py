import os
import json
import random
from utils import DATA_PATH

BASE_DIR = os.path.join(DATA_PATH, "perturbed_model")
OUT_DIR_NAME = "train_without_invalid"
VALIDATION_RATIO = 0.05
SEED = 42

def collect_lines(file_path):
    if not os.path.isfile(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_jsonl(lines, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")

def process_subdir(subdir_path, name):
    affected_path = os.path.join(subdir_path, f"{name}_affected.txt")
    unaffected_path = os.path.join(subdir_path, f"{name}_unaffected.txt")
    
    lines = collect_lines(affected_path) + collect_lines(unaffected_path)

    if not lines:
        print(f"No data found for {name}")
        return

    random.seed(SEED)
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - VALIDATION_RATIO))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    out_dir = os.path.join(subdir_path, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    save_jsonl(train_lines, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val_lines, os.path.join(out_dir, "validation.jsonl"))

    print(f"Processed {name} → {len(train_lines)} train / {len(val_lines)} val")

def batch_process_all():
    for subdir in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(full_path):
            continue

        files = [f for f in os.listdir(full_path) if f.endswith("_affected.txt")]
        if not files:
            print(f"Skipping {subdir} (no *_affected.txt found)")
            continue

        prefix = files[0].replace("_affected.txt", "")
        process_subdir(full_path, prefix)

if __name__ == "__main__":
    batch_process_all()
