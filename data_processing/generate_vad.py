import os
import json
import random
import re
from utils import DATA_PATH

BASE_DIR = os.path.join(DATA_PATH, "perturbed_model")
OUT_DIR_NAME_WO = "train_without_invalid"
OUT_DIR_NAME_WI = "train_with_invalid"
VOCAB_DIR = os.path.join(DATA_PATH, "vocab")
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

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.strip().split()

def build_vocab(lines, vocab_path):
    vocab = set()
    for line in lines:
        vocab.update(tokenize(line))
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")
    print(f"Vocab saved to {vocab_path} ({len(vocab)} words)")

def split_and_save(lines, out_dir, vocab_path, seed=SEED):
    if not lines:
        os.makedirs(out_dir, exist_ok=True)
        save_jsonl([], os.path.join(out_dir, "train.jsonl"))
        save_jsonl([], os.path.join(out_dir, "validation.jsonl"))
        build_vocab([], vocab_path)
        print(f"Processed (empty) → 0 train / 0 val → {out_dir}")
        return

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - VALIDATION_RATIO))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    os.makedirs(out_dir, exist_ok=True)
    save_jsonl(train_lines, os.path.join(out_dir, "train.jsonl"))
    save_jsonl(val_lines, os.path.join(out_dir, "validation.jsonl"))

    build_vocab(train_lines, vocab_path)

    print(f"Processed → {len(train_lines)} train / {len(val_lines)} val → {out_dir}")

def process_subdir(subdir_path, name):
    affected_path = os.path.join(subdir_path, f"{name}_affected.txt")
    unaffected_path = os.path.join(subdir_path, f"{name}_unaffected.txt")
    invalid_path = os.path.join(subdir_path, f"{name}_invalid.txt")

    lines_A = collect_lines(affected_path)
    lines_UA = collect_lines(unaffected_path)
    lines_INV = collect_lines(invalid_path)

    lines_wo = lines_A + lines_UA
    lines_wi = lines_A + lines_UA + lines_INV

    if not lines_wo and not lines_wi:
        print(f"No data found for {name} (A/UA/INV all empty)")
        return

    dataset_name = os.path.basename(subdir_path)
    out_dir_wo = os.path.join(subdir_path, OUT_DIR_NAME_WO)
    out_dir_wi = os.path.join(subdir_path, OUT_DIR_NAME_WI)

    os.makedirs(VOCAB_DIR, exist_ok=True)
    vocab_wo = os.path.join(VOCAB_DIR, f"{dataset_name}_vocab.txt")
    vocab_wi = os.path.join(VOCAB_DIR, f"{dataset_name}_with_invalid_vocab.txt")

    if lines_wo:
        split_and_save(lines_wo, out_dir_wo, vocab_wo, seed=SEED)
    else:
        print(f"⚠️ {name}: no A/UA lines; skipped writing {OUT_DIR_NAME_WO}")

    if lines_wi:
        split_and_save(lines_wi, out_dir_wi, vocab_wi, seed=SEED)
    else:
        print(f"⚠️ {name}: no lines including invalid; skipped writing {OUT_DIR_NAME_WI}")

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
