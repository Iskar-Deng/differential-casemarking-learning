# data_processing/filter_sentences.py

import argparse
import os
import sys
from utils import DATA_PATH

def filter_babylm_sentences(min_len=5, max_len=15):
    raw_dir = os.path.join(DATA_PATH, "raw")
    filtered_dir = os.path.join(DATA_PATH, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    # Find the single .txt file in raw/
    txt_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    if len(txt_files) != 1:
        print(f"Expecting exactly one .txt file in {raw_dir}, found: {txt_files}", file=sys.stderr)
        sys.exit(1)

    input_file = txt_files[0]
    input_path = os.path.join(raw_dir, input_file)
    output_path = os.path.join(filtered_dir, input_file)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("_book_title_") or \
           line.lower().startswith("chapter") or \
           line.lower().startswith("-lcb-"):
            continue
        tokens = line.split()
        if min_len <= len(tokens) <= max_len:
            filtered.append(line)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for sent in filtered:
            f_out.write(sent + "\n")

    print(f"{len(filtered)} sentences written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter BabyLM sentences of appropriate length")
    parser.add_argument("--min_len", type=int, default=5, help="Minimum number of words")
    parser.add_argument("--max_len", type=int, default=15, help="Maximum number of words")
    args = parser.parse_args()

    filter_babylm_sentences(min_len=args.min_len, max_len=args.max_len)
