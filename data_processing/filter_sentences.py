#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter corpus sentences by length.

Input:  $DATA_PATH/raw  containing *_train.txt, *_valid.txt, *_test.txt  
Output: $DATA_PATH/filtered with filtered files and a summary JSON.
"""

import argparse
import sys
import json
import re
from pathlib import Path

import spacy
from tqdm import tqdm
from utils import DATA_PATH


def regex_sentencize(text: str):
    """Split text into sentences using regex."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def filter_sentences(min_len=5, max_len=25, split_mode="spacy", only=None, overwrite=False):
    raw_dir = Path(DATA_PATH) / "raw"
    filtered_dir = Path(DATA_PATH) / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    txt_files = [p for p in raw_dir.glob("*.txt") if any(suf in p.stem for suf in ("train", "valid", "test"))]
    if only:
        txt_files = [p for p in txt_files if only in p.stem]

    if not txt_files:
        print(f"[Error] No matching .txt files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    nlp = None
    if split_mode == "spacy":
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            print("[Error] spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm",
                  file=sys.stderr)
            sys.exit(1)
        if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    total_raw_sents, total_raw_tokens = 0, 0
    total_filt_sents, total_filt_tokens = 0, 0
    stats = {"files": {}, "summary": {}}

    for input_path in txt_files:
        output_path = filtered_dir / input_path.name
        if output_path.exists() and not overwrite:
            print(f"[Skip] {output_path} already exists, skipping...")
            continue

        filtered = []
        raw_sents = raw_tokens = filt_sents = filt_tokens = 0

        with open(input_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)

        print(f"[Info] Processing {input_path} ({n_lines:,} lines) with split_mode={split_mode}")
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=n_lines, desc=f"Filtering {input_path.name}", unit="lines"):
                line = line.strip()
                if not line:
                    continue

                if split_mode == "spacy":
                    doc = nlp(line)
                    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
                else:
                    sentences = regex_sentencize(line)

                for sent in sentences:
                    tokens = sent.split()
                    if not tokens:
                        continue
                    raw_sents += 1
                    raw_tokens += len(tokens)
                    if min_len <= len(tokens) <= max_len:
                        filt_sents += 1
                        filt_tokens += len(tokens)
                        filtered.append(sent)

        with open(output_path, "w", encoding="utf-8") as f_out:
            for sent in tqdm(filtered, total=len(filtered), desc=f"Writing {output_path.name}", unit="sents"):
                f_out.write(sent + "\n")

        print(f"[OK] {filt_sents} sentences written to {output_path}")
        print(f"[Stats] {input_path.name}: "
              f"raw_sents={raw_sents}, raw_tokens={raw_tokens} | "
              f"kept_sents={filt_sents}, kept_tokens={filt_tokens}")

        total_raw_sents += raw_sents
        total_raw_tokens += raw_tokens
        total_filt_sents += filt_sents
        total_filt_tokens += filt_tokens

        stats["files"][input_path.name] = {
            "raw_sents": raw_sents,
            "raw_tokens": raw_tokens,
            "kept_sents": filt_sents,
            "kept_tokens": filt_tokens,
            "dropped_sents": raw_sents - filt_sents,
            "dropped_tokens": raw_tokens - filt_tokens,
            "output_file": str(output_path),
        }

    stats["summary"] = {
        "raw_sents": total_raw_sents,
        "raw_tokens": total_raw_tokens,
        "kept_sents": total_filt_sents,
        "kept_tokens": total_filt_tokens,
        "dropped_sents": total_raw_sents - total_filt_sents,
        "dropped_tokens": total_raw_tokens - total_filt_tokens,
    }

    stats_path = filtered_dir / "filter_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f_json:
        json.dump(stats, f_json, ensure_ascii=False, indent=2)

    print("\n[Summary]")
    for k, v in stats["summary"].items():
        print(f"  {k:15}: {v}")
    print(f"[OK] Stats written to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter corpus sentences by length")
    parser.add_argument("--min_len", type=int, default=3, help="Minimum number of words")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum number of words")
    parser.add_argument("--split-mode", type=str, choices=["spacy", "regex"], default="spacy",
                        help="Sentence splitting mode")
    parser.add_argument("--only", type=str, choices=["train", "valid", "test"],
                        help="Process only one split")
    parser.add_argument("--overwrite", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    filter_sentences(
        min_len=args.min_len,
        max_len=args.max_len,
        split_mode=args.split_mode,
        only=args.only,
        overwrite=args.overwrite,
    )
