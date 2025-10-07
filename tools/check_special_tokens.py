#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test: check whether AGENT_MARK / PATIENT_MARK are in tokenizer vocab
and how they are tokenized.
"""

import argparse
from transformers import AutoTokenizer
from utils import AGENT_MARK, PATIENT_MARK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, required=True,
                    help="Path to tokenizer dir or pretrained model name")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab = tok.get_vocab()

    for mark in [AGENT_MARK, PATIENT_MARK]:
        print(f"\n=== Checking token: {mark} ===")
        if mark in vocab:
            print(f"[OK] '{mark}' exists as single token (id={vocab[mark]})")
        else:
            print(f"[WARN] '{mark}' NOT in vocab — will be split into sub-tokens!")
        print("Tokenization result:", tok.tokenize(mark))

    # 额外：测一句包含标记的句子
    test_sent = f"he argued with his nurse {PATIENT_MARK} as soon as he could speak ."
    print(f"\nTest sentence: {test_sent}")
    print("Tokenized IDs:", tok.encode(test_sent))
    print("Decoded back :", tok.decode(tok.encode(test_sent)))

if __name__ == "__main__":
    main()
