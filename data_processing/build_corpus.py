#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch & merge English corpus (WikiText-103 only).
- 默认下载并合并 WikiText-103 (raw) 的 train/validation/test。
- 可选：spaCy 分句 + 词级长度过滤；GPT-2 BPE 子词计数；目标 token 上限。
- 输出：
  $DATA_PATH/raw/<outname>.txt
  $DATA_PATH/raw/<outname>_stats.json

用法示例
--------
# 全部拉下（默认），分句+长度过滤
python -m data_processing.build_corpus \
  --outname wiki_all \
  --sentencize --min-len 5 --max-len 80

# 设定 120M BPE token 预算（达到阈值即停）
python -m data_processing.build_corpus \
  --outname wiki_120m \
  --target-tokens 120_000_000 \
  --sentencize --min-len 5 --max-len 80
"""

import os
import json
import argparse
from pathlib import Path
from typing import Iterator, Iterable, Optional, Dict

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    import spacy
except Exception:
    spacy = None

from utils import DATA_PATH


# -----------------------------
# Tokenizer
# -----------------------------
def get_gpt2_tokenizer(cache_dir: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = 10**9  # 避免仅用于计数时的截断警告
    return tok


def count_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False, padding=False)["input_ids"])


# -----------------------------
# Sentencizer (optional)
# -----------------------------
def build_sentencizer(model: str = "en_core_web_sm"):
    if spacy is None:
        print("[Warn] spaCy not installed; --sentencize will be ignored.")
        return None
    try:
        nlp = spacy.load(model, disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        print(f"[Warn] spaCy model '{model}' not found. Run: python -m spacy download {model}")
        return None
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def maybe_sentencize(nlp, text: str) -> Iterable[str]:
    if nlp is None:
        s = text.strip()
        return [s] if s else []
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# -----------------------------
# Source: WikiText-103
# -----------------------------
def stream_wikitext103() -> Iterator[str]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    for sp in ("train", "validation", "test"):
        for ex in ds[sp]:
            t = (ex.get("text") or "").strip()
            if t:
                yield t


# -----------------------------
# Builder
# -----------------------------
def build_wiki_only(
    target_tokens: Optional[int],  # None 或 0 => 无上限（下载全部）
    outname: str,
    sentencize: bool,
    min_len: int,
    max_len: int,
    cache_dir: Optional[str] = None,
):
    raw_dir = Path(DATA_PATH) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_txt = raw_dir / f"{outname}.txt"
    out_stats = raw_dir / f"{outname}_stats.json"

    tok = get_gpt2_tokenizer(cache_dir)
    nlp = build_sentencizer() if sentencize else None

    unlimited = (target_tokens is None) or (int(target_tokens) <= 0)
    target_tokens = None if unlimited else int(target_tokens)

    stats: Dict = {
        "source": "wikitext-103-raw-v1",
        "target_subword_tokens": target_tokens if target_tokens is not None else "ALL",
        "filters": {
            "sentencize": bool(sentencize),
            "min_len": int(min_len),
            "max_len": int(max_len),
        },
        "totals": {"kept_lines": 0, "kept_word_tokens": 0, "kept_subword_tokens": 0},
        "out_txt": str(out_txt),
    }

    if unlimited:
        print("[Info] Token budget: ALL")
    else:
        print(f"[Info] Token budget: {target_tokens:,}")
    if sentencize:
        print(f"[Info] Sentencize + length filter: [{min_len}, {max_len}]")

    pbar = tqdm(total=target_tokens, desc=f"Collecting into {outname}.txt", unit="tok", disable=unlimited)

    collected = 0
    with out_txt.open("w", encoding="utf-8") as fout:
        for chunk in stream_wikitext103():
            sentences = maybe_sentencize(nlp, chunk) if sentencize else [chunk]

            for sent in sentences:
                toks = sent.split()
                if not toks:
                    continue
                if not (min_len <= len(toks) <= max_len):
                    continue

                n_sub = count_tokens(tok, sent)

                fout.write(sent + "\n")

                stats["totals"]["kept_lines"] += 1
                stats["totals"]["kept_word_tokens"] += len(toks)
                stats["totals"]["kept_subword_tokens"] += n_sub

                if not unlimited:
                    collected += n_sub
                    pbar.update(n_sub)
                    if collected >= target_tokens:
                        pbar.close()
                        with out_stats.open("w", encoding="utf-8") as f:
                            json.dump(stats, f, ensure_ascii=False, indent=2)
                        print(f"[OK] Wrote merged corpus: {out_txt}")
                        print(f"[OK] Stats JSON: {out_stats}")
                        return

    if not unlimited:
        pbar.close()

    with out_stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote ALL WikiText-103 into: {out_txt}")
    print(f"[OK] Stats JSON: {out_stats}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outname", type=str, required=True, help="Output file stem under $DATA_PATH/raw/")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="Target GPT-2 BPE tokens; 0/absent = ALL (no limit).")
    ap.add_argument("--sentencize", action="store_true", help="Use spaCy sentencizer before length filtering.")
    ap.add_argument("--min-len", type=int, default=1, help="Min words per sentence after sentencize.")
    ap.add_argument("--max-len", type=int, default=10**9, help="Max words per sentence after sentencize.")
    ap.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache dir.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_wiki_only(
        target_tokens=(None if args.target_tokens is None or args.target_tokens <= 0 else int(args.target_tokens)),
        outname=args.outname,
        sentencize=args.sentencize,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        cache_dir=args.cache_dir,
    )
