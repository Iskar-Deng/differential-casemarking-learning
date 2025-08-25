# data_processing/prepare_corpus.py
import argparse, os, re, sys
from utils import DATA_PATH

SPLITS = {
    "train": "train.txt",
    "valid": "valid.txt",
    "test":  "test.txt",
}

def get_splitter(mode: str):
    if mode == "spacy_trf":
        import spacy
        nlp = spacy.load("en_core_web_trf")
        return lambda t: [s.text.strip() for s in nlp(t).sents if s.text.strip()]
    if mode == "spacy_sm":
        import spacy
        nlp = spacy.load("en_core_web_sm")
        if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return lambda t: [s.text.strip() for s in nlp(t).sents if s.text.strip()]
    if mode == "nltk":
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize
        return lambda t: [s.strip() for s in sent_tokenize(t) if s.strip()]
    # fallback: 简单正则按 . ! ? 后空白切句
    splitter = re.compile(r'(?<=[.!?])\s+')
    return lambda t: [p.strip() for p in splitter.split(t.strip()) if p.strip()]

def process_one_file(in_path, out_path, split_fn, min_len, max_len):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kept_sent = 0
    kept_tok = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # 先切句
            for s in split_fn(line):
                n_tok = len(s.split())
                if min_len <= n_tok <= max_len:
                    fout.write(s + "\n")
                    kept_sent += 1
                    kept_tok  += n_tok
    print(f"[DONE] {os.path.basename(in_path)} -> {out_path} | "
          f"sentences={kept_sent}, tokens={kept_tok}")
    return kept_sent, kept_tok

def main(mode, min_len, max_len):
    raw_dir = os.path.join(DATA_PATH, "raw")
    out_dir = os.path.join(DATA_PATH, "filtered")
    splitter = get_splitter(mode)

    total_sent = 0
    total_tok = 0

    for split, fname in SPLITS.items():
        in_path = os.path.join(raw_dir, fname)
        if not os.path.isfile(in_path):
            print(f"[WARN] missing: {in_path}, skip.", file=sys.stderr)
            continue
        out_path = os.path.join(out_dir, f"{split}.txt")
        s, t = process_one_file(in_path, out_path, splitter, min_len, max_len)
        total_sent += s
        total_tok  += t

    print(f"[TOTAL] sentences={total_sent}, tokens={total_tok}")
    return 0 if total_sent > 0 else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Split sentences and filter by length for train/valid/test under DATA_PATH/raw/"
    )
    ap.add_argument("--mode", type=str, default="regex",
                    choices=["regex", "nltk", "spacy_sm", "spacy_trf"],
                    help="Sentence splitter to use.")
    ap.add_argument("--min_len", type=int, default=5, help="Minimum tokens per sentence.")
    ap.add_argument("--max_len", type=int, default=25, help="Maximum tokens per sentence.")
    args = ap.parse_args()
    sys.exit(main(args.mode, args.min_len, args.max_len))
