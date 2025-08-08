# data_processing/prepreprocess.py
import argparse, os, sys, re
from glob import glob
from utils import DATA_PATH

def get_splitter(mode):
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
    splitter = re.compile(r'(?<=[.!?])\s+')
    return lambda t: [p.strip() for p in splitter.split(t.strip()) if p.strip()]

def preprocess_to_single_file(input_subdir="train_100M", pattern="*.train", mode="regex"):
    in_dir = os.path.join(DATA_PATH, input_subdir)
    out_dir = os.path.join(DATA_PATH, "raw")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(in_dir):
        print(f"[ERROR] Input directory not found: {in_dir}", file=sys.stderr)
        return 2
    paths = sorted(glob(os.path.join(in_dir, pattern)))
    if not paths:
        print(f"[ERROR] No files match {pattern} in {in_dir}", file=sys.stderr)
        return 2
    split = get_splitter(mode)
    output_path = os.path.join(out_dir, "BabyLM_100M.txt")
    total_out = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for ip in paths:
            fname = os.path.basename(ip)
            fsize = os.path.getsize(ip)
            processed = 0
            last_bucket = 0
            print(f"[INFO] {fname}: 0%")
            with open(ip, "rb", buffering=1024*1024) as fin:
                for raw in fin:
                    processed += len(raw)
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text:
                        for s in split(text):
                            fout.write(s + "\n")
                            total_out += 1
                    if fsize > 0:
                        pct = int(processed * 100 / fsize)
                        bucket = pct // 5
                        if bucket > last_bucket:
                            print(f"[INFO] {fname}: {pct}%")
                            last_bucket = bucket
            if last_bucket < 20:
                print(f"[INFO] {fname}: 100%")
    print(f"[DONE] Total sentences written: {total_out} -> {output_path}")
    return 0 if total_out > 0 else 1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_subdir", type=str, default="train_100M")
    p.add_argument("--pattern", type=str, default="*.train")
    p.add_argument("--mode", type=str, default="regex", choices=["regex","nltk","spacy_sm","spacy_trf"])
    a = p.parse_args()
    sys.exit(preprocess_to_single_file(a.input_subdir, a.pattern, a.mode))
