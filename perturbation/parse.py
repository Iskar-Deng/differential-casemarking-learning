#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constituency parsing with spaCy + benepar (batch mode).

Input:  $DATA_PATH/filtered/*.txt
Output: $DATA_PATH/parsed/*.jsonl + bad line logs
Features:
    - Resume from previous runs (--resume)
    - Automatic batch fallback
    - Periodic tqdm updates
"""

import os
import sys
import json
import pathlib
import argparse
from typing import Iterator, List, Tuple

import spacy
import benepar
from tqdm import tqdm
from utils import DATA_PATH


filtered_dir = os.path.join(DATA_PATH, "filtered")
parsed_dir = os.path.join(DATA_PATH, "parsed")
os.makedirs(parsed_dir, exist_ok=True)

txt_files = sorted([f for f in os.listdir(filtered_dir) if f.endswith(".txt")])
if not txt_files:
    print(f"[ERROR] no .txt found in {filtered_dir}", file=sys.stderr)
    sys.exit(1)


def build_nlp(model_size: str = "sm"):
    model_name = "en_core_web_trf" if model_size == "trf" else "en_core_web_sm"
    try:
        nlp = spacy.load(model_name, disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        print(f"[INFO] spaCy model {model_name} not found. Run:", file=sys.stderr)
        print(f"       python -m spacy download {model_name}", file=sys.stderr)
        sys.exit(1)

    try:
        benepar.load_trained_model("benepar_en3")
    except Exception:
        print("[INFO] downloading benepar_en3 ...", file=sys.stderr)
        benepar.download("benepar_en3")

    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)

    nlp.max_length = 5_000_000
    return nlp


def annotate_doc(doc, index: int):
    try:
        sent_doc = list(doc.sents)[0]
    except Exception:
        return {"index": index, "tokens": [], "ptb": "", "error": "no_sentence"}

    tree = getattr(sent_doc._, "parse_string", "")
    tokens = [
        {
            "id": tok.i,
            "text": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "dep": tok.dep_,
            "head": tok.head.i,
            "ner": tok.ent_type_ or "O",
        }
        for tok in sent_doc
    ]
    return {"index": index, "tokens": tokens, "ptb": tree}


def clean_text(s: str) -> str:
    return s.replace("\u200b", "").replace("\u200e", "").replace("\ufeff", "").strip()


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_nonempty_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_resume_index(jsonl_path: str) -> int:
    try:
        if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0:
            return -1
        with open(jsonl_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = size
            chunk = 1024
            buf = b""
            while pos > 0:
                pos = max(0, pos - chunk)
                f.seek(pos)
                buf = f.read(size - pos) + buf
                if b"\n" in buf:
                    break
            last = buf.splitlines()[-1].decode("utf-8", "ignore").strip()
        if not last:
            return -1
        obj = json.loads(last)
        return int(obj.get("index", -1))
    except Exception:
        return -1


def nonempty_texts_and_idxs(path: str, start_idx: int) -> Iterator[Tuple[str, int]]:
    skipped = 0
    i = start_idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if skipped < start_idx:
                skipped += 1
                continue
            yield (s, i)
            i += 1


def batched(iterable, n: int):
    batch: List[Tuple[str, int]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def main(model_size="sm", batch_size=32, n_process=1, resume=False):
    if n_process != 1:
        print("[Warn] benepar + spaCy multiprocessing may raise serialization errors; "
              "n_process=1 is recommended.", file=sys.stderr)

    nlp = build_nlp(model_size)

    for name in txt_files:
        src_path = os.path.join(filtered_dir, name)
        dst_path = os.path.join(parsed_dir, name.replace(".txt", "_parsed.jsonl"))
        bad_path = os.path.join(parsed_dir, name.replace(".txt", "_badlines.log"))

        total_lines = count_lines(src_path)
        total_nonempty = count_nonempty_lines(src_path)

        resume_from = -1
        if resume and os.path.exists(dst_path):
            resume_from = get_resume_index(dst_path)
            if resume_from >= 0:
                print(f"[RESUME] {name}: last index = {resume_from}")
            else:
                print(f"[RESUME] {name}: no valid last index found")

        start_idx = resume_from + 1
        remaining = max(0, total_nonempty - start_idx)
        print(f"[INFO] parsing {name} -> {pathlib.Path(dst_path).name} "
              f"(lines={total_lines}, nonempty={total_nonempty}, start_idx={start_idx}, remaining={remaining})")

        bad = 0
        fout_mode = "a" if (resume and os.path.exists(dst_path) and resume_from >= 0) else "w"
        fbad_mode = "a" if (resume and os.path.exists(bad_path) and resume_from >= 0) else "w"

        with open(dst_path, fout_mode, encoding="utf-8") as fout, \
             open(bad_path, fbad_mode, encoding="utf-8") as fbad:

            pbar = tqdm(total=total_nonempty, initial=start_idx,
                        desc=f"Parsing {name}", mininterval=60, dynamic_ncols=True)

            stream = nonempty_texts_and_idxs(src_path, start_idx=start_idx)
            for batch in batched(stream, batch_size):
                texts = [t for t, _ in batch]
                idxs = [i for _, i in batch]

                try:
                    docs = list(nlp.pipe(texts, batch_size=batch_size, n_process=1))
                    assert len(docs) == len(idxs)
                    for doc, i in zip(docs, idxs):
                        try:
                            parsed = annotate_doc(doc, i)
                        except Exception as e:
                            bad += 1
                            fbad.write(f"{i}\t{type(e).__name__}: {e}\n")
                            parsed = {"index": i, "tokens": [], "ptb": "", "error": f"{type(e).__name__}: {e}"}
                        if parsed.get("error"):
                            bad += 1
                            fbad.write(f"{i}\t{parsed['error']}\n")
                        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        pbar.update(1)

                except Exception as batch_e:
                    for text, i in batch:
                        try:
                            doc = nlp(text)
                        except Exception:
                            try:
                                doc = nlp(clean_text(text))
                            except Exception as e2:
                                bad += 1
                                fbad.write(f"{i}\t{type(batch_e).__name__}/{type(e2).__name__}: {e2}\n")
                                fout.write(json.dumps(
                                    {"index": i, "tokens": [], "ptb": "", "error": f"{type(e2).__name__}: {e2}"},
                                    ensure_ascii=False
                                ) + "\n")
                                pbar.update(1)
                                continue

                        try:
                            parsed = annotate_doc(doc, i)
                        except Exception as e:
                            bad += 1
                            fbad.write(f"{i}\t{type(e).__name__}: {e}\n")
                            parsed = {"index": i, "tokens": [], "ptb": "", "error": f"{type(e).__name__}: {e}"}
                        if parsed.get("error"):
                            bad += 1
                            fbad.write(f"{i}\t{parsed['error']}\n")
                        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        pbar.update(1)

            pbar.close()
        print(f"[DONE] {dst_path} | bad_lines={bad} -> {pathlib.Path(bad_path).name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=["sm", "trf"], default="sm",
                    help="spaCy model size: sm (fast) or trf (accurate but slow)")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for parsing")
    ap.add_argument("--n-process", type=int, default=1, help="Number of processes (benepar prefers 1)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing parsed files")
    args = ap.parse_args()

    main(model_size=args.model_size,
         batch_size=args.batch_size,
         n_process=args.n_process,
         resume=args.resume)
