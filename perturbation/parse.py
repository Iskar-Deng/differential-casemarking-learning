# perturbation/parse.py

import spacy
import json
import pathlib
import tqdm
import sys
import os
from benepar import BeneparComponent
import benepar
from utils import DATA_PATH

nlp = spacy.load("en_core_web_trf")
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)

def annotate(sent: str, index: int):
    doc = nlp(sent.strip())
    sent_doc = list(doc.sents)[0]
    tree = sent_doc._.parse_string
    tokens = [
        {
            "id": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.i,
            "ner": token.ent_type_ or "O"
        }
        for token in sent_doc
    ]
    return {
        "index": index,
        "tokens": tokens,
        "ptb": tree
    }

filtered_dir = os.path.join(DATA_PATH, "filtered")
parsed_dir = os.path.join(DATA_PATH, "parsed")
os.makedirs(parsed_dir, exist_ok=True)

txt_files = [f for f in os.listdir(filtered_dir) if f.endswith(".txt")]
if len(txt_files) != 1:
    print(f"Expected exactly one .txt file in {filtered_dir}, found: {txt_files}", file=sys.stderr)
    sys.exit(1)

src_path = os.path.join(filtered_dir, txt_files[0])
dst_path = os.path.join(parsed_dir, txt_files[0].replace(".txt", "_parsed.jsonl"))

with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
    lines = [line.strip() for line in fin if line.strip()]
    for i, line in enumerate(tqdm.tqdm(lines, total=len(lines))):
        try:
            parsed = annotate(line, i)
        except Exception as e:
            parsed = {"index": i, "tokens": [], "ptb": "", "error": str(e)}
        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")

print(f"Parsed output saved to: {dst_path}")
