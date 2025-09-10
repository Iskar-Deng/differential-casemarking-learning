# perturbation/extract_verb.py
import os
import json
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
from utils import DATA_PATH

def build_children_index(tokens):
    """tokens: list[dict] from parser; return {head_id: [child_token_dict,...]}"""
    children = defaultdict(list)
    for t in tokens:
        # Protect unknown head ids
        head_id = t.get("head", -1)
        if isinstance(head_id, int):
            children[head_id].append(t)
    return children

def extract_np_span(token_id, id_to_token, children):
    head = id_to_token[token_id]
    span_tokens = [head]
    for child in children[token_id]:
        if child.get("dep") in {"det", "amod", "compound", "poss", "nmod"}:
            span_tokens.append(child)
    span_tokens.sort(key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    return {
        "text": " ".join(t["text"] for t in span_tokens),
        "span": [min(ids), max(ids)],
        "head": head["text"]
    }

def extract_clause_span(token_id, id_to_token, children):
    visited = set()
    to_visit = [token_id]
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        for child in children.get(current, []):
            cid = child.get("id")
            if isinstance(cid, int) and cid not in visited:
                to_visit.append(cid)
    span_tokens = sorted([id_to_token[i] for i in visited if i in id_to_token], key=lambda x: x["id"])
    if not span_tokens:
        return {"text": "", "span": [token_id, token_id], "head": id_to_token[token_id]["text"]}
    ids = [t["id"] for t in span_tokens]
    return {
        "text": " ".join(t["text"] for t in span_tokens),
        "span": [min(ids), max(ids)],
        "head": id_to_token[token_id]["text"]
    }

def extract_verb_arguments(tokens):
    """
    tokens: list of dicts with keys:
      id, text, lemma, pos, dep, head, ner
    """
    if not tokens:
        return []

    id_to_token = {t["id"]: t for t in tokens if isinstance(t.get("id"), int)}
    children = build_children_index(tokens)

    verbs = [t for t in tokens if t.get("pos") in {"VERB"}]

    results = []
    for verb in verbs:
        vid = verb["id"]
        result = {
            "verb": verb.get("lemma", verb.get("text", "")),
            "verb_id": vid,
            "subject": None,
            "objects": []
        }

        for t in children.get(vid, []):
            dep = t.get("dep")
            tid = t.get("id")
            if not isinstance(tid, int):
                continue

            if dep in {"nsubj", "nsubjpass", "csubj"}:
                try:
                    result["subject"] = extract_np_span(tid, id_to_token, children)
                except Exception:
                    result["subject"] = {"text":"", "span":[tid, tid], "head": id_to_token[tid]["text"]}

            elif dep in {"dobj", "obj", "attr", "dative"}:
                try:
                    span = extract_np_span(tid, id_to_token, children)
                except Exception:
                    span = {"text":"", "span":[tid, tid], "head": id_to_token[tid]["text"]}
                span["dep"] = dep
                result["objects"].append(span)

            elif dep in {"xcomp", "ccomp"}:
                span = extract_clause_span(tid, id_to_token, children)
                span["dep"] = dep
                result["objects"].append(span)

            elif dep == "prep":
                prep_token = t
                subtree_tokens = [prep_token]
                for ch in children.get(prep_token["id"], []):
                    if ch.get("dep") == "pobj":
                        subtree_tokens.append(ch)
                        for gc in children.get(ch["id"], []):
                            if gc.get("dep") in {"det", "amod", "compound", "poss"}:
                                subtree_tokens.append(gc)
                if len(subtree_tokens) > 1:
                    ids = [tok["id"] for tok in subtree_tokens if isinstance(tok.get("id"), int)]
                    span_sorted = sorted([tok for tok in subtree_tokens if isinstance(tok.get("id"), int)],
                                         key=lambda x: x["id"])
                    pobj_head = next((tok["text"] for tok in subtree_tokens if tok.get("dep") == "pobj"), None)
                    result["objects"].append({
                        "text": " ".join(tok["text"] for tok in span_sorted),
                        "span": [min(ids), max(ids)] if ids else [vid, vid],
                        "head": pobj_head,
                        "dep": "prep+pobj"
                    })

        results.append(result)

    return results

def process_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_lines = 0
    n_empty = 0
    n_errors = 0
    n_verbs = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Extracting {os.path.basename(input_path)}"):
            if not line.strip():
                continue
            n_lines += 1
            try:
                data = json.loads(line)
            except Exception:
                n_errors += 1
                continue

            tokens = data.get("tokens") or []
            idx = data.get("index", -1)
            if not tokens:
                n_empty += 1
                fout.write(json.dumps({"index": idx, "tokens": [], "verbs": []}, ensure_ascii=False) + "\n")
                continue

            try:
                verbs = extract_verb_arguments(tokens)
            except Exception as e:
                n_errors += 1
                verbs = []

            n_verbs += len(verbs)
            fout.write(json.dumps({"index": idx, "tokens": tokens, "verbs": verbs}, ensure_ascii=False) + "\n")

    return {
        "lines": n_lines,
        "empty": n_empty,
        "errors": n_errors,
        "verbs_total": n_verbs,
        "out": output_path,
    }

def process_all():
    parsed_dir = os.path.join(DATA_PATH, "parsed")
    structured_dir = os.path.join(DATA_PATH, "structured")
    os.makedirs(structured_dir, exist_ok=True)

    # 处理所有 *_parsed.jsonl（train/test/valid）
    jsonl_files = sorted([f for f in os.listdir(parsed_dir) if f.endswith("_parsed.jsonl")])
    if not jsonl_files:
        print(f"[ERROR] No *_parsed.jsonl in {parsed_dir}", file=sys.stderr)
        sys.exit(1)

    report = {}
    for fname in jsonl_files:
        input_path = os.path.join(parsed_dir, fname)
        output_path = os.path.join(structured_dir, fname.replace("_parsed.jsonl", "_verbs.jsonl"))
        stats = process_file(input_path, output_path)
        key = fname.replace("_parsed.jsonl", "")
        report[key] = stats
        print(f"[DONE] {output_path} | lines={stats['lines']} empty={stats['empty']} "
              f"errors={stats['errors']} verbs={stats['verbs_total']}")

    print("\n[SUMMARY]")
    for k, s in report.items():
        print(f" - {k}: lines={s['lines']}, empty={s['empty']}, errors={s['errors']}, verbs={s['verbs_total']}, out={s['out']}")

if __name__ == "__main__":
    process_all()
