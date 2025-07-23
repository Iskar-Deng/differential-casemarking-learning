import os
import json
import argparse
import torch
from collections import Counter
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK
from tqdm import tqdm

# Load model
model_path = os.path.join(MODEL_PATH, "animacy_bert_model")
label_map = {0: "human", 1: "animal", 2: "inanimate", 3: "event"}
animacy_rank = {"human": 3, "animal": 2, "inanimate": 1, "event": 0}

config = BertConfig.from_pretrained(model_path)
model = BertForSequenceClassification(config)
state_dict = load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(model_path)

def predict_animacy(sentence, np_text):
    text = f"{sentence} [NP] {np_text}"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]

def compare_animacy(subj_cat, obj_cat):
    if animacy_rank[subj_cat] > animacy_rank[obj_cat]:
        return "higher"
    elif animacy_rank[subj_cat] < animacy_rank[obj_cat]:
        return "lower"
    else:
        return "equal"

def should_perturb_rule(subj_cat, obj_cat):
    return compare_animacy(subj_cat, obj_cat) in {"lower", "equal"}

def should_perturb_heuristic(subj_cat):
    return subj_cat == "human"

def is_valid_structure(entry):
    return entry.get("subject") and len(entry.get("objects", [])) == 1

def apply_spans_to_tokens(tokens, spans):
    span_map = {span["span"][0]: span["text"] for span in spans}
    skip_ids = set()
    for span in spans:
        start, end = span["span"]
        skip_ids.update(range(start + 1, end + 1))
    output = []
    for i, tok in enumerate(tokens):
        if i in skip_ids:
            continue
        elif i in span_map:
            output.append(span_map[i])
        else:
            output.append(tok["text"])
    return " ".join(output)

def process_minimal_pairs(mode="rule", strategy="A+P", limit=None):
    assert strategy in {"A+P", "A_only", "P_only"}

    structured_dir = os.path.join(DATA_PATH, "structured")
    output_dir = os.path.join(DATA_PATH, f"minimal_pair/{mode}_{strategy}")
    os.makedirs(output_dir, exist_ok=True)

    jsonl_files = [f for f in os.listdir(structured_dir) if f.endswith("_verbs.jsonl")]
    if len(jsonl_files) != 1:
        raise ValueError(f"Expected one _verbs.jsonl file, found: {jsonl_files}")
    input_path = os.path.join(structured_dir, jsonl_files[0])
    prefix = jsonl_files[0].replace("_parsed_verbs.jsonl", "")
    out_jsonl = os.path.join(output_dir, f"{prefix}_minimal_pairs.jsonl")

    jsonl_pairs = []
    animacy_pairs = Counter()

    with open(input_path, encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, desc=f"Processing {mode}-{strategy}", total=len(lines)):
            if limit and len(jsonl_pairs) >= limit:
                break

            data = json.loads(line)
            tokens = data.get("tokens", [])
            if not data.get("verbs"):
                continue

            sentence = " ".join(tok["text"] for tok in tokens)
            has_valid = False

            for entry in data["verbs"]:
                if not is_valid_structure(entry):
                    continue
                has_valid = True

                subj = entry["subject"]
                obj = entry["objects"][0]
                if obj["dep"] == "ccomp":
                    continue

                subj_cat = predict_animacy(sentence, subj["text"])
                obj_cat = predict_animacy(sentence, obj["text"])
                animacy_pairs[(subj_cat, obj_cat)] += 1

                if (
                    (mode == "rule" and should_perturb_rule(subj_cat, obj_cat)) or
                    (mode == "heuristic" and should_perturb_heuristic(subj_cat))
                ):
                    # 标记后的句子更好（符合策略）
                    spans = []
                    if strategy in {"A+P", "A_only"}:
                        subj_marked = dict(subj)
                        subj_marked["text"] += f" {AGENT_MARK}"
                        spans.append(subj_marked)
                    if strategy in {"A+P", "P_only"}:
                        obj_marked = dict(obj)
                        obj_marked["text"] += f" {PATIENT_MARK}"
                        spans.append(obj_marked)
                    good = apply_spans_to_tokens(tokens, spans)
                    bad = sentence
                else:
                    # 原句更好（不应该添加标记）
                    spans = []
                    if strategy in {"A+P", "A_only"}:
                        subj_marked = dict(subj)
                        subj_marked["text"] += f" {AGENT_MARK}"
                        spans.append(subj_marked)
                    if strategy in {"A+P", "P_only"}:
                        obj_marked = dict(obj)
                        obj_marked["text"] += f" {PATIENT_MARK}"
                        spans.append(obj_marked)
                    bad = apply_spans_to_tokens(tokens, spans)
                    good = sentence

                jsonl_pairs.append({"good": good, "bad": bad})

    # Write to file
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in jsonl_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\nMode: {mode} | Strategy: {strategy}")
    print(f"Minimal pairs written: {len(jsonl_pairs)}")
    print(f"Saved to: {out_jsonl}")

    print("\nAnimacy (subject, object) pair counts:")
    for k, v in animacy_pairs.most_common():
        print(f"  {k}: {v}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["rule", "heuristic"], default="rule")
    parser.add_argument("--strategy", type=str, choices=["A+P", "A_only", "P_only"], default="A+P")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of minimal pairs")
    args = parser.parse_args()
    process_minimal_pairs(mode=args.mode, strategy=args.strategy, limit=args.limit)
