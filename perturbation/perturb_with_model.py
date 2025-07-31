import os
import json
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK
from tqdm import tqdm

# 配置
model_path = os.path.join(MODEL_PATH, "animacy_bert_model")
label_map = {0: "human", 1: "animal", 2: "inanimate", 3: "event"}
animacy_rank = {"human": 3, "animal": 2, "inanimate": 1, "event": 0}

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

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

def process_all(mode="rule", strategy="A+P"):
    assert strategy in {"A+P", "A_only", "P_only", "none"}

    structured_dir = os.path.join(DATA_PATH, "structured")
    perturbed_dir = os.path.join(DATA_PATH, f"perturbed_model/{mode}_{strategy}")
    os.makedirs(perturbed_dir, exist_ok=True)

    jsonl_files = [f for f in os.listdir(structured_dir) if f.endswith("_verbs.jsonl")]
    if len(jsonl_files) != 1:
        raise ValueError(f"Expected one _verbs.jsonl file, found: {jsonl_files}")

    input_path = os.path.join(structured_dir, jsonl_files[0])
    prefix = jsonl_files[0].replace("_parsed_verbs.jsonl", "")
    out_affected = os.path.join(perturbed_dir, f"{prefix}_affected.txt")
    out_unaffected = os.path.join(perturbed_dir, f"{prefix}_unaffected.txt")
    out_invalid = os.path.join(perturbed_dir, f"{prefix}_invalid.txt")

    affected_lines, unaffected_lines, invalid_lines = [], [], []

    with open(input_path, encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, desc=f"Processing {mode}-{strategy}", total=len(lines)):
            data = json.loads(line)
            tokens = data.get("tokens", [])
            if not data.get("verbs"):
                invalid_lines.append(" ".join(tok["text"] for tok in tokens))
                continue

            sentence = " ".join(tok["text"] for tok in tokens)
            has_valid = False
            has_affected = False
            spans = []

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

                if strategy == "full":
                    should_perturb = True
                else:
                    if (
                        (mode == "rule" and should_perturb_rule(subj_cat, obj_cat)) or
                        (mode == "heuristic" and should_perturb_heuristic(subj_cat))
                    ):
                        should_perturb = True
                if should_perturb:
                    has_affected = True
                    if strategy != "none":
                        if strategy in {"A+P", "A_only"}:
                            subj = dict(subj)
                            subj["text"] += f" {AGENT_MARK}"
                            spans.append(subj)
                        if strategy in {"A+P", "P_only"}:
                            obj = dict(obj)
                            obj["text"] += f" {PATIENT_MARK}"
                            spans.append(obj)

            if has_affected:
                new_sent = apply_spans_to_tokens(tokens, spans)
                affected_lines.append(new_sent)
            elif has_valid:
                unaffected_lines.append(sentence)
            else:
                invalid_lines.append(sentence)

    with open(out_affected, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in affected_lines)
    with open(out_unaffected, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in unaffected_lines)
    with open(out_invalid, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in invalid_lines)

    print(f"Mode: {mode} | Strategy: {strategy}")
    print(f"Affected: {len(affected_lines)}")
    print(f"Unaffected: {len(unaffected_lines)}")
    print(f"Invalid: {len(invalid_lines)}")
    print(f"Saved to {perturbed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["rule", "heuristic"], default="rule")
    parser.add_argument("--strategy", type=str, choices=["A+P", "A_only", "P_only", "none", "full"], default="A+P")
    args = parser.parse_args()
    process_all(mode=args.mode, strategy=args.strategy)
