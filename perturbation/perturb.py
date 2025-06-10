# perturbation/perturb.py

import os
import json
import argparse
from utils import (
    DATA_PATH,
    AGENT_MARK,
    PATIENT_MARK,
    compare_animacy,
    should_perturb_heuristic
)

def is_valid_structure(entry):
    return entry.get("subject") and len(entry.get("objects", [])) == 1

def should_perturb_rule(subj_head, obj_head):
    return compare_animacy(subj_head, obj_head) in {"lower", "equal"}

def apply_spans_to_tokens(tokens, spans):
    """
    Replace tokens with marked spans (e.g., add AGENT or PATIENT mark),
    while removing overlapping span tokens.
    """
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

def process_all(mode: str = "rule"):
    structured_dir = os.path.join(DATA_PATH, "structured")
    perturbed_dir = os.path.join(DATA_PATH, "perturbed", mode)
    os.makedirs(perturbed_dir, exist_ok=True)

    jsonl_files = [f for f in os.listdir(structured_dir) if f.endswith("_verbs.jsonl")]
    if len(jsonl_files) != 1:
        raise ValueError(f"Expected exactly one _verbs.jsonl file in {structured_dir}, found: {jsonl_files}")

    input_file = jsonl_files[0]
    input_path = os.path.join(structured_dir, input_file)

    prefix = input_file.replace("_parsed_verbs.jsonl", "")
    suffix = "_affected.txt"

    out_affected = os.path.join(perturbed_dir, f"{prefix}{suffix}")
    out_unaffected = os.path.join(perturbed_dir, f"{prefix}_unaffected.txt")
    out_invalid = os.path.join(perturbed_dir, f"{prefix}_invalid.txt")

    affected_lines, unaffected_lines, invalid_lines = [], [], []

    with open(input_path, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            tokens = data.get("tokens", [])
            index = data.get("index", -1)

            if not data.get("verbs"):
                invalid_lines.append(" ".join(tok["text"] for tok in tokens))
                continue

            has_affected = False
            has_valid = False
            spans = []

            for entry in data["verbs"]:
                if not is_valid_structure(entry):
                    continue
                has_valid = True
                subj = entry["subject"]
                obj = entry["objects"][0]

                if obj["dep"] == "ccomp":
                    continue

                if (
                    (mode == "rule" and should_perturb_rule(subj["head"], obj["head"])) or
                    (mode == "heuristic" and should_perturb_heuristic(subj["head"]))
                ):
                    has_affected = True
                    subj = dict(subj)
                    obj = dict(obj)
                    subj["text"] += f" {AGENT_MARK}"
                    obj["text"] += f" {PATIENT_MARK}"
                    spans.extend([subj, obj])

            if has_affected:
                new_sent = apply_spans_to_tokens(tokens, spans)
                affected_lines.append(new_sent)
            elif has_valid:
                unaffected_lines.append(" ".join(tok["text"] for tok in tokens))
            else:
                invalid_lines.append(" ".join(tok["text"] for tok in tokens))

    with open(out_affected, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in affected_lines)
    with open(out_unaffected, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in unaffected_lines)
    with open(out_invalid, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in invalid_lines)

    print(f"Mode: {mode}")
    print(f"Affected sentences: {len(affected_lines)}")
    print(f"Unaffected sentences: {len(unaffected_lines)}")
    print(f"Invalid sentences: {len(invalid_lines)}")
    print(f"Files saved to: {perturbed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply animacy-based or heuristic case marking perturbation.")
    parser.add_argument("--mode", type=str, choices=["rule", "heuristic"], default="rule", help="Perturbation strategy")
    args = parser.parse_args()

    process_all(mode=args.mode)
