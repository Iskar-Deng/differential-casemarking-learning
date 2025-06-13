# perturbation/perturb.py

import os
import json
import argparse
from collections import defaultdict
from utils import (
    DATA_PATH,
    AGENT_MARK,
    PATIENT_MARK,
    compare_animacy,
    should_perturb_heuristic,
    get_animacy_category
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

def safe_div(num, den):
    return 0.0 if den == 0 else num / den

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

    # --- NEW: 3×3 subject–object matrix ------------------------------------
    animacy_labels = ("human", "animal", "inanimate")
    pair_counts = {s: {o: 0 for o in animacy_labels} for s in animacy_labels}


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

                # ------ update animacy tallies ------
                subj_cat = get_animacy_category(subj["head"])
                obj_cat  = get_animacy_category(obj["head"])

                pair_counts[subj_cat][obj_cat] += 1

                # -----------------------------------------

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

    print("\nSubject × Object animacy counts (3×3):")
    header = " " * 10 + " | ".join(f"{o:^10}" for o in animacy_labels)
    print(header)
    print("-" * len(header))
    for s in animacy_labels:
        row = " ".join(f"{pair_counts[s][o]:>10}" for o in animacy_labels)
        print(f"{s:<10}|{row}")

    # ── overlap / divergence analysis ----------------------------------------
    # Always treat *rule* as gold and *heuristic* as the system prediction,
    # no matter which mode is currently running.
    rule_file      = os.path.join(DATA_PATH, "perturbed", "rule",
                                  f"{prefix}{suffix}")
    heuristic_file = os.path.join(DATA_PATH, "perturbed", "heuristic",
                                  f"{prefix}{suffix}")

    if os.path.isfile(rule_file) and os.path.isfile(heuristic_file):
        with open(rule_file, encoding="utf-8") as f:
            gold = {l.rstrip('\n') for l in f}
        with open(heuristic_file, encoding="utf-8") as f:
            sys  = {l.rstrip('\n') for l in f}

        tp = len(sys & gold)          # overlap  (correctly marked)
        fp = len(sys - gold)          # heuristic-only (over-marking)
        fn = len(gold - sys)          # rule-only      (missed)

        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        f1        = safe_div(2 * precision * recall, precision + recall)

        print("\n--- Overlap / Divergence (rule = gold, heuristic = system) ---")
        print(f"TP (overlap) : {tp:,}")
        print(f"FP           : {fp:,}")
        print(f"FN           : {fn:,}")
        print(f"Precision    : {precision:.4f}")
        print(f"Recall       : {recall:.4f}")
        print(f"F1           : {f1:.4f}")
    else:
        print("\n(Need both rule and heuristic affected files to compute "
              "overlap metrics.  Run the missing mode first.)")

    print(f"Files saved to: {perturbed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply animacy-based or heuristic case marking perturbation.")
    parser.add_argument("--mode", type=str, choices=["rule", "heuristic"], default="rule", help="Perturbation strategy")
    args = parser.parse_args()

    process_all(mode=args.mode)
