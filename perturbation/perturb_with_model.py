import os
import json
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import DATA_PATH, MODEL_PATH, AGENT_MARK, PATIENT_MARK
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to the fine-tuned BERT model for animacy classification
model_path = os.path.join(MODEL_PATH, "animacy_bert_model")

# Label mapping and animacy ranking (higher is more animate)
label_map = {0: "human", 1: "animal", 2: "inanimate", 3: "event"}
animacy_rank = {"human": 3, "animal": 2, "inanimate": 1, "event": 0}

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def predict_animacy(sentence, np_text):
    """
    Predict the animacy category for a noun phrase (np_text) within a sentence.

    We concatenate the sentence and the NP with a special marker "[NP]" so the
    classifier can leverage context. Returns one of: {"human","animal","inanimate","event"}.
    """
    text = f"{sentence} [NP] {np_text}"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]


def compare_animacy(subj_cat, obj_cat):
    """
    Compare animacy levels between subject and object categories.

    Returns:
        "higher" if subj > obj,
        "lower"  if subj < obj,
        "equal"  if subj == obj.
    """
    if animacy_rank[subj_cat] > animacy_rank[obj_cat]:
        return "higher"
    elif animacy_rank[subj_cat] < animacy_rank[obj_cat]:
        return "lower"
    else:
        return "equal"


def should_perturb_rule(subj_cat, obj_cat):
    """
    Rule-based decision:
    Perturb if subject animacy is <= object animacy (i.e., 'lower' or 'equal').
    """
    return compare_animacy(subj_cat, obj_cat) in {"lower", "equal"}


def should_perturb_heuristic(subj_cat):
    """
    Heuristic decision:
    Perturb if the subject is classified as 'human', ignoring object category.
    """
    return subj_cat == "human"


def is_valid_structure(entry):
    """
    Check if a verb entry has exactly one subject and one object span.

    Expected format:
        entry["subject"] -> span dict
        entry["objects"] -> list with exactly one span dict
    """
    return entry.get("subject") and len(entry.get("objects", [])) == 1


def apply_spans_to_tokens(tokens, spans):
    """
    Apply span replacements to the token sequence and rebuild the sentence.

    This function replaces the token range [start, end] with span["text"] for
    each span, skipping inner tokens to avoid duplication. It assumes that spans
    are non-overlapping and aligned to token indices (inclusive boundaries).

    NOTE: If overlapping or nested spans exist, this simple approach may need
    additional handling (e.g., pre-merge spans before applying).
    """
    # Map from span start index -> replacement text
    span_map = {span["span"][0]: span["text"] for span in spans}

    # Collect token indices to skip because they are covered by a span interior
    skip_ids = set()
    for span in spans:
        start, end = span["span"]
        # skip tokens strictly after 'start' up to and including 'end'
        skip_ids.update(range(start + 1, end + 1))

    # Rebuild the sentence by scanning tokens
    output = []
    for i, tok in enumerate(tokens):
        if i in skip_ids:
            continue
        elif i in span_map:
            output.append(span_map[i])
        else:
            output.append(tok["text"])
    return " ".join(output)


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
def process_all(mode="rule", strategy="A+P"):
    """
    Process a structured jsonl file and produce three outputs:
      - affected.txt:    sentences where text was actually changed (marks added)
      - unaffected.txt:  sentences that had valid structures but were not changed
      - invalid.txt:     sentences without usable verb structures

    Semantics:
      - "full":     unconditionally perturb ALL valid entries and add BOTH tags (A+P).
                    Ignores 'mode' and does NOT call the classifier (faster).
      - "A+P":      perturb entries decided by 'mode', add both tags.
      - "A_only":   perturb entries decided by 'mode', add AGENT tag only.
      - "P_only":   perturb entries decided by 'mode', add PATIENT tag only.
      - "none":     DO NOT add any tags. We still compute the decision by 'mode'
                    (for analysis) but we do not change the sentence nor count it
                    as affected. We print how many sentences "would have been
                    affected" under this no-mark setting.
    """
    assert strategy in {"A+P", "A_only", "P_only", "none", "full"}

    structured_dir = os.path.join(DATA_PATH, "structured")
    perturbed_dir = os.path.join(DATA_PATH, f"perturbed_model/{mode}_{strategy}")
    os.makedirs(perturbed_dir, exist_ok=True)

    # Find the single *_verbs.jsonl file
    jsonl_files = [f for f in os.listdir(structured_dir) if f.endswith("_verbs.jsonl")]
    if len(jsonl_files) != 1:
        raise ValueError(f"Expected one _verbs.jsonl file, found: {jsonl_files}")

    input_path = os.path.join(structured_dir, jsonl_files[0])

    # Build a clean prefix for output file names
    # Example: foo_verbs.jsonl -> prefix "foo"
    prefix = jsonl_files[0].rsplit("_verbs.jsonl", 1)[0]

    out_affected = os.path.join(perturbed_dir, f"{prefix}_affected.txt")
    out_unaffected = os.path.join(perturbed_dir, f"{prefix}_unaffected.txt")
    out_invalid = os.path.join(perturbed_dir, f"{prefix}_invalid.txt")

    affected_lines, unaffected_lines, invalid_lines = [], [], []

    # For "none", we also report how many sentences WOULD have been affected.
    would_affect_sentence_count = 0

    with open(input_path, encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, desc=f"Processing {mode}-{strategy}", total=len(lines)):
            data = json.loads(line)
            tokens = data.get("tokens", [])
            sentence = " ".join(tok["text"] for tok in tokens)

            if not data.get("verbs"):
                invalid_lines.append(sentence)
                continue

            # Special case: full strategy — unconditional perturbation
            if strategy == "full":
                spans = []
                valid = False
                for entry in data["verbs"]:
                    if not is_valid_structure(entry):
                        continue
                    valid = True
                    subj = dict(entry["subject"])
                    obj = dict(entry["objects"][0])
                    subj["text"] += f" {AGENT_MARK}"
                    obj["text"] += f" {PATIENT_MARK}"
                    spans.extend([subj, obj])
                if valid:
                    new_sent = apply_spans_to_tokens(tokens, spans)
                    affected_lines.append(new_sent)
                else:
                    invalid_lines.append(sentence)
                continue  # important: skip the rest of the loop for full
                    # span edits to apply if we perturb

            for entry in data["verbs"]:
                if not is_valid_structure(entry):
                    continue
                has_valid = True

                subj = entry["subject"]
                obj = entry["objects"][0]

                # Determine marking plan for this entry based on strategy
                if strategy == "full":
                    # Unconditional perturb with both marks; no inference needed
                    decision = True
                    mark_agent = True
                    mark_patient = True
                    subj_cat = obj_cat = None  # not used
                else:
                    # We need a decision per 'mode'; hence we may infer animacy
                    subj_cat = predict_animacy(sentence, subj["text"])
                    obj_cat = predict_animacy(sentence, obj["text"])

                    if mode == "rule":
                        decision = should_perturb_rule(subj_cat, obj_cat)
                    elif mode == "heuristic":
                        decision = should_perturb_heuristic(subj_cat)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    # Translate 'strategy' into marking actions
                    if strategy == "A+P":
                        mark_agent, mark_patient = True, True
                    elif strategy == "A_only":
                        mark_agent, mark_patient = True, False
                    elif strategy == "P_only":
                        mark_agent, mark_patient = False, True
                    elif strategy == "none":
                        mark_agent, mark_patient = False, False
                    else:
                        # Should not happen; 'full' handled above
                        mark_agent, mark_patient = False, False

                # Apply decision for this entry
                if decision:
                    if mark_agent or mark_patient:
                        has_affected = True
                        if mark_agent:
                            subj_marked = dict(subj)
                            subj_marked["text"] = f"{subj['text']} {AGENT_MARK}"
                            spans.append(subj_marked)
                        if mark_patient:
                            obj_marked = dict(obj)
                            obj_marked["text"] = f"{obj['text']} {PATIENT_MARK}"
                            spans.append(obj_marked)
                    else:
                        # strategy == "none": record that this sentence would've been affected
                        would_affect_flag = True

            # After scanning all entries for the sentence, decide where to write it
            if has_affected:
                new_sent = apply_spans_to_tokens(tokens, spans)
                affected_lines.append(new_sent)
            elif has_valid:
                # No text changes. If there was at least one positive decision under 'none',
                # count it for reporting but still output the original sentence as unaffected.
                if would_affect_flag and strategy == "none":
                    would_affect_sentence_count += 1
                unaffected_lines.append(sentence)
            else:
                invalid_lines.append(sentence)

    # Write outputs
    with open(out_affected, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in affected_lines)
    with open(out_unaffected, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in unaffected_lines)
    with open(out_invalid, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in invalid_lines)

    # Summary
    print(f"Mode: {mode} | Strategy: {strategy}")
    print(f"Affected (text changed): {len(affected_lines)}")
    print(f"Unaffected (text unchanged): {len(unaffected_lines)}")
    print(f"Invalid: {len(invalid_lines)}")
    if strategy == "none":
        print(f"Would-affect (no-mark mode): {would_affect_sentence_count} "
              f"(sentences decided positive but left unchanged)")
    print(f"Saved to {perturbed_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["rule", "heuristic"], default="rule")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["A+P", "A_only", "P_only", "none", "full"],
        default="A+P"
    )
    args = parser.parse_args()
    process_all(mode=args.mode, strategy=args.strategy)
