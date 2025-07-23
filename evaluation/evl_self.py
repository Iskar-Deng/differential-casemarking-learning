import os
import json
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

def load_model(checkpoint_path):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def calculate_sentence_loss(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return outputs.loss.item()

def evaluate_minipairs(jsonl_path, checkpoint_path, limit=None):
    model, tokenizer = load_model(checkpoint_path)

    preds = []
    labels = []
    correct = 0
    total = 0

    mark_errors = 0
    unmark_errors = 0

    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit]

        for line in tqdm(lines, desc="Evaluating"):
            item = json.loads(line)
            good = item["good"]
            bad = item["bad"]

            loss_good = calculate_sentence_loss(good, model, tokenizer)
            loss_bad = calculate_sentence_loss(bad, model, tokenizer)

            pred = 1 if loss_good < loss_bad else 0
            label = 1  # good 应优于 bad

            preds.append(pred)
            labels.append(label)
            total += 1

            if pred == label:
                correct += 1
            else:
                if "🄰" in good or "🄿" in good:
                    mark_errors += 1
                else:
                    unmark_errors += 1

    acc = accuracy_score(labels, preds)

    print(f"\n✅ Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"\n❌ Misclassified where good sentence had marks (🄰/🄿): {mark_errors}")
    print(f"❌ Misclassified where good sentence had no marks: {unmark_errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="Path to minimal pair .jsonl file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LM checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs to evaluate")
    args = parser.parse_args()
    evaluate_minipairs(args.jsonl, args.checkpoint, args.limit)
