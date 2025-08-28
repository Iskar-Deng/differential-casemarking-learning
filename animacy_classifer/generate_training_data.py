import os
import json
import argparse
from tqdm import tqdm
import csv
from openai import OpenAI
from utils import DATA_PATH

# 读取 OPENAI_API_KEY（需先 export OPENAI_API_KEY=...）
client = OpenAI()


def call_openai_chat(sentence: str, np: str, np_type: str, model: str = "gpt-4o") -> str | None:
    prompt = f"""Classify the animacy of the following NP. Answer with one of the categories: "human", "animal", "inanimate", or "event". Answer in a single word without quotes.

Sentence: {sentence}
{np_type} NP: {np}

Respond with one of the following categories:
- "human": refers to a person or people (e.g., the teacher, him, John)
- "animal": refers to an animal or animals (e.g., the dog, a bird)
- "inanimate": refers to a non-living thing, place, or object (e.g., the book, the school, a rock)
- "event": refers to an event, activity, idea, or state, often expressed via gerunds, clauses, or abstract nouns (e.g., running, that she left, the decision)
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2,
        )
        result = (resp.choices[0].message.content or "").strip().lower().replace('"', "")
        if result in {"human", "animal", "inanimate", "event"}:
            return result
        head = (result.split() or [""])[0]
        return head if head in {"human", "animal", "inanimate", "event"} else None
    except Exception as e:
        print(f"[ERROR - OpenAI] {type(e).__name__}: {e}")
        return None


def is_valid_structure(entry: dict) -> bool:
    return bool(entry.get("subject")) and len(entry.get("objects", [])) == 1


def count_existing_examples(csv_path: str) -> int:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0
    with open(csv_path, encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    return max(0, n)


def ensure_csv_with_header(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", "animacy"])
            writer.writeheader()


def find_input_file(structured_dir: str, suffix: str) -> str:
    """在 structured_dir 下寻找第一个以 suffix 结尾的文件"""
    for f in sorted(os.listdir(structured_dir)):
        if f.endswith(suffix):
            return os.path.join(structured_dir, f)
    raise FileNotFoundError(f"No file ending with '{suffix}' found in {structured_dir}")


def extract_training_data(max_instances: int = 2000, input_file: str = "valid_verbs.jsonl", model: str = "gpt-4o"):
    structured_dir = os.path.join(DATA_PATH, "structured")

    # 处理输入文件
    if os.path.isfile(input_file):
        input_path = input_file
        input_file = os.path.basename(input_file)
    else:
        input_path = os.path.join(structured_dir, input_file)
        if not os.path.exists(input_path):
            print(f"[WARN] {input_path} not found, trying suffix search...")
            input_path = find_input_file(structured_dir, input_file)
            input_file = os.path.basename(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    stem = os.path.splitext(input_file)[0]
    out_csv = os.path.join(DATA_PATH, f"training_data_{stem.replace('_verbs','')}.csv")

    ensure_csv_with_header(out_csv)
    existing = count_existing_examples(out_csv)
    count = existing

    print(f"[INFO] Using {input_file}")
    print(f"[INFO] Output: {out_csv}")
    print(f"[INFO] Existing rows: {existing}; Target total: {max_instances}")

    with open(input_path, encoding="utf-8") as fin, \
         open(out_csv, "a", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", "animacy"])

        for i, line in enumerate(tqdm(fin, desc=f"Generating from {input_file}")):
            if count >= max_instances:
                break
            if not line.strip():
                continue

            data = json.loads(line)
            tokens = data.get("tokens", [])
            verbs = data.get("verbs") or []
            if not tokens or not verbs:
                continue

            sentence = " ".join(tok.get("text", "") for tok in tokens)

            for entry in verbs:
                if count >= max_instances:
                    break
                if not is_valid_structure(entry):
                    continue

                subj = entry["subject"]
                obj = entry["objects"][0]
                if obj.get("dep") == "ccomp":
                    continue

                if count < max_instances:
                    r = call_openai_chat(sentence, subj.get("text", ""), "Subject", model=model)
                    if r:
                        writer.writerow({"sentence": sentence, "np": subj.get("text", ""), "np_role": "subject", "animacy": r})
                        fout.flush()
                        count += 1

                if count < max_instances:
                    r = call_openai_chat(sentence, obj.get("text", ""), "Object", model=model)
                    if r:
                        writer.writerow({"sentence": sentence, "np": obj.get("text", ""), "np_role": "object", "animacy": r})
                        fout.flush()
                        count += 1

                if count >= max_instances:
                    break

    print(f"[DONE] Added {count - existing} new examples; total now {count} rows in {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animacy training data with resume support (auto suffix search).")
    parser.add_argument("--max", type=int, default=4000, help="Total number of NP examples to generate (including existing)")
    parser.add_argument("--file", type=str, default="valid_verbs.jsonl", help="Structured verbs file (can be suffix like '_verbs.jsonl')")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    args = parser.parse_args()
    extract_training_data(max_instances=args.max, input_file=args.file, model=args.model)
