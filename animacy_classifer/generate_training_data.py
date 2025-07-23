import os
import json
import argparse
from tqdm import tqdm
import csv
from openai import OpenAI
from utils import DATA_PATH

client = OpenAI()

def call_openai_chat(sentence, np, np_type):
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower().replace('"', '')
        print(f"[DEBUG] OpenAI response: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] {sentence}")
        print(e)
        return None

def is_valid_structure(entry):
    return entry.get("subject") and len(entry.get("objects", [])) == 1

def count_existing_examples(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, encoding="utf-8") as f:
        return sum(1 for line in f) - 1  # minus header

def extract_training_data(max_instances=2000):
    structured_dir = os.path.join(DATA_PATH, "structured")
    jsonl_files = [f for f in os.listdir(structured_dir) if f.endswith("_verbs.jsonl")]
    if len(jsonl_files) != 1:
        raise ValueError(f"Expected one _verbs.jsonl file in {structured_dir}, found: {jsonl_files}")

    input_file = jsonl_files[0]
    input_path = os.path.join(structured_dir, input_file)
    out_csv = os.path.join(DATA_PATH, "training_data_split.csv")

    existing = count_existing_examples(out_csv)
    count = existing
    print(f"[INFO] Existing examples: {existing}, generating up to {max_instances}")

    with open(input_path, encoding="utf-8") as fin, \
         open(out_csv, "a", newline="", encoding="utf-8") as fout:

        writer = csv.DictWriter(fout, fieldnames=["sentence", "np", "np_role", "animacy"])
        if existing == 0:
            writer.writeheader()

        for i, line in enumerate(tqdm(fin, desc="Generating training examples")):
            if i >= 100000 or count >= max_instances:
                break

            data = json.loads(line)
            tokens = data.get("tokens", [])
            if not data.get("verbs"):
                continue

            sentence = " ".join(tok["text"] for tok in tokens)
            for entry in data["verbs"]:
                if not is_valid_structure(entry):
                    continue

                subj = entry["subject"]
                obj = entry["objects"][0]
                if obj["dep"] == "ccomp":
                    continue

                if count >= max_instances:
                    break

                subj_result = call_openai_chat(sentence, subj["text"], "Subject")
                if subj_result:
                    writer.writerow({
                        "sentence": sentence,
                        "np": subj["text"],
                        "np_role": "subject",
                        "animacy": subj_result
                    })
                    fout.flush()
                    count += 1
                    if count >= max_instances:
                        break

                if count >= max_instances:
                    break

                obj_result = call_openai_chat(sentence, obj["text"], "Object")
                if obj_result:
                    writer.writerow({
                        "sentence": sentence,
                        "np": obj["text"],
                        "np_role": "object",
                        "animacy": obj_result
                    })
                    fout.flush()
                    count += 1
                    if count >= max_instances:
                        break

    print(f"Saved {count - existing} new examples to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue generating animacy classification training data with resume support.")
    parser.add_argument("--max", type=int, default=4000, help="Total number of NP examples to generate (including existing)")
    args = parser.parse_args()
    extract_training_data(max_instances=args.max)
