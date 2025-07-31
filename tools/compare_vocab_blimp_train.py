import json
import argparse
import re
from collections import Counter

def tokenize(text):
    # 简单tokenizer：保留字母和数字，其余替换为空格
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return text.strip().split()

def build_vocab_from_train(train_path):
    vocab = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            tokens = tokenize(text)
            vocab.update(tokens)
    return vocab

def find_oov_words(blimp_path, train_vocab):
    counter = Counter()
    with open(blimp_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for field in ["sentence_good", "sentence_bad"]:
                sent = obj.get(field, "")
                tokens = tokenize(sent)
                for tok in tokens:
                    if tok not in train_vocab:
                        counter[tok] += 1
    return counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training dataset (.jsonl)")
    parser.add_argument("--blimp", type=str, required=True, help="Path to BLiMP file (.jsonl)")
    parser.add_argument("--topk", type=int, default=100, help="Show top-K OOV words")
    args = parser.parse_args()

    print("📥 Building training vocab...")
    train_vocab = build_vocab_from_train(args.train)
    print(f"✅ Training vocab size: {len(train_vocab)}")

    print("🔍 Searching for OOV words in BLiMP...")
    oov_counter = find_oov_words(args.blimp, train_vocab)
    total_oov = len(oov_counter)
    print(f"⚠️ Total unique OOV words in BLiMP: {total_oov}")

    for word, freq in oov_counter.most_common(args.topk):
        print(f"{word}\t{freq}")
