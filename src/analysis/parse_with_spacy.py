"""
parse_with_spacy.py

读取一个句子文件，使用 spaCy 分析句法结构并打印结果。

使用示例：
    python parse_with_spacy.py --input data/filtered/cbt_clean.txt --max_lines 10
"""

import spacy
import argparse

def analyze_sentences(file_path, max_lines=10):
    nlp = spacy.load("en_core_web_sm")

    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    for i, sent in enumerate(sentences[:max_lines]):
        print(f"\n🔍 第 {i+1} 句：{sent}")
        doc = nlp(sent)
        print(f"{'Index':<5} {'Word':<15} {'Lemma':<15} {'POS':<6} {'Head':<5} {'Dep'}")
        for token in doc:
            print(f"{token.i:<5} {token.text:<15} {token.lemma_:<15} {token.pos_:<6} {token.head.i:<5} {token.dep_}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 spaCy 对文本文件进行句法分析")
    parser.add_argument("--input", type=str, required=True, help="输入的句子文本文件路径")
    parser.add_argument("--max_lines", type=int, default=10, help="要分析的最大句子数")
    args = parser.parse_args()

    analyze_sentences(args.input, args.max_lines)
