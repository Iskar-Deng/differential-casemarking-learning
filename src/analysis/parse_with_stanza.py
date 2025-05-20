"""
parse_with_stanza.py

读取一个句子文件，使用 Stanza 分析句法结构并打印结果。

使用示例：
    python parse_with_stanza.py --input data/filtered/cbt_clean.txt --max_lines 10
"""

import stanza
import argparse

def analyze_sentences(file_path, max_lines=10):
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", use_gpu=False)

    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    for i, sent in enumerate(sentences[:max_lines]):
        print(f"\n🔍 第 {i+1} 句：{sent}")
        doc = nlp(sent)
        for sentence in doc.sentences:
            print(f"{'Index':<5} {'Word':<15} {'Lemma':<15} {'POS':<6} {'Head':<5} {'Dep'}")
            for word in sentence.words:
                print(f"{word.id:<5} {word.text:<15} {word.lemma:<15} {word.upos:<6} {word.head:<5} {word.deprel}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Stanza 对文本文件进行句法分析")
    parser.add_argument("--input", type=str, required=True, help="输入的句子文本文件路径")
    parser.add_argument("--max_lines", type=int, default=10, help="要分析的最大句子数")
    args = parser.parse_args()

    analyze_sentences(args.input, args.max_lines)
