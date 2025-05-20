"""
inspect_sentences.py

用于查看过滤后的语料统计信息和示例句子，并可保存示例句至文件。

使用方法：
    python inspect_sentences.py --input data/filtered/babylm_short.txt --examples 5 --save data/debug/examples.txt
"""

import argparse
import os

def inspect_sentences(input_path, example_count=5, save_path=None):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    lengths = [len(line.split()) for line in lines]

    print("📊 语料统计信息")
    print("-" * 30)
    print(f"句子总数：{len(lines)}")
    print(f"最短句长度：{min(lengths)}")
    print(f"最长句长度：{max(lengths)}")
    print(f"平均句长：{sum(lengths) / len(lengths):.2f} 词")

    print("\n📌 示例句子（前 {} 条）".format(example_count))
    print("-" * 30)
    examples = lines[:example_count]
    for i, line in enumerate(examples):
        print(f"{i+1}. {line}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f_out:
            for line in examples:
                f_out.write(line + "\n")
        print(f"\n✅ 示例句已保存到: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计过滤后的语料信息")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径")
    parser.add_argument("--examples", type=int, default=5, help="要显示的示例句子数量")
    parser.add_argument("--save", type=str, help="可选：将示例句保存到此文件")
    args = parser.parse_args()

    inspect_sentences(args.input, args.examples, args.save)
