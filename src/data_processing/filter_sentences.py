import argparse
import os

def filter_babylm_sentences(input_path, output_path, min_len=5, max_len=15):
    """
    过滤 BabyLM 语料中长度合适的句子。
    删除以 _book_title_ 或 chapter 开头的行。
    参数：
        input_path: str，原始文本文件路径
        output_path: str，输出文本文件路径
        min_len: int，最小词数
        max_len: int，最大词数
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("_book_title_") or line.lower().startswith("chapter") or line.lower().startswith("-lcb-"):
            continue
        tokens = line.split()
        if min_len <= len(tokens) <= max_len:
            filtered.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for sent in filtered:
            f_out.write(sent + "\n")

    print(f"✅ 保留句子数量：{len(filtered)}，已写入 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤合适长度的句子")
    parser.add_argument("--input", type=str, required=True, help="输入文本文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文本文件路径")
    parser.add_argument("--min_len", type=int, default=5, help="最小词数")
    parser.add_argument("--max_len", type=int, default=15, help="最大词数")
    args = parser.parse_args()

    filter_babylm_sentences(args.input, args.output, args.min_len, args.max_len)
