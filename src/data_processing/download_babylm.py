# src/utils/download_babylm.py

from datasets import load_dataset
import os

def download_babylm_10M(save_path="data/raw/babylm_10M.txt"):
    # 加载数据集
    dataset = load_dataset("nilq/babylm-10M", split="train")

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 写入文本
    with open(save_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(example["text"] + "\n")

    print(f"✅ 数据集已保存到: {save_path}")


if __name__ == "__main__":
    download_babylm_10M()
