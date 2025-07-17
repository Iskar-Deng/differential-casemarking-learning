import os
import json
import random

# === 参数设置 ===
input_txt = "/home/dengh/workspace/relational-casemarking-learning/data/perturbed/rule/combined.txt"         # 每行一句
output_dir = "/home/dengh/workspace/relational-casemarking-learning/data/perturbed/rule/train"
name = "rule"
validation_ratio = 0.05

# === 读取所有非空行 ===
with open(input_txt, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"📘 总行数（样本）：{len(lines)}")

# === 打乱并拆分 ===
random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * (1 - validation_ratio))
train_data = lines[:split_idx]
val_data = lines[split_idx:]

# === 转成 {"text": "..."} 格式
train_data = [{"text": line} for line in train_data]
val_data = [{"text": line} for line in val_data]

# === 输出路径 ===
os.makedirs(output_dir, exist_ok=True)
train_path = os.path.join(output_dir, f"{name}.train.jsonl")
val_path = os.path.join(output_dir, f"{name}.validation.jsonl")

# === 写入 JSONL 文件 ===
with open(train_path, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 拆分完成：训练集 {len(train_data)} 行，验证集 {len(val_data)} 行")
print(f"✔ Train 文件：{train_path}")
print(f"✔ Val 文件：  {val_path}")
