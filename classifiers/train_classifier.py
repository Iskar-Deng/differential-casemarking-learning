#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BERT classifiers for animacy / NP-type / definiteness classification.

Usage:
------
python -m classifiers.train_classifier --task animacy --epochs 10 --amp
python -m classifiers.train_classifier --task nptype --epochs 10 --amp
python -m classifiers.train_classifier --task definiteness --epochs 10 --amp
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification

from utils import DATA_PATH, MODEL_PATH

# -------------------- Configs --------------------
TASK_CONFIGS = {
    "animacy": {
        "field": "animacy",
        "labels": {"animate": 0, "inanimate": 1},
        "targets": ["animate", "inanimate"],
        "save_dir": "animacy_bert_model",
    },
    "nptype": {
        "field": "nptype",
        "labels": {"pronoun": 0, "common": 1},
        "targets": ["pronoun", "common"],
        "save_dir": "nptype_bert_model",
    },
    "definiteness": {
        "field": "definiteness",
        "labels": {"definite": 0, "indef": 1},
        "targets": ["definite", "indef"],
        "save_dir": "definiteness_bert_model",
    },
}

# -------------------- Utils --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def auto_pick_csv(task: str):
    """自动寻找 data 目录下的 training_data_*{task}*.csv"""
    for fname in os.listdir(DATA_PATH):
        if fname.endswith(".csv") and task in fname:
            return os.path.join(DATA_PATH, fname)
    raise FileNotFoundError(
        f"No training_data_*{task}*.csv found in {DATA_PATH}. Please generate training data first."
    )


# -------------------- Dataset --------------------
class NPTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int, field: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.field = field

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['sentence']} [NP] {row['np']}"
        label = int(row[self.field])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -------------------- Main --------------------
def main(args):
    set_seed(args.seed)
    cfg = TASK_CONFIGS[args.task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选 CSV
    csv_path = args.csv or auto_pick_csv(args.task)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    print(f"[INFO] Using CSV: {csv_path}")

    # 读取 & 清洗
    df = pd.read_csv(csv_path)
    need_cols = {"sentence", "np", "np_role", cfg["field"]}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df[df[cfg["field"]].isin(cfg["labels"].keys())].copy()
    df.dropna(subset=["sentence", "np", cfg["field"]], inplace=True)
    df.drop_duplicates(subset=["sentence", "np", "np_role", cfg["field"]], inplace=True)
    df[cfg["field"]] = df[cfg["field"]].map(cfg["labels"])

    if len(df) < 100:
        print(f"[WARN] Very small dataset (n={len(df)}). Consider generating more.")

    # 类别分布
    dist = Counter(df[cfg["field"]].tolist())
    inv_label = {v: k for k, v in cfg["labels"].items()}
    pretty = {inv_label[k]: v for k, v in dist.items()}
    print(f"[INFO] Label distribution: {pretty}")

    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[cfg["field"]] if len(dist) > 1 else None,
    )
    print(f"[INFO] Train size={len(train_df)}, Test size={len(test_df)}")

    # Tokenizer & Dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = NPTaskDataset(train_df, tokenizer, args.max_length, cfg["field"])
    test_ds = NPTaskDataset(test_df, tokenizer, args.max_length, cfg["field"])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(cfg["labels"])
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Training
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[INFO] Epoch {epoch} avg loss: {avg_loss:.4f}")

    # 保存
    save_dir = os.path.join(MODEL_PATH, cfg["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Model saved to: {save_dir}")

    # 评测
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=cfg["targets"],
        digits=4
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["animacy", "nptype", "definiteness"], required=True, help="Task type")
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to training CSV. Default: auto-pick training_data_*<task>*.csv under DATA_PATH.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    args = ap.parse_args()
    main(args)
