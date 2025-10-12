#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a causal LM from a YAML config file.

Usage:
  python -m training.train_lm_v2 \
    --config configs/independent_Anone_Panimate.yaml
"""

import os
import csv
import json
import math
import time
import argparse
import tempfile
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from utils import CHECKPOINT_PATH, CACHE_PATH, AGENT_MARK, PATIENT_MARK


# -------------------------
# 环境设置
# -------------------------
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
print(f"[Info] Set PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

torch.cuda.empty_cache()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# -------------------------
# Helpers
# -------------------------

def _expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


def _load_and_oversample(paths: List[str], oversample_plan: Optional[Dict[str, float]] = None, keep_tmp: bool = False):
    """动态 oversample（整数 + 小数部分）"""
    all_paths = []
    tmp_files = []

    if oversample_plan is None:
        return paths

    for p in paths:
        p_str = str(p)
        ratio = 1.0
        if "affected" in p_str:
            ratio = oversample_plan.get("affected", 1.0)
        elif "unaffected" in p_str:
            ratio = oversample_plan.get("unaffected", 1.0)

        n_full = int(ratio)
        frac = ratio - n_full
        all_paths.extend([p_str] * n_full)

        if frac > 1e-6:
            with open(p_str, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f if x.strip()]
            n_take = max(1, int(math.ceil(len(lines) * frac)))
            random.seed(42)
            sampled = random.sample(lines, n_take)
            tmpfile = tempfile.NamedTemporaryFile(
                mode="w", delete=False, encoding="utf-8", suffix="_partial.txt"
            )
            for line in sampled:
                tmpfile.write(line + "\n")
            tmpfile.close()
            tmp_files.append(tmpfile.name)
            print(f"[Info] fractional oversample: {p_str} +{frac:.2f}x ({n_take} lines) → {tmpfile.name}")
            all_paths.append(tmpfile.name)

    if not keep_tmp:
        import atexit
        def _cleanup():
            for f in tmp_files:
                try:
                    os.remove(f)
                    print(f"[Clean] removed tmp oversample file: {f}")
                except Exception:
                    pass
        atexit.register(_cleanup)

    return all_paths


def _build_dataset(files: List[str]):
    if not files:
        return None
    ds_list = []
    for f in files:
        f = _expand(f)
        if f.endswith(".txt"):
            ds = load_dataset("text", data_files=f, split="train")
        elif f.endswith(".jsonl") or f.endswith(".json"):
            ds = load_dataset("json", data_files=f, split="train")
            if "text" not in ds.column_names:
                raise ValueError(f"{f} 缺少字段 'text'")
        else:
            raise ValueError(f"不支持的文件类型：{f}")
        ds_list.append(ds)
    return concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]


def _group_texts(examples: Dict[str, list], block_size: int):
    keys = [k for k, v in examples.items() if isinstance(v, list) and v and isinstance(v[0], list)]
    concatenated = {k: sum(examples[k], []) for k in keys}
    total_len = len(concatenated.get("input_ids", []))
    total_len = (total_len // block_size) * block_size
    result = {k: [t[i:i + block_size] for i in range(0, total_len, block_size)] for k, t in concatenated.items()}
    if "input_ids" not in result:
        result["input_ids"] = []
    result["labels"] = result["input_ids"].copy()
    return result


def _find_last_checkpoint(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    cks = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    if not cks:
        return None
    cks.sort(key=lambda p: int(p.name.split("-")[-1]))
    return cks[-1]


def _count_total_tokens(tokenized_ds, batch_size: int = 1000) -> int:
    total = 0
    for batch in tokenized_ds.iter(batch_size=batch_size):
        total += sum(len(x) for x in batch["input_ids"])
    return total


# -------------------------
# Callbacks
# -------------------------

class ValidPPLLogger(TrainerCallback):
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._csv_path_str = str(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with open(self._csv_path_str, "w", newline="") as f:
                csv.writer(f).writerow(["step", "eval_loss", "ppl", "wall_time"])
        self.tb_writer = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" not in metrics:
            return
        eval_loss = float(metrics["eval_loss"])
        ppl = float(math.exp(eval_loss))
        step = int(state.global_step)
        with open(self._csv_path_str, "a", newline="") as f:
            csv.writer(f).writerow([step, eval_loss, ppl, time.time()])
        if state.is_world_process_zero:
            print(f"[Eval] step={step} eval_loss={eval_loss:.6f} ppl={ppl:.3f}")


class ThroughputLogger(TrainerCallback):
    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.t0 is None:
            return
        dt = time.time() - self.t0
        tps = self.tokens_per_step / max(dt, 1e-6)
        if state.is_world_process_zero and (state.global_step % max(1, args.logging_steps) == 0):
            print(f"[Speed] step={state.global_step} ~{tps:,.0f} tokens/s")


class DynamicCheckpointSaver(TrainerCallback):
    def __init__(self, freq_list, output_dir: Path):
        self.freq_list = freq_list
        self.output_dir = output_dir

    def _get_interval(self, step: int) -> Optional[int]:
        for interval, max_step in self.freq_list:
            if step <= max_step:
                return interval
        return None

    def on_step_end(self, args, state, control, **kwargs):
        interval = self._get_interval(state.global_step)
        if interval is not None and state.global_step % interval == 0:
            control.should_save = True
        return control


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    run_id = cfg.get("run_id", cfg_path.stem)
    data_cfg: Dict[str, Any] = cfg.get("data", {})
    train_files: List[str] = data_cfg.get("train_files", [])
    eval_files: List[str] = data_cfg.get("eval_files", [])
    oversample_plan = data_cfg.get("oversample_plan", None)

    if oversample_plan:
        print(f"[Info] Oversample plan detected: {oversample_plan}")
        train_files = _load_and_oversample(train_files, oversample_plan)

    run_dir = Path(cfg["artifacts"]["run_dir"]).expanduser().resolve()
    cache_dir = Path(cfg["artifacts"]["cache_dir"]).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

    targs = dict(cfg.get("training_arguments", {}))
    model_name = cfg.get("model_name", "gpt2")
    block_size = int(cfg.get("block_size", 1024))
    seed = int(cfg.get("seed", 42))
    effective_bsz = cfg.get("effective_bsz", None)

    print(f"[Info] ===== Training Config =====")
    print(f"run_id={run_id}")
    print(f"model={model_name}")
    print(f"train_files={len(train_files)} | eval_files={len(eval_files)}")
    print(f"output_dir={run_dir}")
    print("=================================")

    if effective_bsz is not None:
        pbsz = int(targs.get("per_device_train_batch_size", 1))
        gas = math.ceil(float(effective_bsz) / max(pbsz, 1))
        targs["gradient_accumulation_steps"] = int(gas)

    train_ds = _build_dataset(train_files)
    eval_ds = _build_dataset(eval_files) if eval_files else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    to_add = [m for m in [AGENT_MARK, PATIENT_MARK] if m not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        print(f"[Info] Added special tokens: {to_add}")

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir))
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    if bool(targs.get("gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tok_fn(batch):
        return tokenizer(batch["text"], return_attention_mask=False)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names, num_proc=1)
    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=eval_ds.column_names, num_proc=1) if eval_ds else None

    raw_train_tokens = _count_total_tokens(train_tok)
    usable_train_tokens = (raw_train_tokens // block_size) * block_size

    train_tok = train_tok.map(lambda ex: _group_texts(ex, block_size), batched=True, batch_size=1000, num_proc=1)
    if eval_tok:
        eval_tok = eval_tok.map(lambda ex: _group_texts(ex, block_size), batched=True, batch_size=1000, num_proc=1)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(output_dir=str(run_dir), seed=seed, disable_tqdm=True, **targs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if "checkpoint_frequency" in cfg:
        trainer.add_callback(DynamicCheckpointSaver(cfg["checkpoint_frequency"], run_dir))

    if eval_tok is not None:
        csv_path = run_dir / "valid_metrics.csv"
        trainer.add_callback(ValidPPLLogger(csv_path))

    tokens_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * block_size
    trainer.add_callback(ThroughputLogger(tokens_per_step))

    resume_from = None
    if cfg.get("resume", False):
        ckpt = _find_last_checkpoint(run_dir)
        if ckpt:
            resume_from = str(ckpt)

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model()
    tokenizer.save_pretrained(run_dir)

    print(f"[OK] Training complete. Checkpoints at {run_dir}")


if __name__ == "__main__":
    main()
