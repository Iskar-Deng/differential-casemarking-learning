#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constituency parsing with spaCy + benepar (batch mode).
- 输入: $DATA_PATH/filtered 下的 *_train.txt, *_valid.txt, *_test.txt
- 输出: $DATA_PATH/parsed 下的 *_parsed.jsonl + 坏行日志
- 特性: --resume 断点续跑；tqdm 每 60 秒刷新；批量失败自动回退逐条解析并跳过坏样本
"""

import os
import sys
import json
import pathlib
import argparse
from typing import Iterator, List, Tuple

from tqdm import tqdm
import spacy
import benepar
from utils import DATA_PATH

# ---------- IO base ----------
filtered_dir = os.path.join(DATA_PATH, "filtered")
parsed_dir   = os.path.join(DATA_PATH, "parsed")
os.makedirs(parsed_dir, exist_ok=True)

txt_files = sorted([f for f in os.listdir(filtered_dir) if f.endswith(".txt")])
if not txt_files:
    print(f"[ERROR] no .txt found in {filtered_dir}", file=sys.stderr)
    sys.exit(1)


def build_nlp(model_size: str = "sm"):
    """选择 spaCy 模型并加载 benepar_en3"""
    model_name = "en_core_web_trf" if model_size == "trf" else "en_core_web_sm"
    try:
        nlp = spacy.load(model_name, disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        print(f"[INFO] spaCy model {model_name} not found. Run:", file=sys.stderr)
        print(f"       python -m spacy download {model_name}", file=sys.stderr)
        sys.exit(1)

    # benepar 模型（en3）
    try:
        benepar.load_trained_model("benepar_en3")
    except Exception:
        print("[INFO] downloading benepar_en3 ...", file=sys.stderr)
        benepar.download("benepar_en3")

    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)

    nlp.max_length = 5_000_000
    return nlp


def annotate_doc(doc, index: int):
    """doc -> JSON 可序列化结构"""
    try:
        sent_doc = list(doc.sents)[0]
    except Exception:
        return {"index": index, "tokens": [], "ptb": "", "error": "no_sentence"}
    tree = getattr(sent_doc._, "parse_string", "")
    tokens = [
        {
            "id": tok.i,
            "text": tok.text,
            "lemma": tok.lemma_,
            "pos": tok.pos_,
            "dep": tok.dep_,
            "head": tok.head.i,
            "ner": tok.ent_type_ or "O",
        }
        for tok in sent_doc
    ]
    return {"index": index, "tokens": tokens, "ptb": tree}


def clean_text(s: str) -> str:
    # 去除常见零宽/方向控制/ BOM 等字符
    return (
        s.replace("\u200b", "")
         .replace("\u200e", "")
         .replace("\ufeff", "")
         .strip()
    )


def count_lines(path: str) -> int:
    """总行数（包含空行）"""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_nonempty_lines(path: str) -> int:
    """非空行数量（与实际处理句子数一致）"""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_resume_index(jsonl_path: str) -> int:
    """
    返回已写入 jsonl 的最后 index（从 0 开始）。若文件不存在或解析失败，返回 -1。
    注意：index 与“已处理的非空行数 - 1”对齐。
    """
    try:
        if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0:
            return -1
        with open(jsonl_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = size
            chunk = 1024
            buf = b""
            while pos > 0:
                pos = max(0, pos - chunk)
                f.seek(pos)
                buf = f.read(size - pos) + buf
                if b"\n" in buf:
                    break
            last = buf.splitlines()[-1].decode("utf-8", "ignore").strip()
        if not last:
            return -1
        obj = json.loads(last)
        return int(obj.get("index", -1))
    except Exception:
        return -1


def nonempty_texts_and_idxs(path: str, start_idx: int) -> Iterator[Tuple[str, int]]:
    """按文件顺序返回 (非空行文本, index)，跳过前 start_idx 个非空行。"""
    skipped = 0
    i = start_idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if skipped < start_idx:
                skipped += 1
                continue
            yield (s, i)
            i += 1


def batched(iterable, n: int):
    """把 (text, idx) 迭代器分批次产出 list，最后一批可短。"""
    batch: List[Tuple[str, int]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def main(model_size="sm", batch_size=32, n_process=1, resume=False):
    # benepar 与 spaCy 多进程常有序列化/对象问题，建议 n_process=1
    if n_process != 1:
        print("[Warn] benepar + spaCy multiprocessing may raise serialization errors; "
              "n_process=1 is recommended.", file=sys.stderr)

    nlp = build_nlp(model_size)

    for name in txt_files:
        src_path = os.path.join(filtered_dir, name)
        dst_path = os.path.join(parsed_dir, name.replace(".txt", "_parsed.jsonl"))
        bad_path = os.path.join(parsed_dir, name.replace(".txt", "_badlines.log"))

        total_lines = count_lines(src_path)
        total_nonempty = count_nonempty_lines(src_path)

        # 断点续跑
        resume_from = -1
        if resume and os.path.exists(dst_path):
            resume_from = get_resume_index(dst_path)
            if resume_from >= 0:
                print(f"[RESUME] {name}: last index in {pathlib.Path(dst_path).name} = {resume_from}")
            else:
                print(f"[RESUME] {name}: no valid last index found, starting from scratch")

        start_idx = resume_from + 1
        remaining = max(0, total_nonempty - start_idx)

        print(f"[INFO] parsing {name} -> {pathlib.Path(dst_path).name}  "
              f"(lines={total_lines}, nonempty={total_nonempty}, start_idx={start_idx}, remaining={remaining})")

        bad = 0
        # 输出文件模式（续跑则追加）
        fout_mode = "a" if (resume and os.path.exists(dst_path) and resume_from >= 0) else "w"
        fbad_mode = "a" if (resume and os.path.exists(bad_path) and resume_from >= 0) else "w"

        with open(dst_path, fout_mode, encoding="utf-8") as fout, \
             open(bad_path, fbad_mode, encoding="utf-8") as fbad:

            # 进度条（总非空行；initial=已完成）
            pbar = tqdm(total=total_nonempty, initial=start_idx,
                        desc=f"Parsing {name}", mininterval=60, dynamic_ncols=True)

            # 逐批读取文本与索引，先尝试批量解析；失败时回退到逐条解析
            stream = nonempty_texts_and_idxs(src_path, start_idx=start_idx)
            for batch in batched(stream, batch_size):
                texts = [t for t, _ in batch]
                idxs  = [i for _, i in batch]

                try:
                    docs = list(nlp.pipe(texts, batch_size=batch_size, n_process=1))
                    assert len(docs) == len(idxs)
                    for doc, i in zip(docs, idxs):
                        try:
                            parsed = annotate_doc(doc, i)
                        except Exception as e:
                            bad += 1
                            fbad.write(f"{i}\t{type(e).__name__}: {e}\n")
                            parsed = {"index": i, "tokens": [], "ptb": "", "error": f"{type(e).__name__}: {e}"}
                        if parsed.get("error"):
                            bad += 1
                            fbad.write(f"{i}\t{parsed['error']}\n")
                        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        pbar.update(1)

                except Exception as batch_e:
                    # 批失败：逐条处理本批，逐条 try/clean_text 重试
                    for text, i in batch:
                        try:
                            doc = nlp(text)
                        except Exception:
                            try:
                                doc = nlp(clean_text(text))
                            except Exception as e2:
                                bad += 1
                                fbad.write(f"{i}\t{type(batch_e).__name__}/{type(e2).__name__}: {e2}\n")
                                fout.write(json.dumps(
                                    {"index": i, "tokens": [], "ptb": "", "error": f"{type(e2).__name__}: {e2}"},
                                    ensure_ascii=False
                                ) + "\n")
                                pbar.update(1)
                                continue

                        try:
                            parsed = annotate_doc(doc, i)
                        except Exception as e:
                            bad += 1
                            fbad.write(f"{i}\t{type(e).__name__}: {e}\n")
                            parsed = {"index": i, "tokens": [], "ptb": "", "error": f"{type(e).__name__}: {e}"}
                        if parsed.get("error"):
                            bad += 1
                            fbad.write(f"{i}\t{parsed['error']}\n")
                        fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                        pbar.update(1)

            pbar.close()

        print(f"[DONE] {dst_path}  | bad_lines={bad} -> {pathlib.Path(bad_path).name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=["sm", "trf"], default="sm",
                    help="spaCy model size: sm (fast) or trf (accurate but slow)")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for robust batching")
    ap.add_argument("--n-process", type=int, default=1,
                    help="Number of processes for nlp.pipe (benepar建议=1)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing *_parsed.jsonl if present (append mode)")
    args = ap.parse_args()

    main(model_size=args.model_size,
         batch_size=args.batch_size,
         n_process=args.n_process,
         resume=args.resume)
