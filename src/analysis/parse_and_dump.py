import spacy, json, pathlib, tqdm
from benepar import BeneparComponent  # ✅ 推荐方式导入
import benepar

# ---------- 1. 下载并加载 benepar 模型 --------

# ---------- 2. 加载 spaCy 模型 ----------
nlp = spacy.load("en_core_web_trf")  # 自动检测 GPU
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"}, last=True)

# ---------- 3. 单句分析函数 ----------
def annotate(sent: str):
    doc = nlp(sent.strip())
    sent_doc = list(doc.sents)[0]  # 取第一个句子
    tree = sent_doc._.parse_string
    tokens = [
        {
            "id": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.i,
            "ner": token.ent_type_ or "O"
        }
        for token in sent_doc
    ]
    return {"tokens": tokens, "ptb": tree}

# ---------- 4. 批量处理文件 ----------
src = pathlib.Path("debugs/cbt_examples.txt")
dst = pathlib.Path("debugs/cbt_parsed.jsonl")

with src.open() as fin, dst.open("w") as fout:
    lines = fin.readlines()
    for line in tqdm.tqdm(lines, total=len(lines)):
        if line.strip():
            parsed = annotate(line)
            fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
