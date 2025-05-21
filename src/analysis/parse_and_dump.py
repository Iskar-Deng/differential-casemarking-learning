import spacy, json, pathlib, tqdm, benepar
benepar.download('benepar_en_bert_base')

# ---------- 1. 加载 spaCy 模型 ----------
nlp = spacy.load("en_core_web_trf")  # 自动检测 GPU
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_en_bert_base"}, last=True)

# ---------- 2. 解析函数 ----------
def annotate(sent: str):
    doc = nlp(sent.strip())
    sent_doc = list(doc.sents)[0]  # 把 doc 转换成句子列表再取第一个句子
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

# ---------- 3. 批量处理文件 ----------
src = pathlib.Path("debugs/art_examples.txt")
dst = pathlib.Path("debugs/parsed.jsonl")

with src.open() as fin, dst.open("w") as fout:
    lines = fin.readlines()
    for line in tqdm.tqdm(lines, total=len(lines)):
        if line.strip():
            parsed = annotate(line)
            fout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
