import json
from collections import defaultdict

def build_children_index(tokens):
    children = defaultdict(list)
    for t in tokens:
        children[t["head"]].append(t)
    return children

def extract_np_span(token_id, tokens, children):
    """提取完整名词短语并返回其 span"""
    head = tokens[token_id]
    span_tokens = [head]

    for child in children[token_id]:
        if child["dep"] in {"det", "amod", "compound", "poss", "nmod"}:
            span_tokens.append(child)

    span_tokens.sort(key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    phrase = " ".join(t["text"] for t in span_tokens)
    return {
        "text": phrase,
        "span": [min(ids), max(ids)]
    }

def extract_clause_span(token_id, tokens, children):
    """提取完整从句子树并返回其 span"""
    visited = set()
    to_visit = [token_id]

    while to_visit:
        current = to_visit.pop()
        visited.add(current)
        for child in children[current]:
            if child["id"] not in visited:
                to_visit.append(child["id"])

    span_tokens = sorted([tokens[i] for i in visited], key=lambda x: x["id"])
    ids = [t["id"] for t in span_tokens]
    phrase = " ".join(t["text"] for t in span_tokens)
    return {
        "text": phrase,
        "span": [min(ids), max(ids)]
    }

def extract_verb_arguments(tokens):
    results = []
    id_to_token = {t["id"]: t for t in tokens}
    children = build_children_index(tokens)

    verbs = [t for t in tokens if t["pos"] == "VERB"]

    for verb in verbs:
        verb_id = verb["id"]
        result = {
            "verb": verb["lemma"],
            "verb_id": verb_id,
            "subject": None,
            "objects": []   # <- 所有宾语（直接 + 从句 + 介词宾语）
        }

        for t in children[verb_id]:
            if t["dep"] in {"nsubj", "nsubjpass", "csubj"}:
                result["subject"] = extract_np_span(t["id"], id_to_token, children)

            elif t["dep"] in {"dobj", "obj", "attr", "dative"}:
                result["objects"].append({
                    **extract_np_span(t["id"], id_to_token, children),
                    "dep": t["dep"]
                })

            elif t["dep"] in {"xcomp", "ccomp"}:
                result["objects"].append({
                    **extract_clause_span(t["id"], id_to_token, children),
                    "dep": t["dep"]
                })

            elif t["dep"] == "prep":
                prep_token = t
                subtree_tokens = [prep_token]

                # 找 pobj + 其修饰语
                for child in children[prep_token["id"]]:
                    if child["dep"] == "pobj":
                        subtree_tokens.append(child)
                        # 包含修饰词（det, amod, compound）
                        for grandchild in children[child["id"]]:
                            if grandchild["dep"] in {"det", "amod", "compound", "poss"}:
                                subtree_tokens.append(grandchild)

                if len(subtree_tokens) > 1:
                    ids = [tok["id"] for tok in subtree_tokens]
                    span = sorted(subtree_tokens, key=lambda x: x["id"])
                    result["objects"].append({
                        "text": " ".join(tok["text"] for tok in span),
                        "span": [min(ids), max(ids)],
                        "dep": "prep+pobj"
                    })


        results.append(result)

    return results

def process_jsonl(input_path, output_path):
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            tokens = data["tokens"]
            extracted = extract_verb_arguments(tokens)
            fout.write(json.dumps(extracted, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取动词结构（含主语/宾语/介词宾语 + span）")
    parser.add_argument("input", help="输入 JSONL 文件（每行一个句子的 tokens）")
    parser.add_argument("output", help="输出 JSONL 文件（每行为动词结构列表）")
    args = parser.parse_args()

    process_jsonl(args.input, args.output)
