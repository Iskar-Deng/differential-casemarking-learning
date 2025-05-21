import json

def extract_verb_arguments(token_list):
    """
    给定一个 token 列表（来自 spaCy 的解析），提取所有动词及其主语 / 宾语。

    返回：
        List[Dict] 格式，例如：
        [
            {"verb": "bite", "subject": "dog", "object": "man"},
            {"verb": "see", "subject": "cat", "object": "boy"},
        ]
    """
    results = []
    id_to_token = {t["id"]: t for t in token_list}
    verbs = [t for t in token_list if t["pos"] == "VERB"]

    for verb in verbs:
        verb_id = verb["id"]
        verb_lemma = verb["lemma"]

        subject = None
        obj = None

        for t in token_list:
            if t["head"] == verb_id:
                if t["dep"] in {"nsubj", "csubj", "nsubjpass"}:
                    subject = t["text"]
                elif t["dep"] in {"dobj", "obj", "dative", "attr", "xcomp", "ccomp"}:
                    obj = t["text"]

        results.append({
            "verb": verb_lemma,
            "subject": subject,
            "object": obj
        })

    return results


def process_jsonl(input_path, output_path):
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            token_list = data["tokens"]
            triples = extract_verb_arguments(token_list)
            fout.write(json.dumps(triples, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取动词-主语-宾语结构（三元组）")
    parser.add_argument("input", help="输入 JSONL 文件")
    parser.add_argument("output", help="输出 JSONL，每行一个动词结构列表")

    args = parser.parse_args()
    process_jsonl(args.input, args.output)
