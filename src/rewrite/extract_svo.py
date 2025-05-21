import spacy
import argparse

nlp = spacy.load("en_core_web_sm")

def extract_structure(sentence):
    doc = nlp(sentence)
    result = {
        "subject": None,
        "verb": None,
        "objects": [],
        "complements": []
    }

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            verb_phrase = [w.text for w in sorted(
                [token] + [aux for aux in token.children if aux.dep_ in {"aux", "auxpass"}],
                key=lambda x: x.i
            )]
            result["verb"] = " ".join(verb_phrase)

            # 主语
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubj:pass"}:
                    result["subject"] = " ".join(w.text for w in child.subtree)

            # 直接宾语 / 从句宾语
            for child in token.children:
                if child.dep_ in {"obj", "dobj", "attr", "ccomp"}:
                    phrase = " ".join(w.text for w in child.subtree)
                    result["objects"].append(("direct/clausal", phrase))

                # 介词宾语
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            phrase = f"{child.text} " + " ".join(w.text for w in pobj.subtree)
                            result["objects"].append(("prepositional", phrase))

                # 补语（不定式、动名词）
                elif child.dep_ == "xcomp":
                    phrase = " ".join(w.text for w in child.subtree)
                    result["complements"].append(phrase)

    return result

def process_file(file_path, max_lines=10):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines[:max_lines]):
        print(f"\n🔍 第 {i+1} 句：{line}")
        result = extract_structure(line)

        print("🟦 Subject:", result["subject"] if result["subject"] else "(无)")
        print("🟨 Verb:", result["verb"] if result["verb"] else "(无)")

        if result["objects"]:
            print("🟥 Objects:")
            for kind, phrase in result["objects"]:
                print(f"    - [{kind}] {phrase}")
        else:
            print("🟥 Objects: (无)")

        if result["complements"]:
            print("🟩 Complements:")
            for comp in result["complements"]:
                print(f"    - {comp}")
        else:
            print("🟩 Complements: (无)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="结构提取器：主语 / 谓语 / 宾语 / 补语")
    parser.add_argument("--input", type=str, required=True, help="输入句子文件路径")
    parser.add_argument("--max_lines", type=int, default=10, help="最大处理句数")
    args = parser.parse_args()

    process_file(args.input, args.max_lines)