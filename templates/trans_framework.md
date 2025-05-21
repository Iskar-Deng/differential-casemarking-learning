# transformation.py 结构设计

## Parse_and_dump
```bash
src/analysis/parse_and_dump.py

# 安装依赖
pip install spacy benepar torch transformers tqdm protobuf==3.20.*

# 下载 spaCy transformer 模型
python -m spacy download en_core_web_trf

# 下载 benepar 模型（进入 Python shell）
python -c "import benepar; benepar.download('benepar_en_bert_base')"

# 运行主脚本
python3 src/analysis/parse_and_dump.py
```